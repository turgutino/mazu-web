import queue
import threading
import uuid
from pathlib import Path

from mazu.action_log.store import ActionLogStore, record_action
from mazu.agent.compaction import compact_if_needed, extract_memories_on_compaction, force_compact
from mazu.agent.context import build_system_prompt
from mazu.agent.registry_factory import build_registry
from mazu.agent.session import finalize_session
from mazu.chat.store import ChatStore
from mazu.config import load_config, router_suggestions_enabled
from mazu.llm.client import _split_model, default_model, extract_cost_tokens, run_turn_stream, summarize_usage
from mazu.llm.errors import MazuAPIError, MazuContextLengthError
from mazu.llm.pricing import estimate_cost
from mazu.memory.store import MemoryStore
from mazu.runs.router import suggest_model
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.tools.shell import denylist_reason, is_allowed_by_shell_allowlist
from mazu.usage.store import UsageStore


class ChatSession:
    """The web equivalent of mazu/agent/loop.py's run_chat_loop -- same turn logic
    (build_system_prompt once, run_turn_stream per round, the same tool-execution and
    destructive-tool-confirmation rules), but driven by two queues instead of
    input()/print() so it can run in a background thread while an HTTP server streams
    its output over SSE. Mirrors _run_until_done closely enough on purpose: this must
    stay behaviorally identical to `mazu chat`, just with a different I/O surface.

    Every sqlite-backed store (memory/usage/action-log) -- and the tool registry,
    which closes over them -- is opened lazily inside the background thread's own
    _run(), not in __init__ (which runs on the Flask request thread that calls
    ChatSession(...)). sqlite3 connections are only valid on the thread that opened
    them; opening them here first and handing them to the loop thread crashed with
    "SQLite objects created in a thread can only be used in that same thread" the
    moment a tool call or memory write happened, caught by this module's own tests.
    """

    def __init__(
        self,
        root: Path,
        model: str | None,
        shell_allowlist: list[str] | None,
        session_id: str | None = None,
        resumed_messages: list[dict] | None = None,
        auto_approve: bool = False,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.root = root
        self.model = model
        self.shell_allowlist = shell_allowlist
        # Mutable at runtime via set_auto_approve() (the Chat tab's toggle) -- not
        # just a constructor default -- so a user can flip it mid-conversation the
        # same way terminal `mazu chat`'s /auto command does.
        self.auto_approve = auto_approve

        self.messages: list[dict] = list(resumed_messages) if resumed_messages else []
        # A resumed session reuses the SAME session_id (and therefore the same
        # `sessions` row MemoryStore already has for it, from the original
        # process) -- start_session()'s INSERT would violate that row's primary
        # key and crash. Same fix as mazu/agent/loop.py's run_chat_loop for
        # `mazu chat --resume`.
        self.is_resume = resumed_messages is not None
        self.system_prompt: str | None = None
        self.total_cost = 0.0

        # Items are {"text": str, "images": list[dict]} dicts, except the one bare
        # "__mazu_web_close__" string close() puts to signal shutdown -- see send().
        self.inbox: "queue.Queue[dict | str]" = queue.Queue()
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._confirm_response: "queue.Queue[bool]" = queue.Queue()

        provider_name, model_name = _split_model(model or default_model())
        self.resolved_model = f"{provider_name}:{model_name}"
        self.cost_trackable = estimate_cost(self.resolved_model, 0, 0) is not None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(
        self, user_input: str, images: list[dict] | None = None, files: list[dict] | None = None
    ) -> None:
        # A plain dict, not a richer type, to keep the inbox's existing Queue[str]
        # sentinel trick working -- close() still puts the bare string
        # "__mazu_web_close__", and _run()'s loop tells the two apart with a
        # simple isinstance check rather than needing a wrapper class for one
        # sentinel value. `images` is a list of {"media_type": "image/png",
        # "data": <base64 str, no data: prefix>} dicts; `files` is a list of
        # {"filename": ..., "text": ...} dicts whose text has ALREADY been
        # extracted (see mazu/files/document_extract.py) -- both validated and,
        # for files, extracted by the /api/chat/<id>/message route before this
        # is ever called, since a raw .docx/.pdf/.xlsx is useless to any
        # provider and extraction failures need to be a clean 400, not a
        # confusing error several turns deep in a provider SDK call.
        self.inbox.put({"text": user_input, "images": images or [], "files": files or []})

    def answer_confirm(self, approved: bool) -> None:
        self._confirm_response.put(approved)

    def set_auto_approve(self, enabled: bool) -> None:
        self.auto_approve = enabled

    def set_model(self, model: str | None) -> None:
        """Changes the model this session's remaining turns use, effective from the
        next message on -- lets the Chat page's model chip apply to an already-
        running session instead of only sessions started after the change. Mirrors
        __init__'s own resolution so resolved_model/cost_trackable stay consistent
        with a session that had been started with this model from the start.
        """
        self.model = model
        provider_name, model_name = _split_model(model or default_model())
        self.resolved_model = f"{provider_name}:{model_name}"
        self.cost_trackable = estimate_cost(self.resolved_model, 0, 0) is not None

    def close(self) -> None:
        self._closed = True
        self.inbox.put("__mazu_web_close__")

    def join(self, timeout: float | None = None) -> None:
        """Waits for the background thread to actually exit. Not needed in
        production (the thread is a daemon; the process exiting is enough), but
        real test hygiene: this thread reads Path.home() -- the process-global HOME
        env var -- inside _run(). pytest's monkeypatch.setenv("HOME", ...) reverts
        at test teardown; an unjoined thread left running past its own test's end
        can still be mid-_run() when the NEXT test's fixture repoints HOME to a
        different tmp_path, corrupting that unrelated test in flaky, hard-to-
        reproduce ways (caught via a real, intermittent CI failure -- a `git add -A`
        in a completely different test's fixture failing with exit 128 for no
        reason connected to that test's own code).
        """
        self._thread.join(timeout=timeout)

    # -- background thread ---------------------------------------------------

    def _run(self) -> None:
        # Real bug caught live: mazu-web is a long-running server process, not a
        # per-invocation CLI call, so nothing had ever called load_config() here --
        # a key saved via `mazu setup`/`mazu config set` (or this app's own Config
        # tab) sits in config.toml but never reaches this process's environment,
        # so a real, working, verified key still hit "DEEPSEEK_API_KEY is not
        # set." Called fresh per session (not once at server startup) so a key
        # added to the Config tab after the server started works without a
        # restart.
        load_config()
        self.memory_store = MemoryStore(self.root / ".mazu" / "memory.db")
        self.global_memory_store = MemoryStore(Path.home() / ".mazu" / "global_memory.db")
        self.skill_manager = SkillManager(self.root)
        self.usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
        self.action_log_store = ActionLogStore(self.root / ".mazu" / "action_log.db")
        self.chat_store = ChatStore(self.root / ".mazu" / "chat_history.db")
        self.registry = build_registry(
            self.root, self.memory_store, self.global_memory_store, self.skill_manager, self.session_id
        )
        try:
            while True:
                item = self.inbox.get()
                if item == "__mazu_web_close__":
                    return
                user_input = item["text"]
                images = item["images"]
                files = item["files"]
                if not user_input.strip() and not images and not files:
                    continue
                self._handle_message(user_input, images, files)
        finally:
            finalize_session(
                self.memory_store, self.session_id, self.messages,
                model=self.model, usage_store=self.usage_store,
            )
            self.global_memory_store.close()
            self.usage_store.close()
            self.action_log_store.close()
            self.chat_store.close()
            self.outbox.put({"type": "closed"})

    def _handle_message(
        self, user_input: str, images: list[dict] | None = None, files: list[dict] | None = None
    ) -> None:
        if self.system_prompt is None:
            if self.memory_store is not None and not self.is_resume:
                self.memory_store.start_session(self.session_id)
            self.system_prompt = build_system_prompt(
                self.memory_store, self.skill_manager, query=user_input,
                global_memory_store=self.global_memory_store,
            )
            if self.system_prompt.strip() != "":
                self.outbox.put({"type": "info", "text": "loaded prior context relevant to this task"})
            if self.model is None and router_suggestions_enabled():
                self._emit_router_suggestion(user_input)

        # Images (if any) become the canonical Anthropic-shaped {"type": "image",
        # "source": {...}} blocks every provider converter already understands
        # (see mazu/llm/providers/*). Files (if any) were already extracted to
        # plain text by the /api/chat/<id>/message route (see
        # mazu/files/document_extract.py) -- no provider has a native .docx/
        # .xlsx content-block type, so each one becomes its own delimited text
        # block instead. A plain string content stays the common case (no
        # per-provider conversion needed at all) when there are neither.
        if images or files:
            content: str | list[dict] = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
                }
                for img in (images or [])
            ]
            content.extend(
                {
                    "type": "text",
                    "text": f"--- Attached file: {f['filename']} ---\n{f['text']}\n--- end of {f['filename']} ---",
                }
                for f in (files or [])
            )
            if user_input.strip():
                content.append({"type": "text", "text": user_input})
        else:
            content = user_input

        # Snapshotted BEFORE appending this turn's user message -- real bug found
        # via live testing: a turn that fails with a non-retryable MazuAPIError
        # (e.g. attaching an image to a model that doesn't support vision) used to
        # leave that failed user message sitting in self.messages forever. Since
        # every LLM call resends the full history, EVERY subsequent message in the
        # session -- including ones with no image at all -- kept re-triggering the
        # exact same "this model does not support image" error, permanently
        # breaking the conversation until a new chat was started. _run_until_done
        # restores this snapshot on any unrecoverable failure so a bad turn can't
        # poison the ones after it. ChatStore's durable record is deliberately NOT
        # rolled back -- ChatStore keeps everything that was actually attempted,
        # same as it keeps messages compaction later summarizes away.
        messages_before_turn = list(self.messages)

        self.messages.append({"role": "user", "content": content})
        self.chat_store.append_message(self.session_id, "user", content, model=self.resolved_model)
        self._run_until_done(messages_before_turn)

    def _emit_router_suggestion(self, task: str) -> None:
        try:
            run_store = RunStore(self.root / ".mazu" / "runs.db")
            usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
            try:
                suggestion = suggest_model(run_store, usage_store, task)
            finally:
                run_store.close()
                usage_store.close()
            if suggestion:
                self.outbox.put({"type": "router_suggestion", "text": suggestion})
        except Exception:
            pass

    def _run_until_done(self, messages_before_turn: list[dict]) -> None:
        provider_name, model_name = _split_model(self.resolved_model)
        while True:
            on_compacted = extract_memories_on_compaction(
                self.memory_store, self.session_id, self.model, usage_store=self.usage_store
            )
            new_messages, compacted = compact_if_needed(
                self.messages, self.model, usage_store=self.usage_store,
                session_id=self.session_id, on_compacted=on_compacted,
            )
            if compacted:
                self.messages[:] = new_messages
                self.outbox.put({
                    "type": "context_compacted",
                    "text": (
                        f"Compacted conversation history to stay within budget "
                        f"({len(self.messages)} messages remain in this turn's context)."
                    ),
                    "detail": "Nothing is lost -- your full conversation is still saved and viewable from Chat History.",
                })

            def _on_delta(chunk: str) -> None:
                self.outbox.put({"type": "delta", "text": chunk})

            try:
                response = run_turn_stream(
                    self.messages, self.system_prompt, self.registry.schemas(),
                    on_delta=_on_delta, model=self.model,
                )
            except MazuContextLengthError:
                self.outbox.put({
                    "type": "context_compacted",
                    "text": "Hit the model's context limit -- compacting aggressively and retrying once.",
                    "detail": "Your full conversation remains saved; only what's sent to the model just shrank.",
                })
                on_compacted = extract_memories_on_compaction(
                    self.memory_store, self.session_id, self.model, usage_store=self.usage_store
                )
                self.messages[:] = force_compact(
                    self.messages, self.model, usage_store=self.usage_store,
                    session_id=self.session_id, on_compacted=on_compacted,
                )
                try:
                    response = run_turn_stream(
                        self.messages, self.system_prompt, self.registry.schemas(),
                        on_delta=_on_delta, model=self.model,
                    )
                except MazuAPIError as e:
                    self.messages[:] = messages_before_turn
                    self.outbox.put({"type": "error", "text": str(e)})
                    return
            except MazuAPIError as e:
                self.messages[:] = messages_before_turn
                self.outbox.put({"type": "error", "text": str(e)})
                return
            self.messages.append({"role": "assistant", "content": response.content})
            self.chat_store.append_message(
                self.session_id, "assistant", response.content, model=self.resolved_model
            )

            usage = response.usage
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            cost_input_tokens, _, cache_read_tokens, cache_write_tokens = extract_cost_tokens(usage)
            step_cost = (
                estimate_cost(
                    self.resolved_model, cost_input_tokens, output_tokens,
                    cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
                )
                if self.cost_trackable else None
            )
            if step_cost is not None:
                self.total_cost += step_cost
            if self.usage_store is not None:
                self.usage_store.log(
                    "chat", self.session_id, provider_name, model_name, input_tokens, output_tokens, step_cost,
                    cache_read_tokens=cache_read_tokens or None, cache_write_tokens=cache_write_tokens or None,
                )
            self.outbox.put({
                "type": "usage",
                "text": summarize_usage(usage),
                "step_cost": step_cost,
                "total_cost": self.total_cost if step_cost is not None else None,
            })

            if response.stop_reason != "tool_use":
                self.outbox.put({"type": "turn_done"})
                return

            tool_results = self._execute_tools(response)
            self.messages.append({"role": "user", "content": tool_results})
            self.chat_store.append_message(
                self.session_id, "user", tool_results, model=self.resolved_model
            )

    def _execute_tools(self, response) -> list[dict]:
        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            tool = self.registry.get(block["name"])
            if tool is None:
                record_action(
                    self.action_log_store, self.session_id, "chat", block["name"], block["input"],
                    "unknown_tool", f"Unknown tool: {block['name']}",
                )
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block["id"],
                    "content": f"Unknown tool: {block['name']}", "is_error": True,
                })
                continue

            if tool.name == "run_shell":
                command = block["input"].get("command", "")
                reason = denylist_reason(command)
                if reason is not None:
                    msg = f"Blocked: command {reason} (safety denylist)."
                    record_action(self.action_log_store, self.session_id, "chat", tool.name, block["input"], "blocked", msg)
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True})
                    continue
                if not is_allowed_by_shell_allowlist(command, self.shell_allowlist):
                    msg = f"Blocked: command is not in the shell allowlist ({', '.join(self.shell_allowlist)})."
                    record_action(self.action_log_store, self.session_id, "chat", tool.name, block["input"], "blocked", msg)
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": msg, "is_error": True})
                    continue

            if tool.destructive:
                if self.auto_approve:
                    self.outbox.put({
                        "type": "tool_auto_run",
                        "text": f"ran {tool.name} without confirmation (auto mode is on)",
                    })
                else:
                    self.outbox.put({
                        "type": "confirm_request", "tool_name": tool.name, "tool_input": block["input"],
                    })
                    approved = self._confirm_response.get()
                    if not approved:
                        record_action(
                            self.action_log_store, self.session_id, "chat", tool.name, block["input"],
                            "declined", "User declined to run this tool.",
                        )
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": block["id"],
                            "content": "User declined to run this tool.", "is_error": True,
                        })
                        continue

            result = tool.handler(block["input"])
            record_action(
                self.action_log_store, self.session_id, "chat", tool.name, block["input"],
                "error" if result.is_error else "ok", result.content,
            )
            tool_results.append({
                "type": "tool_result", "tool_use_id": block["id"],
                "content": result.content, "is_error": result.is_error,
            })
        return tool_results
