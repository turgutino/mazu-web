import contextlib
import io
import queue
import threading
import uuid
from pathlib import Path

from mazu.action_log.store import ActionLogStore
from mazu.agent.autonomous import run_autonomous
from mazu.agent.council import run_council
from mazu.agent.explore import format_explore_report, run_explore
from mazu.agent.registry_factory import build_registry
from mazu.checkpoint.manager import CheckpointManager
from mazu.config import load_config
from mazu.memory.store import MemoryStore
from mazu.runs.store import RunStore
from mazu.skills.manager import SkillManager
from mazu.usage.store import UsageStore

# contextlib.redirect_stdout replaces sys.stdout process-wide, not per-thread --
# two concurrent RunSession/ExploreSession threads would stomp on each other's
# capture (whichever exits first restores stdout under the other's feet, and
# their lines could interleave into the wrong session's queue). Both session
# types share this one lock so only one print()-capturing task runs at a time;
# the HTTP layer checks it non-blockingly and rejects a second launch with 409
# rather than silently queueing behind it.
STDOUT_CAPTURE_LOCK = threading.Lock()


class _QueueWriter(io.TextIOBase):
    """A file-like object that turns print()'d lines into outbox events. Both
    run_autonomous and run_explore are terminal-oriented (print()-based, no
    progress callback -- confirmed by reading their source before building this),
    so this is how their output reaches the browser: redirect stdout to this for
    the duration of the call, same idea as ChatSession's on_delta callback but at
    the print() layer instead of the token layer.
    """

    def __init__(self, outbox: "queue.Queue[dict]") -> None:
        self._outbox = outbox
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._outbox.put({"type": "log", "text": line})
        return len(text)

    def flush(self) -> None:
        pass


class RunSession:
    """One-shot `mazu run` (autonomous) invocation, executed in a background
    thread so the HTTP request that starts it can return immediately; its stdout
    is captured line-by-line into `outbox` for an SSE endpoint to forward.
    Caller must hold STDOUT_CAPTURE_LOCK (non-blocking) before constructing this.

    Mirrors mazu/cli.py's `run` command's full option set (not just the basics):
    --resume and --from-checkpoint/--branch resolve the same way here as they do
    there (same validation, same RunStore/CheckpointManager calls) -- this isn't
    a smaller reimplementation, it's the same logic with click.UsageError swapped
    for a returned error dict.
    """

    def __init__(
        self,
        root: Path,
        task: str | None,
        model: str | None,
        max_steps: int,
        checkpoint_every: int,
        allow_shell: bool,
        keep_checkpoints: int | None,
        max_cost: float | None,
        shell_allowlist: list[str] | None,
        dry_run: bool,
        resume_run_id: str | None,
        from_checkpoint_id: str | None,
        branch_name: str | None,
    ) -> None:
        self.task_id = str(uuid.uuid4())
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            args=(
                root, task, model, max_steps, checkpoint_every, allow_shell, keep_checkpoints,
                max_cost, shell_allowlist, dry_run, resume_run_id, from_checkpoint_id, branch_name,
            ),
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Waits for the background thread to exit -- see ChatSession.join's
        docstring for why this matters (Path.home() reads process-global state a
        lingering unjoined thread could touch after a test's fixture has already
        moved on to a different tmp_path)."""
        self._thread.join(timeout=timeout)

    def _run(
        self, root, task, model, max_steps, checkpoint_every, allow_shell, keep_checkpoints,
        max_cost, shell_allowlist, dry_run, resume_run_id, from_checkpoint_id, branch_name,
    ) -> None:
        load_config()  # see ChatSession._run's comment on why this matters here
        writer = _QueueWriter(self.outbox)
        checkpoint_kwargs = {"retention": keep_checkpoints} if keep_checkpoints is not None else {}
        checkpoint_manager = CheckpointManager(root, **checkpoint_kwargs)
        run_store = RunStore(root / ".mazu" / "runs.db")
        try:
            with contextlib.redirect_stdout(writer):
                resume_messages = None
                origin_checkpoint_id = None
                parent_run_id = None

                if from_checkpoint_id is not None:
                    origin_entry = checkpoint_manager.show_entry(from_checkpoint_id)
                    fork_result = checkpoint_manager.fork(origin_entry["id"], branch_name)
                    origin_checkpoint_id = origin_entry["id"]
                    parent_run_id = origin_entry.get("session_id")
                    resume_messages = fork_result["messages"]
                    print(
                        f"Forked from {origin_checkpoint_id} onto new branch {branch_name!r} "
                        f"({len(resume_messages)} prior message(s)). Will run: {task}"
                    )
                elif resume_run_id is not None:
                    run_row = run_store.get(resume_run_id)
                    if run_row is None:
                        self.outbox.put({"type": "log", "text": f"No run found with id {resume_run_id}."})
                        return
                    checkpoint_entry = checkpoint_manager.latest_for_session(resume_run_id)
                    if checkpoint_entry is None:
                        self.outbox.put({
                            "type": "log",
                            "text": f"No checkpoint found for run {resume_run_id} -- nothing to resume from.",
                        })
                        return
                    resume_messages = checkpoint_manager.inspect_conversation(checkpoint_entry["id"])
                    task = run_row["task"]
                    model = run_row["model"]
                    max_steps = run_row["max_steps"]
                    checkpoint_every = run_row["checkpoint_every"]
                    allow_shell = bool(run_row["allow_shell"])
                    shell_allowlist = run_row["shell_allowlist"]
                    max_cost = run_row["max_cost"]
                    dry_run = bool(run_row["dry_run"])
                    print(
                        f"Resuming run {resume_run_id} from {checkpoint_entry['id']} "
                        f"({len(resume_messages)} prior message(s))."
                    )

                session_id = resume_run_id if resume_run_id is not None else str(uuid.uuid4())
                memory_store = MemoryStore(root / ".mazu" / "memory.db")
                global_memory_store = MemoryStore(Path.home() / ".mazu" / "global_memory.db")
                skill_manager = SkillManager(root)
                usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
                action_log_store = ActionLogStore(root / ".mazu" / "action_log.db")
                registry = build_registry(
                    root, memory_store, global_memory_store, skill_manager, session_id, dry_run=dry_run
                )
                # shell_allowlist may be the fresh comma-string request value or a
                # resumed run's raw DB column (also a comma string, or None) --
                # parsed into a list once, uniformly, right here, same as the CLI's
                # own single _parse_shell_allowlist() call right before this.
                parsed_allowlist = (
                    [p.strip() for p in shell_allowlist.split(",") if p.strip()] if shell_allowlist else None
                )
                try:
                    run_autonomous(
                        registry, task, session_id, checkpoint_manager,
                        memory_store=memory_store, global_memory_store=global_memory_store,
                        skill_manager=skill_manager, max_steps=max_steps, checkpoint_every=checkpoint_every,
                        allow_shell=allow_shell, max_cost=max_cost, model=model, usage_store=usage_store,
                        action_log_store=action_log_store, shell_allowlist=parsed_allowlist, dry_run=dry_run,
                        run_store=run_store, resume_messages=resume_messages,
                        origin_checkpoint_id=origin_checkpoint_id, parent_run_id=parent_run_id,
                        branch_name=branch_name,
                    )
                finally:
                    memory_store.close()
                    global_memory_store.close()
                    usage_store.close()
                    action_log_store.close()
        except Exception as e:
            self.outbox.put({"type": "log", "text": f"[error] {e}"})
        finally:
            writer.flush()
            run_store.close()
            self.outbox.put({"type": "done"})
            STDOUT_CAPTURE_LOCK.release()


class ExploreSession:
    """One `mazu explore` invocation (N parallel branches), same stdout-capture
    idea as RunSession. Branches print concurrently from a thread pool inside
    run_explore itself; their lines interleave in the capture the same way they
    would in a real terminal running `mazu explore` directly -- not simulated,
    just observed as-is. Caller must hold STDOUT_CAPTURE_LOCK (non-blocking)
    before constructing this.
    """

    def __init__(
        self,
        root: Path,
        task: str,
        models: list[str],
        test_command: str | None,
        max_cost: float | None,
        max_steps: int,
        from_checkpoint_id: str | None = None,
    ) -> None:
        self.task_id = str(uuid.uuid4())
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            args=(root, task, models, test_command, max_cost, max_steps, from_checkpoint_id),
            daemon=True,
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self, root, task, models, test_command, max_cost, max_steps, from_checkpoint_id) -> None:
        load_config()  # see ChatSession._run's comment on why this matters here
        checkpoint_manager = CheckpointManager(root)
        writer = _QueueWriter(self.outbox)
        try:
            with contextlib.redirect_stdout(writer):
                results = run_explore(
                    task, models=models, root=root, checkpoint_manager=checkpoint_manager,
                    from_checkpoint_id=from_checkpoint_id, max_cost=max_cost,
                    test_command=test_command, max_steps=max_steps,
                )
            report = format_explore_report(results, test_command)
            for line in report.splitlines():
                self.outbox.put({"type": "log", "text": line})
        except Exception as e:
            self.outbox.put({"type": "log", "text": f"[error] {e}"})
        finally:
            writer.flush()
            self.outbox.put({"type": "done"})
            STDOUT_CAPTURE_LOCK.release()


class CouncilSession:
    """One `mazu council` invocation. Read-only (council members get a read-only
    tool registry internally -- see run_council), but still needs the shared
    stdout-capture lock like Run/Explore: sys.stdout redirection is process-wide
    regardless of whether the underlying work writes anything. Caller must hold
    STDOUT_CAPTURE_LOCK (non-blocking) before constructing this.
    """

    def __init__(
        self,
        root: Path,
        question: str,
        models: list[str],
        lead_model: str,
        max_cost: float | None,
    ) -> None:
        self.task_id = str(uuid.uuid4())
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(root, question, models, lead_model, max_cost), daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self, root, question, models, lead_model, max_cost) -> None:
        load_config()  # see ChatSession._run's comment on why this matters here
        session_id = str(uuid.uuid4())
        memory_store = MemoryStore(root / ".mazu" / "memory.db")
        global_memory_store = MemoryStore(Path.home() / ".mazu" / "global_memory.db")
        skill_manager = SkillManager(root)
        usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
        action_log_store = ActionLogStore(root / ".mazu" / "action_log.db")
        registry = build_registry(root, memory_store, global_memory_store, skill_manager, session_id)

        writer = _QueueWriter(self.outbox)
        try:
            with contextlib.redirect_stdout(writer):
                answer = run_council(
                    question, models=models, lead_model=lead_model, full_registry=registry,
                    memory_store=memory_store, global_memory_store=global_memory_store,
                    skill_manager=skill_manager, usage_store=usage_store, session_id=session_id,
                    action_log_store=action_log_store, max_cost=max_cost,
                )
            self.outbox.put({"type": "log", "text": ""})
            self.outbox.put({"type": "log", "text": "=== Final answer ==="})
            for line in answer.splitlines():
                self.outbox.put({"type": "log", "text": line})
        except Exception as e:
            self.outbox.put({"type": "log", "text": f"[error] {e}"})
        finally:
            writer.flush()
            memory_store.close()
            global_memory_store.close()
            usage_store.close()
            action_log_store.close()
            self.outbox.put({"type": "done"})
            STDOUT_CAPTURE_LOCK.release()
