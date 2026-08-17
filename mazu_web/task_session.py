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
    """

    def __init__(
        self,
        root: Path,
        task: str,
        model: str | None,
        max_steps: int,
        allow_shell: bool,
        max_cost: float | None,
    ) -> None:
        self.task_id = str(uuid.uuid4())
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(root, task, model, max_steps, allow_shell, max_cost), daemon=True
        )
        self._thread.start()

    def _run(self, root, task, model, max_steps, allow_shell, max_cost) -> None:
        session_id = str(uuid.uuid4())
        memory_store = MemoryStore(root / ".mazu" / "memory.db")
        global_memory_store = MemoryStore(Path.home() / ".mazu" / "global_memory.db")
        skill_manager = SkillManager(root)
        checkpoint_manager = CheckpointManager(root)
        usage_store = UsageStore(Path.home() / ".mazu" / "usage.db")
        run_store = RunStore(root / ".mazu" / "runs.db")
        registry = build_registry(root, memory_store, global_memory_store, skill_manager, session_id)

        writer = _QueueWriter(self.outbox)
        try:
            with contextlib.redirect_stdout(writer):
                run_autonomous(
                    registry, task, session_id, checkpoint_manager,
                    memory_store=memory_store, global_memory_store=global_memory_store,
                    skill_manager=skill_manager, max_steps=max_steps, allow_shell=allow_shell,
                    max_cost=max_cost, model=model, usage_store=usage_store, run_store=run_store,
                )
        except Exception as e:
            self.outbox.put({"type": "log", "text": f"[error] {e}"})
        finally:
            writer.flush()
            memory_store.close()
            global_memory_store.close()
            usage_store.close()
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
    ) -> None:
        self.task_id = str(uuid.uuid4())
        self.outbox: "queue.Queue[dict]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(root, task, models, test_command, max_cost, max_steps), daemon=True
        )
        self._thread.start()

    def _run(self, root, task, models, test_command, max_cost, max_steps) -> None:
        checkpoint_manager = CheckpointManager(root)
        writer = _QueueWriter(self.outbox)
        try:
            with contextlib.redirect_stdout(writer):
                results = run_explore(
                    task, models=models, root=root, checkpoint_manager=checkpoint_manager,
                    max_cost=max_cost, test_command=test_command, max_steps=max_steps,
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

    def _run(self, root, question, models, lead_model, max_cost) -> None:
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
