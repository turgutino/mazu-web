# mazu-web

A local browser UI for [Mazu](https://github.com/turgutino/Mazu) -- the core coding-agent CLI stays terminal-first and dependency-light; this is a separate, optional companion for people who'd rather work in a browser tab.

Ships as its own package (`mazu-web`, console script `mazu-web`) so Mazu's core install never pulls in a web framework just to run `mazu chat` in a terminal. Depends on `mazu[documents]>=0.24.0`; 150+ tests.

## What it does

Full parity with the terminal command surface -- every route calls straight into the same `mazu` package classes (`CheckpointManager`, `MemoryStore`, `RunStore`, the router, `run_autonomous`, `run_explore`, `run_council`, diagnostics) the CLI itself uses. This UI is a different way to reach the same operations, not a second implementation of them.

Known gaps versus the terminal: `mazu memory-belief` (observation-only belief logging), `mazu bench`, and `mazu chat --max-cost` / the final chat spend total have no web equivalent. The Chat tab keeps its own copy of the chat loop (`mazu_web/chat_session.py`) instead of calling the core one, so loop-level changes in `mazu` do not reach it automatically; Run, Explore, and Council call the same functions as the terminal.

- **Chat** -- the same turn logic as `mazu chat`, streamed over Server-Sent Events instead of printed to a terminal. Also supports attaching an image or a document (`.docx`/`.pdf`/`.xlsx`/`.pptx`, extracted to plain text), a live per-provider model picker (real API-backed, not a single hardcoded default), and reasoning-effort control for models that support it -- the model and effort controls apply to the session immediately, not just to a brand-new chat, and both warn (rather than silently no-op) when the currently selected model doesn't actually support what's being asked of it.
- **Run** -- one-shot autonomous tasks (`mazu run`), streamed.
- **Explore** -- parallel branch comparison (`mazu explore`), streamed, with a live multi-model picker and a best-effort test-command suggestion (pytest/npm/go/cargo/maven/gradle markers) pre-filling the optional field instead of leaving it blank.
- **Council** -- ask multiple models independently, one lead synthesizes (`mazu council`), streamed.
- **Checkpoints** -- timeline, diff, compare, inspect, prune, branch-from, and one-click rollback.
- **Memory** -- search, pin/unpin, edit, forget, stats, "why would this retrieve", find/merge duplicates.
- **Skills**, **Runs** (+ compare-branches), **Router**, **Usage**, **Action log**, **Models**, **Doctor**, **Config** -- all full read (and write, where the CLI itself allows it).
- **Curator** -- an opt-in, fully autonomous background process that maintains Mazu's own state in four areas (memory, skills, housekeeping, and memory conflicts) using its own, completely separate API key, matching `mazu curator ...`. Inert until configured; every mutation is reversible and logged with a rationale.
- **Init / Setup** -- works against a directory that isn't a Mazu project yet; an in-page banner and a Config-tab form do what `mazu init`/`mazu setup` would from the terminal.

Run and Explore and Council share one server-wide lock: all three wrap print()-only agent functions via `sys.stdout` redirection, which is process-global, not thread-local -- two concurrent captures would corrupt each other's output. Starting a second one while the first is still running gets a 409, not a silently queued mess.

## Install

```bash
pip install mazu-web
```

Depends on the core [`mazu`](https://pypi.org/project/mazu/) package (`mazu[documents]>=0.24.0`, pulled from PyPI automatically).

## Run

```bash
cd your-project        # doesn't need to be `mazu init`-ed first -- the browser handles that
mazu-web                # http://127.0.0.1:8765
```

```
mazu-web --host 127.0.0.1 --port 8765 --model deepseek:deepseek-chat --shell-allowlist git,npm,pytest   # add --allowed-host NAME only if you serve under another hostname
```

## Security

Binds to `127.0.0.1` only by default, and **there is no authentication**: anything that can reach the port can read your project, change memory, roll back checkpoints, and start agent runs that execute commands with your API keys. Do not pass `--host 0.0.0.0` or otherwise expose it beyond your own machine.

Even on localhost, a web page open in your browser can try to reach a local server. Since 0.4.1 mazu-web rejects requests whose `Host` header is not `localhost`, `127.0.0.1`, or `::1` (DNS rebinding), and rejects state-changing requests (POST, PUT, PATCH, DELETE) whose `Origin` is not the page's own origin (cross-site requests). Before that, both were possible: a cross-site POST could trigger a checkpoint rollback, and a rebound hostname could reach every endpoint. If you deliberately serve it under another hostname, list it with `--allowed-host NAME` (repeatable); this is not recommended. Update to the latest release if you are on 0.4.0 or earlier.

The terminal's own limits apply inside the browser too: the shell denylist is not a sandbox, `--shell-allowlist` only checks how a command starts, and in the Run tab `allow_shell` skips confirmation for shell commands while saved skills run without a prompt. See the core [SECURITY.md](https://github.com/turgutino/Mazu/blob/main/SECURITY.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, same as the core project.
