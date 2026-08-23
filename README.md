# mazu-web

A local browser UI for [Mazu](https://github.com/turgutino/Mazu) -- the core coding-agent CLI stays terminal-first and dependency-light; this is a separate, optional companion for people who'd rather work in a browser tab.

Ships as its own package (`mazu-web`, console script `mazu-web`) so Mazu's core install never pulls in a web framework just to run `mazu chat` in a terminal. Depends on `mazu>=0.23.0`; 108+ tests, kept in sync with the core package's own test suite.

## What it does

Full parity with the terminal command surface -- every route calls straight into the same `mazu` package classes (`CheckpointManager`, `MemoryStore`, `RunStore`, the router, `run_autonomous`, `run_explore`, `run_council`, diagnostics) the CLI itself uses. This UI is a different way to reach the same operations, not a second implementation of them.

One known exception as of this writing: the core CLI's newer, experimental `mazu memory-belief stats`/`review` commands (observation-only belief-shadow inspection, see the core [README](https://github.com/turgutino/Mazu#belief-shadow-observation-experimental-observation-only--nothing-is-corrected-automatically-yet)) don't have a web equivalent yet -- use the terminal for those specifically until this catches up.

- **Chat** -- the same turn logic as `mazu chat`, streamed over Server-Sent Events instead of printed to a terminal.
- **Run** -- one-shot autonomous tasks (`mazu run`), streamed.
- **Explore** -- parallel branch comparison (`mazu explore`), streamed.
- **Council** -- ask multiple models independently, one lead synthesizes (`mazu council`), streamed.
- **Checkpoints** -- timeline, diff, compare, inspect, prune, branch-from, and one-click rollback.
- **Memory** -- search, pin/unpin, edit, forget, stats, "why would this retrieve", find/merge duplicates.
- **Skills**, **Runs** (+ compare-branches), **Router**, **Usage**, **Action log**, **Models**, **Doctor**, **Config** -- all full read (and write, where the CLI itself allows it).
- **Curator** -- an opt-in, fully autonomous background process that maintains Mazu's own state (memory, skills, run/checkpoint history, usage, config, council roster, action log) using its own, completely separate API key, matching `mazu curator ...`. Inert until configured; every mutation is reversible and logged with a rationale.
- **Init / Setup** -- works against a directory that isn't a Mazu project yet; an in-page banner and a Config-tab form do what `mazu init`/`mazu setup` would from the terminal.

Run and Explore and Council share one server-wide lock: all three wrap print()-only agent functions via `sys.stdout` redirection, which is process-global, not thread-local -- two concurrent captures would corrupt each other's output. Starting a second one while the first is still running gets a 409, not a silently queued mess.

## Install

```bash
pip install mazu-web
```

Depends on the core [`mazu`](https://pypi.org/project/mazu/) package (`mazu>=0.23.0`, pulled from PyPI automatically).

## Run

```bash
cd your-project        # doesn't need to be `mazu init`-ed first -- the browser handles that
mazu-web                # http://127.0.0.1:8765
```

```
mazu-web --host 127.0.0.1 --port 8765 --model deepseek:deepseek-chat --shell-allowlist git,npm,pytest
```

Binds to `127.0.0.1` only by default -- **there is no authentication**, so don't pass `--host 0.0.0.0` or otherwise expose this beyond your own machine.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, same as the core project.
