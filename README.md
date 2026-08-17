# mazu-web

A local browser UI for [Mazu](https://github.com/turgutino/Mazu) -- the core coding-agent CLI stays terminal-first and dependency-light; this is a separate, optional companion for people who'd rather work in a browser tab.

Ships as its own package (`mazu-web`, console script `mazu-web`) so Mazu's core install never pulls in a web framework just to run `mazu chat` in a terminal.

## What it does

Full parity with the terminal command surface -- every route calls straight into the same `mazu` package classes (`CheckpointManager`, `MemoryStore`, `RunStore`, the router, `run_autonomous`, `run_explore`, `run_council`, diagnostics) the CLI itself uses. This UI is a different way to reach the same operations, not a second implementation of them.

- **Chat** -- the same turn logic as `mazu chat`, streamed over Server-Sent Events instead of printed to a terminal.
- **Run** -- one-shot autonomous tasks (`mazu run`), streamed.
- **Explore** -- parallel branch comparison (`mazu explore`), streamed.
- **Council** -- ask multiple models independently, one lead synthesizes (`mazu council`), streamed.
- **Checkpoints** -- timeline, diff, compare, inspect, prune, branch-from, and one-click rollback.
- **Memory** -- search, pin/unpin, edit, forget, stats, "why would this retrieve", find/merge duplicates.
- **Skills**, **Runs** (+ compare-branches), **Router**, **Usage**, **Action log**, **Models**, **Doctor**, **Config** -- all full read (and write, where the CLI itself allows it).
- **Init / Setup** -- works against a directory that isn't a Mazu project yet; an in-page banner and a Config-tab form do what `mazu init`/`mazu setup` would from the terminal.

Run and Explore and Council share one server-wide lock: all three wrap print()-only agent functions via `sys.stdout` redirection, which is process-global, not thread-local -- two concurrent captures would corrupt each other's output. Starting a second one while the first is still running gets a 409, not a silently queued mess.

## Install

```bash
pip install "mazu-web @ git+https://github.com/turgutino/mazu-web.git"
```

Depends on the core [`mazu`](https://github.com/turgutino/Mazu) package, installed automatically. Once `mazu` is published to PyPI, this package's dependency will switch from a git URL to a normal version pin.

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
