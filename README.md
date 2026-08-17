# mazu-web

A local browser UI for [Mazu](https://github.com/turgutino/Mazu) -- the core coding-agent CLI stays terminal-first and dependency-light; this is a separate, optional companion for people who'd rather work in a browser tab.

Ships as its own package (`mazu-web`, console script `mazu-web`) so Mazu's core install never pulls in a web framework just to run `mazu chat` in a terminal.

## What it does

- **Chat** -- the same turn logic as `mazu chat` (same system prompt, same tools, same destructive-tool confirmation rules), streamed over Server-Sent Events instead of printed to a terminal.
- **Checkpoints** -- browse the timeline, roll back with one click (same `checkpoint_manager.restore()` the CLI's `mazu rollback` uses).
- **Memory** -- browse everything the agent has remembered for this project.
- **Router** -- the same win-rate/cost stats as `mazu router stats`, filterable by task type.

Nothing here is a second implementation of Mazu's core logic -- every route calls straight into the same `mazu` package classes (`CheckpointManager`, `MemoryStore`, `RunStore`, the router) the CLI itself uses. This UI is a different way to reach the same operations, not a fork of them.

## Install

```bash
pip install "mazu-web @ git+https://github.com/turgutino/mazu-web.git"
```

Requires an existing Mazu project (`mazu init` first, from the core [`mazu`](https://github.com/turgutino/Mazu) package -- installed automatically as a dependency of this package).

Once the core `mazu` package is published to PyPI, this package's dependency will switch from a git URL to a normal version pin.

## Run

```bash
cd your-project        # already `mazu init`-ed
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
