from pathlib import Path

import click


def _parse_shell_allowlist(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address. Keep this local -- there is no auth.")
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--model", default=None, help="Override the model for chat sessions started in this UI.")
@click.option(
    "--shell-allowlist",
    default=None,
    help="Comma-separated program names shell tool calls are restricted to, same as `mazu chat --shell-allowlist`.",
)
def main(host: str, port: int, model: str | None, shell_allowlist: str | None) -> None:
    """Launch a local browser UI for Mazu: chat (streamed, same rules as `mazu chat`),
    plus read-only Checkpoints/Memory/Router tabs. Requires the current directory to
    already be a Mazu project (`mazu init` first, from the core `mazu` package).
    Binds to localhost only by default -- there is no authentication, so do not
    expose --host beyond your own machine.
    """
    from mazu_web.app import create_app

    root = Path.cwd()
    if not (root / ".mazu").exists():
        click.echo("No .mazu/ here yet -- run `mazu init` first.")
        return

    app = create_app(root, model, _parse_shell_allowlist(shell_allowlist))
    click.echo(f"mazu-web running at http://{host}:{port} (Ctrl+C to stop)")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
