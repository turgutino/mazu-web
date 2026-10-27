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
@click.option(
    "--allowed-host",
    "allowed_hosts",
    multiple=True,
    help="Extra hostname the server may be reached by (repeatable). Requests whose Host header is not "
    "localhost, 127.0.0.1, ::1, or listed here are rejected -- this is what blocks DNS-rebinding attacks. "
    "Only needed if you deliberately serve beyond your own machine, which is not recommended.",
)
def main(
    host: str, port: int, model: str | None, shell_allowlist: str | None, allowed_hosts: tuple[str, ...]
) -> None:
    """Launch a local browser UI for Mazu: chat, run, explore, council, checkpoints,
    memory, skills, runs, router, usage, action log, models, doctor, and config --
    the full terminal command surface. Works against a directory that isn't a Mazu
    project yet, too -- an Init/Setup banner in the browser handles that the same
    way `mazu init`/`mazu setup` would from the terminal. Binds to localhost only
    by default -- there is no authentication, so do not expose --host beyond your
    own machine.
    """
    from mazu_web.app import create_app

    root = Path.cwd()
    extra_hosts = list(allowed_hosts)
    if host not in ("0.0.0.0", "::"):
        extra_hosts.append(host)
    app = create_app(root, model, _parse_shell_allowlist(shell_allowlist), allowed_hosts=extra_hosts)
    click.echo(f"mazu-web running at http://{host}:{port} (Ctrl+C to stop)")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
