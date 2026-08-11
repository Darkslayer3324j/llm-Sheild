"""
cli.py — local management CLI for llm-shield.

Usage:
    python -m app.cli keys create --name "my-app" --daily-budget 5.0 --rate-limit 60
    python -m app.cli keys list
    python -m app.cli keys revoke <key>
    python -m app.cli stats
    python -m app.cli serve
"""
from __future__ import annotations

import asyncio

import typer

from app.config import get_settings
from app.db import Database

app = typer.Typer(help="llm-shield management CLI")
keys_app = typer.Typer(help="Manage virtual API keys")
app.add_typer(keys_app, name="keys")

settings = get_settings()


def _run(coro):
    return asyncio.run(coro)


async def _get_db() -> Database:
    db = Database(settings.db_path)
    await db.connect()
    return db


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", help="Human-readable label for this key."),
    daily_budget: float = typer.Option(
        settings.default_daily_budget_usd, "--daily-budget", help="Daily USD spend cap."
    ),
    rate_limit: int = typer.Option(
        settings.default_rate_limit_rpm, "--rate-limit", help="Requests per minute cap."
    ),
    admin: bool = typer.Option(
        False, "--admin", help="Grant admin rights (can manage keys, view the dashboard)."
    ),
) -> None:
    async def _create():
        db = await _get_db()
        record = await db.create_api_key(name, daily_budget, rate_limit, is_admin=admin)
        await db.close()
        return record

    record = _run(_create())
    typer.secho("Created API key:", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  key:            {record.key}")
    typer.echo(f"  name:           {record.name}")
    typer.echo(f"  daily_budget:   ${record.daily_budget_usd:.2f}")
    typer.echo(f"  rate_limit_rpm: {record.rate_limit_rpm}")
    typer.echo(f"  admin:          {record.is_admin}")
    typer.echo("\nUse it as: Authorization: Bearer " + record.key)


@keys_app.command("list")
def keys_list() -> None:
    async def _list():
        db = await _get_db()
        records = await db.list_api_keys()
        await db.close()
        return records

    records = _run(_list())
    if not records:
        typer.echo("No API keys yet. Create one with: python -m app.cli keys create --name ...")
        return

    for r in records:
        status = "active" if r.is_active else "REVOKED"
        admin_tag = " [ADMIN]" if r.is_admin else ""
        typer.echo(f"{r.key}  [{status}]{admin_tag}  name={r.name!r}  daily_budget=${r.daily_budget_usd:.2f}  rate_limit={r.rate_limit_rpm}/min")


@keys_app.command("revoke")
def keys_revoke(key: str = typer.Argument(..., help="The virtual key to revoke.")) -> None:
    async def _revoke():
        db = await _get_db()
        ok = await db.revoke_api_key(key)
        await db.close()
        return ok

    ok = _run(_revoke())
    if ok:
        typer.secho(f"Revoked {key}", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"No such key: {key}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("stats")
def stats() -> None:
    """Print today's spend/usage summary."""

    async def _stats():
        db = await _get_db()
        data = await db.get_dashboard_stats()
        await db.close()
        return data

    data = _run(_stats())
    typer.secho("=== llm-shield — today ===", bold=True)
    typer.echo(f"  Spend:      ${data['today_spend_usd']:.4f}")
    typer.echo(f"  Requests:   {data['today_requests']}")
    typer.echo(f"  Redactions: {data['today_redactions']}")
    typer.echo(f"  Cache hit rate: {data['cache_hit_rate'] * 100:.1f}%")
    if data["spend_by_model"]:
        typer.echo("\n  By model:")
        for row in data["spend_by_model"]:
            typer.echo(f"    {row['model']:<30} {row['requests']:>4} req   ${row['spend']:.4f}")


@app.command("serve")
def serve(
    host: str = typer.Option(settings.host, "--host"),
    port: int = typer.Option(settings.port, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the llm-shield proxy server."""
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
