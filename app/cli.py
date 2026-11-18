"""Admin command line.

    docker compose exec api python -m app.cli create-admin --email you@example.com --name "You"
    docker compose exec api python -m app.cli seed-demo

`create-admin` bootstraps the first account (there is no sign-up). `seed-demo` creates the
demo groups and users from samples/demo-users.json and shares the sample documents with
them; it refuses to run when ENVIRONMENT=production.
"""

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.auth import MIN_PASSWORD_LENGTH
from app.core.cache import Cache
from app.core.config import get_settings
from app.core.security import hash_password
from app.db.models import Document, Group, User
from app.db.session import build_engine, build_session_factory

DEMO_FILE = Path(__file__).resolve().parents[1] / "samples" / "demo-users.json"


async def create_admin(email: str, name: str, password: str) -> None:
    engine = build_engine(get_settings())
    try:
        async with build_session_factory(engine)() as session:
            email = email.strip().lower()
            if await session.scalar(select(User).where(func.lower(User.email) == email)):
                sys.exit(f"A user with email {email} already exists")
            session.add(
                User(
                    email=email,
                    display_name=name,
                    password_hash=hash_password(password),
                    is_admin=True,
                )
            )
            await session.commit()
            print(f"Created admin {email}")
    finally:
        await engine.dispose()


async def seed_demo() -> None:
    settings = get_settings()
    if settings.environment == "production":
        sys.exit("seed-demo creates accounts with published passwords; refusing in production")
    spec = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    engine = build_engine(settings)
    try:
        async with build_session_factory(engine)() as session:
            groups: dict[str, Group] = {g.name: g for g in await session.scalars(select(Group))}
            for item in spec["groups"]:
                if item["name"] not in groups:
                    groups[item["name"]] = Group(name=item["name"], description=item["description"])
                    session.add(groups[item["name"]])
                    print(f"group  {item['name']}")

            existing = {u.email for u in await session.scalars(select(User))}
            for item in spec["users"]:
                if item["email"] in existing:
                    continue
                session.add(
                    User(
                        email=item["email"],
                        display_name=item["name"],
                        password_hash=hash_password(item["password"]),
                        is_admin=item.get("is_admin", False),
                        groups=[groups[name] for name in item.get("groups", [])],
                    )
                )
                print(f"user   {item['email']}")

            documents = await session.scalars(
                select(Document).options(selectinload(Document.groups))
            )
            by_title = {d.title: d for d in documents}
            for title, group_names in spec["document_access"].items():
                document = by_title.get(title)
                if document is None:
                    print(f"skip   '{title}' (not uploaded yet: upload samples/corpus first)")
                    continue
                document.groups = [groups[name] for name in group_names]
                print(f"share  '{title}' -> {', '.join(group_names)}")
            await session.commit()
    finally:
        await engine.dispose()
    # Sharing changed: drop cached results.
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await Cache(redis, settings).bump_index_version()
    await redis.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser("create-admin", help="create an admin account")
    admin.add_argument("--email", required=True)
    admin.add_argument("--name", required=True)
    commands.add_parser("seed-demo", help="create demo groups/users and share sample docs")
    args = parser.parse_args()

    if args.command == "create-admin":
        password = getpass.getpass("Password: ")
        if len(password) < MIN_PASSWORD_LENGTH:
            sys.exit(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        if getpass.getpass("Repeat password: ") != password:
            sys.exit("Passwords don't match")
        asyncio.run(create_admin(args.email, args.name, password))
    elif args.command == "seed-demo":
        asyncio.run(seed_demo())


if __name__ == "__main__":
    main()
