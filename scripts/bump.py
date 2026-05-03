"""Release helper script."""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def get(cmd: list[str]) -> str:
    return subprocess.check_output(cmd).decode().strip()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/bump.py patch|minor|major|tag-and-push")
        sys.exit(1)

    action = sys.argv[1]

    if action in ("patch", "minor", "major"):
        run(["uv", "version", "--bump", action])
        version = get(["uv", "version", "--short"])
        run(["git", "add", "pyproject.toml"])
        run(["git", "commit", "-m", f"bump version to {version}"])
        print(
            f"Bumped to {version} — review with 'git diff HEAD~1' then run 'uv run task tag-and-push'",
        )

    elif action == "tag-and-push":
        version = get(["uv", "version", "--short"])
        run(["git", "tag", f"v{version}"])
        run(["git", "push", "origin", "HEAD"])
        run(["git", "push", "origin", f"v{version}"])
        print(f"Tagged and pushed v{version}")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
