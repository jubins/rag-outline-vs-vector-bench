"""Check that the YAML and JSON config files parse and that the dashboard has panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    errors: list[str] = []
    for path in [*ROOT.glob("observability/**/*.yml"), *ROOT.glob("observability/**/*.yaml")]:
        try:
            yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in ROOT.glob("observability/grafana/dashboards/*.json"):
        try:
            dashboard = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        if not dashboard.get("panels"):
            errors.append(f"{path.relative_to(ROOT)}: no panels")
    for message in errors:
        print(message, file=sys.stderr)
    print("config check passed" if not errors else f"{len(errors)} problem(s) found")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
