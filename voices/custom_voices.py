#!/usr/bin/env python3
"""
Create and manage Kokoro custom voices on the pod (no API required).

Persisted at: custom_voices/custom_voices.json (auto-created)

Recipe format:
  - A '+'-separated blend of Kokoro voices, e.g. "af_aoede+am_michael"
  - You can also store a single base voice, e.g. "af_bella"

Examples:
  # Add/update a custom voice
  python voices/custom_voices.py add --name my_blend --recipe "af_aoede+am_michael" --validate

  # List
  python voices/custom_voices.py list

  # Remove
  python voices/custom_voices.py remove --name my_blend

Note: The server reads custom voices on startup. After changes, restart:
  bash scripts/stop.sh || true && bash scripts/start.sh
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict


CUSTOM_DIR = Path("custom_voices").resolve()
CUSTOM_JSON = CUSTOM_DIR / "custom_voices.json"


def load_db() -> Dict[str, str]:
    if CUSTOM_JSON.exists():
        try:
            return json.loads(CUSTOM_JSON.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def save_db(db: Dict[str, str]) -> None:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_JSON.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_recipe(recipe: str) -> None:
    """Light validation: optionally try loading each base voice via Kokoro.
    Skips if kokoro not installed.
    """
    try:
        from kokoro import KPipeline  # type: ignore
    except Exception:
        print("[warn] kokoro not available; skipping validation")
        return

    lang = os.getenv("LANG_CODE", "a")
    device = os.getenv("KOKORO_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    pipe = KPipeline(lang_code=lang, device=device)
    for token in [t.strip() for t in recipe.split("+") if t.strip()]:
        try:
            pipe.load_voice(token)
            print(f"[ok] validated base voice: {token}")
        except Exception as e:
            print(f"[warn] could not validate {token}: {e}")


def cmd_add(name: str, recipe: str, do_validate: bool) -> None:
    if not name or not recipe:
        raise SystemExit("name and recipe are required")
    if do_validate:
        validate_recipe(recipe)
    db = load_db()
    db[name] = recipe
    save_db(db)
    print(f"Saved custom voice '{name}' → {recipe}")
    print(f"DB: {CUSTOM_JSON}")


def cmd_remove(name: str) -> None:
    db = load_db()
    if name in db:
        db.pop(name, None)
        save_db(db)
        print(f"Removed '{name}'")
    else:
        print(f"No entry named '{name}'")


def cmd_list() -> None:
    db = load_db()
    if not db:
        print("(empty)")
        print(f"DB: {CUSTOM_JSON}")
        return
    print(f"DB: {CUSTOM_JSON}")
    for k, v in sorted(db.items()):
        print(f"- {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Manage Kokoro custom voices on disk")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add or update a custom voice")
    a.add_argument("--name", required=True)
    a.add_argument("--recipe", required=True)
    a.add_argument("--validate", action="store_true", help="attempt to load base voices via Kokoro")

    r = sub.add_parser("remove", help="remove a custom voice")
    r.add_argument("--name", required=True)

    sub.add_parser("list", help="list custom voices")

    args = ap.parse_args()
    if args.cmd == "add":
        cmd_add(args.name, args.recipe, args.validate)
    elif args.cmd == "remove":
        cmd_remove(args.name)
    elif args.cmd == "list":
        cmd_list()


if __name__ == "__main__":
    main()


