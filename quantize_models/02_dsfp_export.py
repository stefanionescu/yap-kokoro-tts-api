import os
import subprocess
import sys
import importlib.util
from pathlib import Path

# Source (FP16/BF16) model directory downloaded by 01_fetch.py
SRC = os.environ.get("SRC", "./base_model")

# Output directory for exported DeepSpeedFP (FP6) weights
OUT = os.environ.get("OUT", "./dsfp_model")

# Optional: override the exporter entrypoint if your environment differs.
# We will try multiple candidates automatically.
DSFP_EXPORT_CMD = os.environ.get("DSFP_EXPORT_CMD")
EXPORTER_CANDIDATES = [
    # common module path
    "python -m llmcompressor.export.deepspeedfp",
    # alt module path (older tree)
    "python -m llmcompressor.scripts.export.deepspeedfp",
    # CLI form
    "llmcompressor export deepspeedfp",
]

def _discover_exporter_file_candidates() -> list[str]:
    candidates: list[str] = []
    # Prefer vendored repo path if provided by caller
    vendor_repo = os.environ.get("VENDOR_REPO")
    search_roots: list[Path] = []
    if vendor_repo and os.path.isdir(vendor_repo):
        search_roots.append(Path(vendor_repo))
    # Also search installed package location
    spec = importlib.util.find_spec("llmcompressor")
    if spec and spec.origin:
        pkg_dir = Path(spec.origin).parent
        search_roots.append(pkg_dir)
    # Walk and collect plausible script files
    name_hints = ("deepspeed", "dsfp")
    for root in search_roots:
        for p in root.rglob("*.py"):
            n = p.name.lower()
            if any(h in n for h in name_hints) and ("export" in str(p.parent).lower() or "export" in n):
                candidates.append(str(p))
    return candidates


def main() -> None:
    if not os.path.isdir(SRC):
        raise SystemExit(f"[dsfp] Source model directory not found: {SRC}. Run 01_fetch.py first or set SRC.")

    os.makedirs(OUT, exist_ok=True)

    # Basic sanity: require HF token if the exporter needs to fetch anything
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("[dsfp] Hint: set HUGGING_FACE_HUB_TOKEN or HF_TOKEN if model access is gated.")

    # Build candidate commands
    candidates = []
    if DSFP_EXPORT_CMD:
        candidates.append(DSFP_EXPORT_CMD)
    candidates.extend(EXPORTER_CANDIDATES)

    last_err = None

    # Add file path candidates discovered dynamically (run as `python <file> ...`)
    file_candidates = _discover_exporter_file_candidates()
    for fc in file_candidates:
        candidates.insert(0, f"python {fc}")
    for base_cmd in candidates:
        cmd = (
            f"{base_cmd} "
            f"--model {SRC} "
            f"--output {OUT} "
            f"--bits 6"
        )
        print("[dsfp] Trying exporter:\n  " + cmd)
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"[dsfp] Export succeeded with: {base_cmd}")
            break
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"[dsfp] Exporter failed with '{base_cmd}', trying next if available...", file=sys.stderr)
    else:
        msg = (
            "[dsfp] All exporter candidates failed. Ensure llm-compressor is installed and choose the correct entrypoint.\n"
            "You can override via DSFP_EXPORT_CMD env var. Examples:\n"
            "  export DSFP_EXPORT_CMD='python -m llmcompressor.scripts.export.deepspeedfp'\n"
            "  export DSFP_EXPORT_CMD='llmcompressor export deepspeedfp'\n"
        )
        print(msg, file=sys.stderr)
        if last_err is not None:
            raise SystemExit(last_err.returncode)
        raise SystemExit(1)

    print(f"[dsfp] Export complete -> {OUT}")


if __name__ == "__main__":
    main()


