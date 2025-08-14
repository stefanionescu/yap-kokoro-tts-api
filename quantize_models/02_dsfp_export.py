import os
import subprocess
import sys

# Source (FP16/BF16) model directory downloaded by 01_fetch.py
SRC = os.environ.get("SRC", "./base_model")

# Output directory for exported DeepSpeedFP (FP6) weights
OUT = os.environ.get("OUT", "./dsfp_model")

# Optional: override the exporter entrypoint if your environment differs
# Examples that may work depending on llm-compressor version:
#   - python -m llmcompressor.export.deepspeedfp
#   - python -m llmcompressor.scripts.export.deepspeedfp
#   - llmcompressor export deepspeedfp
DSFP_EXPORT_CMD = os.environ.get(
    "DSFP_EXPORT_CMD",
    "python -m llmcompressor.export.deepspeedfp",
)


def main() -> None:
    if not os.path.isdir(SRC):
        raise SystemExit(f"[dsfp] Source model directory not found: {SRC}. Run 01_fetch.py first or set SRC.")

    os.makedirs(OUT, exist_ok=True)

    # Basic sanity: require HF token if the exporter needs to fetch anything
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("[dsfp] Hint: set HUGGING_FACE_HUB_TOKEN or HF_TOKEN if model access is gated.")

    # Construct exporter command
    # We pass common args; adjust if your llm-compressor version requires different flags.
    cmd = (
        f"{DSFP_EXPORT_CMD} "
        f"--model {SRC} "
        f"--output {OUT} "
        f"--bits 6"
    )

    print("[dsfp] Running exporter:\n  " + cmd)
    try:
        # Use shell=True to allow the default string command and environment overrides
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        msg = (
            "[dsfp] Export failed. Ensure llm-compressor is installed and DSFP exporter path is correct.\n"
            "Tried command: "
            + cmd
            + "\n"
            "You can override the exporter via DSFP_EXPORT_CMD env var.\n"
            "Example: export DSFP_EXPORT_CMD='python -m llmcompressor.scripts.export.deepspeedfp'\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(e.returncode)

    print(f"[dsfp] Export complete -> {OUT}")


if __name__ == "__main__":
    main()


