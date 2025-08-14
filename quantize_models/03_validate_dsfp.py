import os
from transformers import AutoTokenizer
from vllm import LLM

# Validate that vLLM can load the exported DSFP model directory
MODEL_DIR = os.environ.get("MODEL_DIR", "./dsfp_model")


def main() -> None:
    print(f"[dsfp-validate] Loading DSFP model with vLLM from: {MODEL_DIR}")
    # This will instantiate a minimal LLM instance pointing at local path
    llm = LLM(model=MODEL_DIR, quantization="deepspeedfp", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    prompt = "tara: hello<|eot_id|>"
    out = llm.generate([prompt], sampling_params=None)
    print("[dsfp-validate] vLLM generate() returned ", len(out), "outputs")


if __name__ == "__main__":
    main()


