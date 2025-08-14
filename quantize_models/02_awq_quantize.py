import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

SRC = os.environ.get("SRC", "./base_model")
OUT = os.environ.get("OUT", "./awq_model")

def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Loading base model from: {SRC}")
    # Ensure index exists (sharded checkpoints)
    idx = os.path.join(SRC, "model.safetensors.index.json")
    if not os.path.exists(idx):
        raise SystemExit(f"Missing {idx}. Re-run 01_fetch.py or set SRC correctly.")
    model = AutoAWQForCausalLM.from_pretrained(SRC, device_map="auto", torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(SRC, use_fast=True)

    print("Quantizing with AWQ 6-bit...")
    model.quantize(
        tok,
        quant_config={
            "w_bit": 6,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        },
    )

    print(f"Saving quantized model to: {OUT}")
    model.save_quantized(OUT, tok, safetensors=True)
    print("Done.")

if __name__ == "__main__":
    main()


