import os
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

MODEL_DIR = os.environ.get("MODEL_DIR", "./awq_model")

def main():
    print(f"Loading quantized model from: {MODEL_DIR}")
    model = AutoAWQForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    prompt = "tara: Hello there<|eot_id|>"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with model.inference_mode():
        out = model.generate(**inputs, max_new_tokens=8)
    print(tok.decode(out[0]))

if __name__ == "__main__":
    main()


