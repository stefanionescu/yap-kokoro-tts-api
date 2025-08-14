import os
from huggingface_hub import snapshot_download

BASE_REPO = os.environ.get("BASE_REPO", "canopylabs/orpheus-3b-0.1-ft")
OUT_DIR = os.environ.get("OUT_DIR", "./base_model")

def main():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=OUT_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "*.model", "*.spm", "*.tiktoken*",
        ],
        ignore_patterns=[
            "optimizer*.bin",
            "pytorch_model_fsdp.bin",
            "training_args*",
            "trainer_state*",
            "rng_state_*.pth",
            "scheduler.pt",
            "*.bin",
        ],
        token=token,
    )
    print(f"Downloaded base to {OUT_DIR}")

if __name__ == "__main__":
    main()


