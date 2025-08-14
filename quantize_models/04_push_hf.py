import os
from huggingface_hub import HfApi, create_repo, upload_folder

SRC = os.environ.get("SRC", "./awq_model")
REPO_ID = os.environ.get("REPO_ID", "your-org/orpheus-3b-awq-6bit")

def main():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    assert token, "Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN"
    api = HfApi(token=token)
    try:
        create_repo(repo_id=REPO_ID, token=token, repo_type="model")
    except Exception:
        pass
    print(f"Uploading {SRC} -> {REPO_ID}")
    upload_folder(
        folder_path=SRC,
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
        commit_message="Add AWQ 6-bit quantized Orpheus",
    )
    print("Done.")

if __name__ == "__main__":
    main()


