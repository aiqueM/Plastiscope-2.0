from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

best_dir = "./best"
base_repo = "zhihan1996/DNABERT-2-117M"

tokenizer = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True)
model = AutoModel.from_pretrained(base_repo, trust_remote_code=True)

state_dict = load_file(f"{best_dir}/model.safetensors")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("✓ Loaded fine-tuned weights into base model")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

