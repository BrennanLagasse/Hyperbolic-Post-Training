"""
Register the HybridQwen model for loading at later points
"""

from hybrid_qwen import HyperbolicQwen, HyperbolicQwenConfig

from transformers import AutoConfig, AutoTokenizer

# Initialize new config with the same values as the pretrained model up to architecture changes
base_config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
new_config = HyperbolicQwenConfig(**base_config.to_dict(), k=1.0)

print("Initializing Model...")
model = HyperbolicQwen(new_config)
print("Model initialized.")

print("\nLoading model weights...")
model.initialize_from_pretrained()

print("\nSanity forward pass")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
prompt = "The sky is blue because"
inputs = tokenizer(prompt, return_tensors="pt")
out = model(inputs)



print("\nDone!")