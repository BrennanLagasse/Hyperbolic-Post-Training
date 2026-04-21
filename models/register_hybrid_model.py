"""
Register the HybridQwen model for loading at later points
"""

from hybrid_qwen import HyperbolicQwen, HyperbolicQwenConfig

from transformers import AutoConfig, AutoModelForCausalLM

# Initialize new config with the same values as the pretrained model up to architecture changes
base_config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
new_config = HyperbolicQwenConfig(**base_config.to_dict(), k=1.0)

print("Initializing Model...")
model = HyperbolicQwen(new_config)
print("Model initialized.")

print("\nLoading model weights...")
model.initialize_from_pretrained()

print("\nRegistering config...")
AutoConfig.register(
    "hyperbolic_qwen",
    HyperbolicQwenConfig
)

print("\nRegistering model...")
AutoModelForCausalLM.register(
    HyperbolicQwenConfig,
    HyperbolicQwen
)

print("\nSaving model...")
model.save_pretrained("hyperbolic-qwen")

print("\nLoading model...")
model_new = AutoModelForCausalLM.from_pretrained(
    "hyperbolic-qwen"
)

print("\nDone!")