# Ended up using these resources for flash attention:
# https://mjunya.com/flash-attention-prebuild-wheels/?python=3.10&torch=2.9&cuda=12.9

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    attn_implementation="flash_attention_2",
    torch_dtype="auto",  # flash-attn requires float16 or bfloat16, not float32
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

