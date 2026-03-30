from datasets import load_dataset

ds = load_dataset("Open-Orca/OpenOrca", split="train[:23000]")

print(ds)

def format_example(example: dict) -> str:
    """
    OpenOrca fields: system_prompt, question, response
    Format into ChatML template used by Qwen models.
    """
    system = example.get("system_prompt", "").strip()
    question = example.get("question", "").strip()
    response = example.get("response", "").strip()

    parts = []
    if system:
        parts.append(f"{SYSTEM_TOKEN}{system}{END_TOKEN}")
    parts.append(f"{USER_TOKEN}{question}{END_TOKEN}")
    parts.append(f"{ASSISTANT_TOKEN}{response}{END_TOKEN}")

    return "".join(parts)