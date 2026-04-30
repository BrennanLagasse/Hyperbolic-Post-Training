
from eval_utils import save_results

metrics = {
    "correct": 745,
    "parse_failure": 5,
    "total_samples": 1000,
}

save_results(
    model_name="Qwen3-1.7B-SFT",
    model_path="./runs/qwen3-openorca/final",
    dataset_name="ARC-Challenge",
    dataset_path="allenai/ai2_arc",
    metrics=metrics,
    results_dir="./results"
)