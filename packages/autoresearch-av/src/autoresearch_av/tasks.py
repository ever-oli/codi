from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    modality: str
    problem: str
    dataset: str
    dataset_url: str
    metric: str
    label_count: str
    notes: str


VISION_TASKS = {
    "food101": TaskSpec(
        name="food101",
        modality="vision",
        problem="food classification",
        dataset="Food-101",
        dataset_url="https://huggingface.co/datasets/ethz/food101",
        metric="accuracy",
        label_count="101",
        notes="Large but standard image classification benchmark.",
    ),
    "beans": TaskSpec(
        name="beans",
        modality="vision",
        problem="plant disease classification",
        dataset="beans",
        dataset_url="https://huggingface.co/datasets/randall-lab/beans",
        metric="accuracy",
        label_count="3",
        notes="Good first-pass small benchmark.",
    ),
    "trashnet": TaskSpec(
        name="trashnet",
        modality="vision",
        problem="waste and recycling classification",
        dataset="TrashNet",
        dataset_url="https://huggingface.co/datasets/garythung/trashnet",
        metric="accuracy",
        label_count="6",
        notes="Small and practical image classification task.",
    ),
    "cub200": TaskSpec(
        name="cub200",
        modality="vision",
        problem="fine-grained bird classification",
        dataset="CUB-200-2011",
        dataset_url="https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011",
        metric="accuracy",
        label_count="200",
        notes="Fine-grained benchmark; harder than the starter tasks.",
    ),
    "flowers102": TaskSpec(
        name="flowers102",
        modality="vision",
        problem="flower classification",
        dataset="Flowers102",
        dataset_url="https://huggingface.co/datasets/pufanyi/flowers102",
        metric="accuracy",
        label_count="102",
        notes="Clean image classification benchmark with balanced scale.",
    ),
    "oxford_pet": TaskSpec(
        name="oxford_pet",
        modality="vision",
        problem="pet breed classification",
        dataset="Oxford-IIIT Pet",
        dataset_url="https://huggingface.co/datasets/timm/oxford-iiit-pet",
        metric="accuracy",
        label_count="37",
        notes="Good medium-size image classification benchmark.",
    ),
}

AUDIO_TASKS = {
    "esc50": TaskSpec(
        name="esc50",
        modality="audio",
        problem="environmental sound classification",
        dataset="ESC-50",
        dataset_url="https://huggingface.co/datasets/renumics/esc50",
        metric="accuracy",
        label_count="50",
        notes="Clean broad audio tagging benchmark.",
    ),
    "nsynth": TaskSpec(
        name="nsynth",
        modality="audio",
        problem="instrument classification",
        dataset="NSynth",
        dataset_url="https://huggingface.co/datasets/jg583/NSynth",
        metric="accuracy",
        label_count="11+",
        notes="Large audio benchmark; likely subset first.",
    ),
    "gtzan": TaskSpec(
        name="gtzan",
        modality="audio",
        problem="music genre classification",
        dataset="GTZAN",
        dataset_url="https://huggingface.co/datasets/marsyas/gtzan",
        metric="accuracy",
        label_count="10",
        notes="Classic benchmark with known label noise.",
    ),
    "ravdess": TaskSpec(
        name="ravdess",
        modality="audio",
        problem="speech emotion classification",
        dataset="RAVDESS",
        dataset_url="https://huggingface.co/datasets/narad/ravdess",
        metric="accuracy",
        label_count="8",
        notes="Useful benchmark with licensing caveats for commercial use.",
    ),
    "speech_commands": TaskSpec(
        name="speech_commands",
        modality="audio",
        problem="keyword spotting",
        dataset="Speech Commands",
        dataset_url="https://huggingface.co/datasets/google/speech_commands",
        metric="accuracy",
        label_count="35",
        notes="Best starter task for bounded audio experiments.",
    ),
}

TASKS = {**VISION_TASKS, **AUDIO_TASKS}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{name}'. Valid tasks: {valid}") from exc
