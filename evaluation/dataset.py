import json
from pathlib import Path
from typing import Any


DATASET_PATH = Path(__file__).parent / "data" / "synthetic_contracts.json"


def load_dataset() -> dict[str, Any]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)
