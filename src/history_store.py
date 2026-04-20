from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import HISTORY_PATH


def append_predictions(records: Iterable[dict], history_path: Path = HISTORY_PATH) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(list(records))
    if incoming.empty:
        return

    incoming["predicted_at"] = datetime.utcnow().isoformat(timespec="seconds")

    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined.to_csv(history_path, index=False)


def read_history(history_path: Path = HISTORY_PATH) -> pd.DataFrame:
    if history_path.exists():
        return pd.read_csv(history_path)
    return pd.DataFrame()
