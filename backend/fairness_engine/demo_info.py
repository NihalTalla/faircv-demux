"""Read-only structural information about the frozen FairCV demo dataset.

Used by the API to describe the demo dataset (rows, columns, groups) and by
the frontend's setup page. Cached by (path, mtime) so repeated calls do not
re-read the 203 MB file. This module never writes anything.
"""

from pathlib import Path

import numpy as np

from app.config import settings

_cache: dict = {}


def demo_info() -> dict | None:
    path = settings.FAIRCV_DB_PATH
    if not Path(path).exists():
        return None

    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return None
    if _cache.get("mtime") == mtime and _cache.get("path") == str(path):
        return _cache["info"]

    db = np.load(path, allow_pickle=True).item()
    P_tr, P_te = db["Profiles Train"], db["Profiles Test"]

    gender_te = P_te[:, 1].astype(int)
    ethnicity_te = P_te[:, 0].astype(int)
    gender_counts = {
        {0: "Male", 1: "Female"}[int(v)]: int((gender_te == v).sum())
        for v in np.unique(gender_te)
    }
    eth_counts = {
        {0: "G1", 1: "G2", 2: "G3"}[int(v)]: int((ethnicity_te == v).sum())
        for v in np.unique(ethnicity_te)
    }

    info = {
        "rows": int(P_tr.shape[0]) + int(P_te.shape[0]),
        "train_rows": int(P_tr.shape[0]),
        "test_rows": int(P_te.shape[0]),
        "columns": int(P_tr.shape[1]),
        "columns_list": [
            "ethnicity (0)", "gender (1)", "occupation (2)", "suitability (3)",
            *[f"CV competency {i}" for i in range(4, 11)],
            *[f"face embedding {i}" for i in range(11, 31)],
            *[f"blind face block {i}" for i in range(31, 51)],
        ],
        "groups": {"gender": gender_counts, "ethnicity": eth_counts},
        "keys": sorted(str(k) for k in db.keys()),
    }

    _cache["path"] = str(path)
    _cache["mtime"] = mtime
    _cache["info"] = info
    return info
