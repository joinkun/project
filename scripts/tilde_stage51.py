"""総資産比の状態ベクトル x̃ の読み込み(sklearn・torch・cvxpy・POT を読み込まない)。

``density_stage51`` から分離した(ステージ67)。cvxpy や POT のプロセスから使っても、sklearn が
同じプロセスに入らないようにするため。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.datasets.exclusion import exclusion_masks, load_excluded_docs

_AGG = Path("data/interim/values/aggregated_core.csv")
_LABELS = Path("data/interim/labels/labels.csv")
_EXCLUDE = Path("data/interim/stage28/excluded.csv")


def load_tilde(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """``run_generation_stage16.load_dataset`` と同じ抽出(torch を読み込まないために複製)。"""
    agg = pd.read_csv(_AGG, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    lab = pd.read_csv(_LABELS, dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    m = agg.merge(lab[["docID", "filerName", "label"]], on="docID", how="inner")
    m = m[m["total_assets"].notna() & (m["total_assets"] != 0)]
    row_keep, src_keep = exclusion_masks(m["docID"].tolist(), m["label"].astype(int).tolist(),
                                         load_excluded_docs(_EXCLUDE))
    m = m[row_keep]
    x = m[list(names)].fillna(0.0).to_numpy(dtype=float) / m["total_assets"].to_numpy()[:, None]
    return x, m["label"].astype(int).to_numpy(), m["docID"].tolist(), src_keep
