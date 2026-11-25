"""ステージ83 の共通の定義(pandas だけ。どのプロセスからも使える)。

規則は ``results/stage83/criteria_stage83.md``。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/interim/stage83")
OUT = Path("results/stage83")
D82 = Path("data/interim/stage82")
_PANEL = Path("data/interim/stage23/panel/panel.csv")
FOLDS = ("f0", "f1", "f2", "f3", "f4")
MAIN = "KDE0p"  # 主モデル: KDE0 + indiv_scale = False + stage1_penalty
BASE = "KDE0"  # 比較: KDE0 + base
# D1 の表の設定名(パートA の summary.json のキー)
NAMES = {"KDE0p": "KDE0(False・stage1_penalty)", "KDE0": "KDE0(False・base)",
         "R1p": "R1(False・stage1_penalty)"}


def recovery_groups() -> dict[str, list[str]]:
    """バックテストの群: fold f のホールドアウト(57 社の D_f)と、class 2 → 回復の 15 社(c2rec)。"""
    w75 = json.loads(Path("data/interim/stage75/weights.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    out: dict[str, list[str]] = {f: list(map(str, w75["folds"][f])) for f in FOLDS}
    out["c2rec"] = sorted(pairs[pairs["group"] == "class 2 → 回復"]["edinetCode"])
    return out


def compositions(codes: list[str], names: tuple[str, ...]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """2022・2024 年度の構成(各年度の総資産比)と、2022・2024 年度の総資産(百万円ではなく円のまま)。

    両年度とも総資産が正の企業だけを返す(返した企業コードの並び)。
    """
    p = pd.read_csv(_PANEL, dtype={"edinetCode": str})
    a = p[p["t"] == 2022].set_index("edinetCode")
    b = p[p["t"] == 2024].set_index("edinetCode")
    ok = [c for c in codes if c in a.index and c in b.index
          and a.loc[c, "cur_total_assets"] > 0 and b.loc[c, "cur_total_assets"] > 0]
    cols = [f"cur_{n}" for n in names]
    ta = a.loc[ok, "cur_total_assets"].to_numpy(float)
    tb = b.loc[ok, "cur_total_assets"].to_numpy(float)
    x0 = a.loc[ok, cols].fillna(0).to_numpy(float) / ta[:, None]
    x1 = b.loc[ok, cols].fillna(0).to_numpy(float) / tb[:, None]
    return x0, x1, np.column_stack([ta, tb]), ok
