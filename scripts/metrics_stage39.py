#!/usr/bin/env python
"""ステージ39 C・D: 1 つのルート(D = 32 か 38)の生成を指標にする(lightgbm/sklearn のプロセス)。

作業ディレクトリのルートの canonical・集計・パネルを使う。``--gen-dir`` の下の
``c2/<設定>/`` と ``c1/<設定>/``(各 ``<設計>__seed{0,1,2}.parquet``)を読み、``--out`` に書く:

- ``rows_{c2,c1}.parquet``: 行(設定 × seed × 企業)ごとの指標(``run_stage25.rows_for_version``)。
  ``proximity_32`` は共通の 32 科目に集約した空間での等重み proximity(D の比較用)
- ``viol_{c2,c1}.csv``: 13 項目のチェックで違反した行(企業・チェック)
- ``deltas_{c2,c1}.parquet``: 企業ごとの科目の変化(seed の中央値)。元の科目と、32 科目への集約
- ``refs.parquet``: 実データの参照(科目の変化の中央値)。class 2 は回復 15 社の 2 年
  (ステージ38 の連結と同じ)、class 1 は class 1 → class 0 の 1 年の遷移(ステージ38 の T2 と
  同じ組)。元の科目と 32 科目への集約

validity の分類器は class 1 を含めない(class 0 対 class 2、ステージ38 の除外、seed 42)。

    cd data/stage39_root && uv run --project ../.. python ../../scripts/metrics_stage39.py \\
        --out ../../data/interim/stage39/metrics/expanded
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from evaluate_stage34 import load_real_no_class1  # noqa: E402
from run_stage25 import rows_for_version, select_pcfm  # noqa: E402

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names  # noqa: E402
from fsgen.accounting.plausibility import (  # noqa: E402
    CHECK_NAMES,
    EXTRA_CHECK_NAMES,
    evaluate_checks,
    evaluate_extra_checks,
    subtotal_members,
)
from fsgen.evaluation.classifiers import train_classifiers  # noqa: E402

_PROJECT = Path(__file__).resolve().parent.parent
_CORE = _PROJECT / "conf/accounts/core.yaml"
# 細分化で開いた小計 → その子(32 科目への集約)。貸倒引当金(固定)は core でも基礎科目
OPENED = {
    "jppfs_cor_PropertyPlantAndEquipment": (
        "land", "buildings_net", "machinery_net", "construction_in_progress",
        "__OTHER__jppfs_cor_PropertyPlantAndEquipment"),
    "jppfs_cor_InvestmentsAndOtherAssets": (
        "investment_securities", "deferred_tax_assets_nc",
        "__OTHER__jppfs_cor_InvestmentsAndOtherAssets"),
}


def core_names() -> tuple[str, ...]:
    return state_vector_names(load_canonical_spec(_CORE), include_subtotals=False)


def to_core(df: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """列が ``names`` の表を core の 32 科目に集約する(core なら並べ替えるだけ)。"""
    out = {}
    for n in core_names():
        if n in names:
            out[n] = df[n]
        elif n in OPENED:
            out[n] = df[list(OPENED[n])].sum(axis=1)
        else:
            raise KeyError(f"{n} を 32 科目に集約できない")
    return pd.DataFrame(out, index=df.index)


def load_gen(directory: Path) -> tuple[pd.DataFrame, str]:
    files = sorted(directory.glob("*__seed*.parquet"))
    tags = {f.name.split("__seed")[0] for f in files}
    if len(files) != 3 or len(tags) != 1:
        raise FileNotFoundError(f"{directory}: 1 設計 × 3 seed のはず({[f.name for f in files]})")
    g = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return select_pcfm(g).reset_index(drop=True), tags.pop()


def real_refs(names: tuple[str, ...]) -> pd.DataFrame:
    pan = pd.read_csv("data/interim/stage23/panel/panel.csv", dtype={"edinetCode": str})

    def deltas(pp: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame({n: (pp[f"cur_{n}"].fillna(0) - pp[f"prev_{n}"].fillna(0))
                          / pp["prev_total_assets"] for n in names})
        d["edinetCode"] = pp["edinetCode"].to_numpy()
        return d

    y30 = pd.read_csv(_PROJECT / "results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    y30 = y30[(y30["group"] == "Y") & (y30["reason"].fillna("") == "") & y30["in_panel"]]
    pp = pan[pan["edinetCode"].isin(y30["edinetCode"]) & pan["t"].isin([2023, 2024])]
    c2 = deltas(pp).groupby("edinetCode").sum()
    r38 = pd.read_parquet(_PROJECT / "data/interim/stage38/real_deltas.parquet")
    t2 = r38[(r38["label_prev"] == 1) & (r38["label_cur"] == 0)][["edinetCode", "t"]]
    pp = pan.merge(t2, on=["edinetCode", "t"], how="inner")
    if len(pp) != len(t2):
        raise ValueError(f"T2 の組 {len(t2)} のうちパネルにあるのは {len(pp)}")
    c1 = deltas(pp).drop(columns="edinetCode")
    rows = []
    for cls, d, n in (("c2", c2, len(y30)), ("c1", c1, len(t2))):
        for space, dd in (("native", d[list(names)]), ("core32", to_core(d, names))):
            s = dd.median()
            rows.append(pd.DataFrame({"cls": cls, "space": space, "account": s.index,
                                      "median": s.to_numpy(), "n": n}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", type=Path, default=Path("data/interim/stage39/gen"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    spec = load_canonical_spec(Path("conf/accounts/core.yaml"))
    names = state_vector_names(spec, include_subtotals=False)
    members = subtotal_members(spec, names)
    excl = set(pd.read_csv("data/interim/stage38/excluded.csv", dtype={"docID": str})["docID"])
    x_real, is2 = load_real_no_class1(names, excl)
    clfs = train_classifiers(x_real, is2, seed=42)
    args.out.mkdir(parents=True, exist_ok=True)
    real_refs(names).to_parquet(args.out / "refs.parquet", index=False)
    for cls in ("c2", "c1"):
        if not (args.gen_dir / cls).is_dir():  # ステージ51: class 2 だけの再評価
            continue
        rows, viols, dels = [], [], []
        for sdir in sorted(p for p in (args.gen_dir / cls).iterdir() if p.is_dir()):
            g, tag = load_gen(sdir)
            r, _ = rows_for_version(args.gen_dir / cls, sdir.name, names, members, clfs, tag)
            x0 = g[[f"x0_{n}" for n in names]].to_numpy()
            x1 = g[[f"x1_{n}" for n in names]].to_numpy()
            d = pd.DataFrame(x1 - x0, columns=list(names))
            r["proximity_32"] = np.linalg.norm(to_core(d, names).to_numpy(), axis=1)
            r["infeasible"] = (g["status"] == "infeasible").to_numpy()
            rows.append(r)
            fl = np.column_stack([evaluate_checks(x0, x1, spec, names),
                                  evaluate_extra_checks(x1, x0, spec, names)])
            vf = pd.DataFrame(fl.astype(int), columns=[*CHECK_NAMES, *EXTRA_CHECK_NAMES])
            vf.insert(0, "docID", g["docID"].to_numpy())
            viols.append(vf.melt(id_vars="docID", var_name="check", value_name="rows")
                         .query("rows > 0").assign(setting=sdir.name))
            d["docID"] = g["docID"].to_numpy()
            med = d.groupby("docID").median()
            for space, dd in (("native", med), ("core32", to_core(med, names))):
                long = dd.reset_index().melt(id_vars="docID", var_name="account",
                                             value_name="delta")
                dels.append(long.assign(setting=sdir.name, space=space))
            print(f"[{cls}] {sdir.name}: {tag}、{g['docID'].nunique()} 社、{len(g)} 行", flush=True)
        pd.concat(rows, ignore_index=True).to_parquet(args.out / f"rows_{cls}.parquet",
                                                      index=False)
        pd.concat(viols, ignore_index=True).to_csv(args.out / f"viol_{cls}.csv", index=False)
        pd.concat(dels, ignore_index=True).to_parquet(args.out / f"deltas_{cls}.parquet",
                                                      index=False)


if __name__ == "__main__":
    main()
