#!/usr/bin/env python
"""ステージ16 §A.4: 貸倒引当金を canonical に追加した影響の検証(旧30科目 vs 新32科目)。

旧: data/interim/canonical_v30/{core.yaml, aggregated_core.csv, residuals.csv}
新: conf/accounts/core.yaml, data/interim/values/{aggregated_core.csv, residuals.csv}
確認: 対象社の同一性 / 残余科目が負の社数 / 新科目の符号 / 下限0を緩和なしで課せるか /
      D・rank(C)・実効次元 / 恒等式残差 r1〜r4。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fsgen.accounting.constraints import (
    build_constraint_matrix,
    compute_rank,
    load_canonical_spec,
    state_vector_names,
)

_OLD = Path("data/interim/canonical_v30")


def load(agg_path: Path, spec_path: Path) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray]:
    spec = load_canonical_spec(spec_path)
    names = state_vector_names(spec, include_subtotals=False)
    agg = pd.read_csv(agg_path, dtype={"docID": str})
    agg = agg[(agg["parse_status"] == "ok") & (agg["is_financial"].astype(str) != "True")]
    agg = agg[agg["total_assets"].notna() & (agg["total_assets"] != 0)].reset_index(drop=True)
    c = build_constraint_matrix(spec, include_subtotals=False)
    return agg, names, c


def main() -> None:  # noqa: PLR0915
    old, n_old, c_old = load(_OLD / "aggregated_core.csv", _OLD / "core.yaml")
    new, n_new, c_new = load(Path("data/interim/values/aggregated_core.csv"),
                             Path("conf/accounts/core.yaml"))
    print(f"対象社(非金融・ok): 旧{len(old)} 新{len(new)} 同一={old['docID'].equals(new['docID'])}")
    print(f"total_assets同一={np.array_equal(old['total_assets'], new['total_assets'])}")
    print(f"\nD(状態ベクトル次元): {len(n_old)} → {len(n_new)}")
    print(f"制約行列 C の形: {c_old.shape} → {c_new.shape}")
    print(f"rank(C): {compute_rank(c_old)} → {compute_rank(c_new)}")
    ta_o, ta_n = old["total_assets"].to_numpy(float), new["total_assets"].to_numpy(float)
    xo = old[list(n_old)].fillna(0.0).to_numpy(float) / ta_o[:, None]
    xn = new[list(n_new)].fillna(0.0).to_numpy(float) / ta_n[:, None]
    for label, x in (("旧", xo), ("新", xn)):
        xc = x - x.mean(axis=0)
        s = np.linalg.svd(xc, compute_uv=False) ** 2
        cum = np.cumsum(s) / s.sum()
        pr = s.sum() ** 2 / np.sum(s**2)
        print(f"実効次元({label}): 分散90%={int(np.searchsorted(cum, .9)) + 1} "
              f"95%={int(np.searchsorted(cum, .95)) + 1} 99%={int(np.searchsorted(cum, .99)) + 1} "
              f"参加比={pr:.2f}")

    print("\n=== 残余科目が負になる社数(総資産比 < 0 / < -1e-3 / < -1e-2) ===")
    rows = []
    for nm in n_new:
        if not nm.startswith("__OTHER__"):
            continue
        vn = xn[:, n_new.index(nm)]
        vo = xo[:, n_old.index(nm)] if nm in n_old else np.full(len(xo), np.nan)
        rows.append({
            "残余科目": nm[9:], "旧<0": int((vo < 0).sum()), "新<0": int((vn < 0).sum()),
            "旧<-1e-3": int((vo < -1e-3).sum()), "新<-1e-3": int((vn < -1e-3).sum()),
            "旧<-1e-2": int((vo < -1e-2).sum()), "新<-1e-2": int((vn < -1e-2).sum()),
            "新最小": float(vn.min()),
        })
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 新科目の符号(値あり=非0) ===")
    for nm in ("allowance_doubtful_current", "allowance_doubtful_noncurrent"):
        v = new[nm].dropna()
        v = v[v != 0]
        print(f"{nm}: 非0 {len(v)}社({len(v) / len(new):.1%}) 負{int((v < 0).sum())} "
              f"正{int((v > 0).sum())} 最小比{float((new[nm].fillna(0) / ta_n).min()):.4f}")
    j = n_new.index("jppfs_cor_InvestmentsAndOtherAssets")
    jo = n_old.index("jppfs_cor_InvestmentsAndOtherAssets")
    print(f"IOA(投資その他の資産)の中央値 総資産比: 旧(純額){np.median(xo[:, jo]):.4f} → "
          f"新(総額){np.median(xn[:, j]):.4f}")

    print("\n=== 恒等式残差 r1〜r4(対象社のみ) ===")
    for label, path, ids in (("旧", _OLD / "residuals.csv", old["docID"]),
                             ("新", Path("data/interim/values/residuals.csv"), new["docID"])):
        r = pd.read_csv(path, dtype={"docID": str}).set_index("docID").loc[ids]
        print(f"{label}: max r1〜r4 = {r[['r1', 'r2', 'r3', 'r4']].max().to_dict()} "
              f"非0社数={(r[['r1', 'r2', 'r3', 'r4']] != 0).sum().to_dict()}")
    print(f"制約行列での残差 max|C^T x| (tilde): 旧 {np.abs(xo @ c_old).max():.2e} "
          f"新 {np.abs(xn @ c_new).max():.2e}")

    lab = pd.read_csv("data/interim/labels/labels.csv", dtype={"docID": str})
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()]
    for label, agg in (("旧", old), ("新", new)):
        m = agg.merge(lab[["docID", "label"]], on="docID")
        lb = m["label"].astype(int)
        print(f"ラベル数({label}): class0={int((lb == 0).sum())} class2={int((lb == 2).sum())}")


if __name__ == "__main__":
    main()
