#!/usr/bin/env python
"""ステージ57 B・C・D: 再建群の重み × K × 刻みの評価。

``evaluate_stage51.py`` の評価を流用する。

規則は ``results/stage57/criteria.md``(集計・生成の前に記録)。評価からは、
ステージ55 の再建群 72 社と同じ EDINET コードの始点を除く。``--phase main`` は刻み n = 1 の
6 設定、``--phase steps`` は
「最良の組み合わせ」の刻み n = 1, 4, 20, 100 を評価する。

出力(``results/stage57/``): main は ``comparison.md``・``realdata.md``・``targets.md``・
``effects.md``・``judgment.md``、steps は ``comparison_steps.md``・``steps.md``(D.3・B.3)。

    uv run python scripts/evaluate_stage57.py --phase main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import evaluate_stage51 as e51
import numpy as np
import pandas as pd

from fsgen.accounting.constraints import load_canonical_spec, state_vector_names
from fsgen.models.industry import load_industry_config

OUT = Path("results/stage57")
_CS, _CSUR, _RE = ("jppfs_cor_CapitalStock", "jppfs_cor_CapitalSurplus",
                   "jppfs_cor_RetainedEarnings")
YEARS, THR = 5.0, 0.001
MAIN = {"s1_equal_k13": "1 等重み・K = 13", "s2_equal_k31": "2 等重み・K = 31",
        "s3_rec_k13": "3 再建群の重み・K = 13", "s3d_recdir_k13": "3d 再建群の重み(方向別)・K = 13",
        "s4_rec_k31": "4 再建群の重み・K = 31", "s4d_recdir_k31": "4d 再建群の重み(方向別)・K = 31"}
CANDS = ("s4_rec_k31", "s4d_recdir_k31", "s3_rec_k13", "s3d_recdir_k13")  # 優先順(criteria §4)
PLANS = {"s1_equal_k13": "data/interim/stage28/plans_equal",
         "s2_equal_k31": "data/interim/stage28/plans_equal",
         "s3_rec_k13": "data/interim/stage57/plans_rec", "s4_rec_k31": "data/interim/stage57/pla"
                                                                       "ns_rec",
         "s3d_recdir_k13": "data/interim/stage57/plans_recdir",
         "s4d_recdir_k31": "data/interim/stage57/plans_recdir"}
REAL = {"c2": "再建群(class 2 → 回復)", "c1": "再建群(class 1 → class 0)"}


def configure(settings: dict[str, str], base: str, title: str, out: dict[str, str]) -> None:
    e51._M = Path("data/interim/stage57/metrics")
    e51._GEN = Path("data/interim/stage57/gen")
    e51.SETTINGS = settings
    e51.BASE = base
    e51.PAIRS_F2 = ()
    e51.STAGE = "ステージ57"
    e51.TITLE = title
    e51.OUT = out
    pairs = pd.read_csv("results/stage55/pairs.csv", dtype={"edinetCode": str})
    codes = set(pairs["edinetCode"])
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    code_of = agg.drop_duplicates("docID").set_index("docID")["edinetCode"].astype(str)
    for cls in ("c2", "c1"):
        g = pd.read_parquet(sorted((e51._GEN / cls / base).glob("*__seed0.parquet"))[0])
        drop = [d for d in g["docID"].unique() if code_of.get(d, "") in codes]
        e51.EXCL[cls] = tuple(dict.fromkeys((*e51.EXCL[cls], *drop)))


def realdata_rows(settings: dict[str, str]) -> list[str]:
    th = json.loads(Path("results/stage56/theta.json").read_text(encoding="utf-8"))
    paid = pd.read_csv("results/stage56/paidin.csv", dtype={"edinetCode": str})
    rf = pd.read_csv("results/stage30/recovery_firms.csv", dtype={"edinetCode": str})
    lines = ["# ステージ57 C.2: 実データとの比較", "",
             "年率(総資産比 ±0.001、ステージ47・56 と同じ定義)。生成は 5 年 / 5、実データは 2 "
             "年の和 / 2。"
             "減らす深さ = 減らした企業の払込資本の比(終点 / 始点)の中央値。資本金 1% 未満 = 資"
             "本金を始点の 1% 未満にした"
             "(ステージ47 の定義)。利益剰余金が増えた = 年率 > 0.001。生成は企業ごとの seed 中"
             "央値、評価から除いた企業を除く。", "",
             "| class | 設定 | 払込資本を増やした | 減らした | \\|減らした − 実データ\\| | 減ら"
             "す深さ | 資本金 1% 未満 "
             "| 利益剰余金が増えた |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cls in ("c2", "c1"):
        g = th["groups"][REAL[cls]]
        real_down = g["down_annual"] / g["n_annual"]
        rr = paid[paid["group"] == REAL[cls]]
        codes = rr["edinetCode"].tolist()
        rre = rf.set_index("edinetCode").reindex(codes)["d_re"] if cls == "c2" else None
        re_txt = (f"{int((rre > THR).sum())} / {int(rre.notna().sum())}" if rre is not None
                  else "—")
        lines.append(f"| {cls} | **実データ({REAL[cls]})** | {g['up_annual']} / {g['n_annual']}"
                     f"({g['up_annual'] / g['n_annual']:.0%}) | {real_down:.0%} | — "
                     f"| {rr[rr['ratio'] < 1]['ratio'].median():.3f} | — | {re_txt} |")
        for s, lab in settings.items():
            gg = e51.load_gen(cls, s)
            gg = gg[~gg["docID"].isin(e51.EXCL[cls])]
            p0 = gg[f"x0_{_CS}"] + gg[f"x0_{_CSUR}"]
            p1 = gg[f"x1_{_CS}"] + gg[f"x1_{_CSUR}"]
            df = pd.DataFrame({"docID": gg["docID"], "an": (p1 - p0) / YEARS,
                               "ratio": p1 / p0.where(p0 > 0),
                               "lt1": (gg[f"x0_{_CS}"] > 1e-6)
                               & (gg[f"x1_{_CS}"] < 0.01 * gg[f"x0_{_CS}"]),
                               "re": (gg[f"x1_{_RE}"] - gg[f"x0_{_RE}"]) / YEARS})
            f = df.groupby("docID").agg(an=("an", "median"), ratio=("ratio", "median"),
                                        lt1=("lt1", "mean"), re=("re", "median"))
            n = len(f)
            down = (f["an"] < -THR).mean()
            lines.append(f"| {cls} | {lab} | {(f['an'] > THR).mean():.0%} | {down:.0%} "
                         f"| {abs(down - real_down):.2f} |"
                         f" {f[f['an'] < -THR]['ratio'].median():.3f} "
                         f"| {(f['lt1'] >= 2 / 3).mean():.0%} | {(f['re'] > THR).sum()} / {n} |")
    return lines + [""]


def targets_rows(settings: dict[str, str]) -> list[str]:
    agg = pd.read_csv("data/interim/values/aggregated_core.csv", dtype={"docID": str})
    agg = agg.drop_duplicates("docID").set_index("docID")
    paidr = (agg[_CS].fillna(0) + agg[_CSUR].fillna(0)) / agg["total_assets"]
    ind = load_industry_config(Path("conf/accounts/industry.yaml"))
    lines = ["# ステージ57 C.3: 対応先の性質(全社の計画、class 2 57 社)", "",
             "| 設定 | 対応先の払込資本比率(中央値) | 異なる対応先 | 業種一致率(33 業種) | 対応"
             "先の総資産 / 始点 |",
             "|---|---:|---:|---:|---:|"]
    seen = set()
    for s, lab in settings.items():
        d = PLANS[s]
        if d in seen:
            continue
        seen.add(d)
        t = pd.read_csv(Path(d) / "full_targets_nearest_na0.csv",
                        dtype={"source_docID": str, "target_docID": str})
        si = ind.labels(agg.loc[t["source_docID"], "edinetCode"].astype(str).tolist())
        ti = ind.labels(agg.loc[t["target_docID"], "edinetCode"].astype(str).tolist())
        match = np.mean([a == b and a != "不明" for a, b in zip(si, ti, strict=True)])
        lines.append(f"| {lab} | {paidr.loc[t['target_docID']].median():.3f} "
                     f"| {t['target_docID'].nunique()} | {match:.1%} |"
                     f" {t['asset_ratio'].median():.2f} |")
    return lines + [""]


def judge(names: tuple[str, ...], cands: tuple[str, ...], base: str
          ) -> tuple[list[str], dict[str, Any]]:
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    lines = ["| 設定 | class | 違反 | 記号(純資産の改善・解消率・順位相関) | 基準 1・2 |",
             "|---|---|---:|---|---|"]
    res: dict[str, Any] = {}
    for s in cands:
        oks = {}
        for cls in ("c2", "c1"):
            c = ctx[cls].compare(s, base, rng)
            nv = sum(ctx[cls].viol_items(s).values())
            ok, sym = e51.wins(c, nv)
            oks[cls] = ok
            lines.append(f"| {e51.SETTINGS[s]} | {cls} | {nv} | {sym} |"
                         f" {'満たす' if ok else '満たさない'} |")
        res[s] = oks
    return lines, res


def effects(names: tuple[str, ...]) -> list[str]:
    rng = np.random.default_rng(0)
    ctx = {c: e51.Ctx(c, names) for c in ("c2", "c1")}
    pairs = (("s2_equal_k31", "s1_equal_k13", "K の効果(等重み): 2 − 1"),
             ("s4_rec_k31", "s3_rec_k13", "K の効果(再建群の重み): 4 − 3"),
             ("s3_rec_k13", "s1_equal_k13", "重みの効果(K = 13): 3 − 1"),
             ("s4_rec_k31", "s2_equal_k31", "重みの効果(K = 31): 4 − 2"),
             ("s3d_recdir_k13", "s3_rec_k13", "方向別の効果(K = 13): 3d − 3"),
             ("s4d_recdir_k31", "s4_rec_k31", "方向別の効果(K = 31): 4d − 4"))
    lines = ["# ステージ57 D.2: K と重みの効果の分離", "",
             "企業単位のペアの差の中央値 [95% 区間](順位相関は企業の同時復元抽出)。"
             "交互作用 = (4 − 3) − (2 − 1) を、純資産の改善の点推定で示す。", "",
             "| 比較 | class | 純資産の改善 | 解消率 | 順位相関 | sparsity |", "|---|---|---|---"
                                                                "|---|---|"]
    point: dict[tuple[str, str], float] = {}
    for a, b, lab in pairs:
        for cls in ("c2", "c1"):
            c = ctx[cls].compare(a, b, rng)
            point[(lab, cls)] = float(c["純資産の改善"][0])
            lines.append(f"| {lab} | {cls} | {e51.fc(c['純資産の改善'])} | {e51.fc(c['解消率'])} "
                         f"| {e51.fc(c['順位相関'])} | {e51.fc(c['sparsity'])} |")
    lines.append("")
    for cls in ("c2", "c1"):
        inter = (point[("K の効果(再建群の重み): 4 − 3", cls)]
                 - point[("K の効果(等重み): 2 − 1", cls)])
        lines.append(f"- 交互作用({cls}、純資産の改善の点推定): {inter:+.4f}")
    return lines + [""]


def smoothness(cls: str, setting: str) -> dict[str, float]:
    """B.3: 隣接する t の変化量 / 全体の変化量 の分散(射影後・射影前)と、射影による移動量。"""
    var_p: list[float] = []
    var_r: list[float] = []
    cv_p: list[float] = []
    disp: list[float] = []
    n_t = 0
    for f in sorted((e51._GEN / cls / setting).glob("*__paths.npz")):
        z = np.load(f)
        keys = sorted({k.rsplit("__", 1)[0] for k in z.files if k.endswith("__raw")})
        for k in keys:
            raw, proj = z[f"{k}__raw"], z[f"{k}__proj"]
            n_t = len(raw)
            for arr, acc in ((proj, var_p), (raw, var_r)):
                tot = np.linalg.norm(arr[-1] - arr[0])
                if tot > 0 and len(arr) > 2:
                    acc.append(float(np.var(np.linalg.norm(np.diff(arr, axis=0), axis=1) / tot)))
            st = np.linalg.norm(np.diff(proj, axis=0), axis=1)
            if len(st) > 1 and st.mean() > 0:
                cv_p.append(float(st.std() / st.mean()))
            disp.extend(np.linalg.norm(proj - raw, axis=1)[1:].tolist())
    return {"n_t": n_t, "var_proj": float(np.median(var_p)) if var_p else float("nan"),
            "var_raw": float(np.median(var_r)) if var_r else float("nan"),
            "cv_proj": float(np.median(cv_p)) if cv_p else float("nan"),
            "disp_med": float(np.median(disp)), "disp_p90": float(np.percentile(disp, 90))}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("main", "steps"), required=True)
    ap.add_argument("--best", default="s1_equal_k13", help="steps: 最良の組み合わせの設定名")
    args = ap.parse_args()
    names = state_vector_names(load_canonical_spec(Path("conf/accounts/core.yaml")),
                               include_subtotals=False)
    if args.phase == "main":
        configure(MAIN, "s1_equal_k13", "ステージ57 C.1: 再建群の重みと K の比較(刻み n = 1)",
                  {"comparison": "comparison.md", "losses": "losses.md", "assets": "assets_extre"
                                                                                   "me.md",
                   "bs": "bs.md", "summary": "summary_c.json"})
        jl, res = judge(names, CANDS, "s1_equal_k13")
        passed = [s for s in CANDS if res[s]["c2"] and res[s]["c1"]]
        best = passed[0] if passed else "s1_equal_k13"
        e51.BS_ORDER = (best if best != "s1_equal_k13" else "s3_rec_k13",)
        sys.argv = [sys.argv[0], "--out", str(OUT)]
        e51.main()
        (OUT / "realdata.md").write_text("\n".join(realdata_rows(MAIN)) + "\n", encoding="utf-8")
        (OUT / "targets.md").write_text("\n".join(targets_rows(MAIN)) + "\n", encoding="utf-8")
        (OUT / "effects.md").write_text("\n".join(effects(names)) + "\n", encoding="utf-8")
        out = ["# ステージ57 D.1: 判定(刻み n = 1、対 設定 1)", "", *jl, "",
               f"基準を満たした設定: {', '.join(MAIN[s] for s in passed) or 'なし'}。",
               f"**最良の組み合わせ(刻みの比較に使う): {MAIN[best]}**", ""]
        (OUT / "judgment.md").write_text("\n".join(out) + "\n", encoding="utf-8")
        (OUT / "summary_main.json").write_text(json.dumps({"judge": res, "best": best},
                                                          ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        print("\n".join(out))
        return
    steps = {f"{args.best}": f"n = 1({MAIN[args.best]})",
             **{f"{args.best}_n{n}": f"n = {n}" for n in (4, 20, 100)}}
    steps = {k: v for k, v in steps.items() if (Path("data/interim/stage57/gen/c2") / k).is_dir()}
    configure(steps, args.best, f"ステージ57 D.3: 刻みの比較({MAIN[args.best]})",
              {"comparison": "comparison_steps.md", "losses": "losses_steps.md",
               "assets": "assets_steps.md", "bs": "bs_steps.md", "summary": "summary_steps.json"})
    e51.BS_ORDER = tuple(k for k in steps if k != args.best)
    sys.argv = [sys.argv[0], "--out", str(OUT)]
    e51.main()
    jl, res = judge(names, tuple(k for k in steps if k != args.best), args.best)
    lines = ["# ステージ57 D.3・B.3: 刻みの効果", "",
             f"「最良の組み合わせ」= {MAIN[args.best]}。n = 4・20・100 を n = 1 と比べる(基準 1"
             f"〜3)。", "",
             *jl, "", "## B.3 経路の滑らかさと計算時間", "",
             "変化量の分散 = 隣接する t の ‖x(t_{k+1}) − x(t_k)‖ / ‖x(1) − x(0)‖ の分散(行ごと"
             "、中央値)。"
             "射影がなければ、射影前の経路でも等間隔の t で変化量は一定ではない(Euler 法の速度は"
             " t と z で変わる)。"
             "射影による移動 = ‖x(t) − x̂(t)‖(射影後 − 射影前、t > 0 の全点)。計算時間 = 学習・"
             "積分・射影の合計"
             "(seed 0、class 2 は 5 fold の合計、class 1 は積分と射影)。", "",
             "変動係数 = 射影後の隣接する変化量の標準偏差 / 平均"
             "(n によらず比べられる。0 なら等速)。", "",
             "| 刻み | class | 点の数 | 変化量の分散(射影後) | (射影前) | 変動係数(射影後) "
             "| 射影による移動 中央値 / p90 | 計算時間(秒) |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for k, lab in steps.items():
        for cls in ("c2", "c1"):
            sm = smoothness(cls, k)
            log = Path(f"data/interim/stage57/logs/gen_{cls}_{k}_0.txt").read_text(encoding="utf-8")
            sec = [ln for ln in log.splitlines() if "時間" in ln]
            t_txt = sec[-1].split("時間")[1].split("秒")[0].strip() if sec else "—"
            lines.append(f"| {lab} | {cls} | {sm['n_t']} | {sm['var_proj']:.2e} "
                         f"| {sm['var_raw']:.2e} | {sm['cv_proj']:.3f} "
                         f"| {sm['disp_med']:.4f} / {sm['disp_p90']:.4f} | {t_txt} |")
    passed = [k for k in steps if k != args.best and res[k]["c2"] and res[k]["c1"]]
    step_txt = (("n を増やす: " + steps[passed[0]]) if passed
                else "どの n も基準を満たさない → n = 1 のまま")
    lines += ["", f"**刻みの判定**: {step_txt}", ""]
    (OUT / "steps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
