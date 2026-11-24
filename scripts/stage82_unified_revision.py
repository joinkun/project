#!/usr/bin/env python
"""ステージ82 の統合実行スクリプト(各段階を別のプロセスで起動する。重いライブラリは読み込まない)。

規則は ``results/stage82/criteria.md``。プロセスの分離を守るため、段階ごとに次のスクリプトを
起動する。

- ``costs``: ``stage82_costs.py``(numpy)
- ``ot``: ``stage82_ot.py``(POT・numpy)
- ``qp``: ``stage82_qp.py``(cvxpy)
- ``metrics``: ``metrics_stage39.py``(lightgbm)
- ``evaluate``: ``stage82_evaluate.py``(lightgbm、D1 の比較と主モデルの選択)
- ``full``: 主モデルの単一の生成(制約の 3 階層・苦境全体のコスト)と指標
- ``final``: ``stage82_final.py``(lightgbm、実験 3〜6 の表と ``report.md``)

    uv run python scripts/stage82_unified_revision.py --steps costs,ot,qp,metrics,evaluate
    uv run python scripts/stage82_unified_revision.py --steps full,final
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from stage82_common import COSTS, D1_SETTINGS, DATA, EXP2, FOLDS, OUT

LOG = DATA / "logs"
OT_PAIRS = ("R1:prop,KDE0:prop,IPW:prop,scale:prop,A1:prop,R1i:prop,R1p:prop,R1ip:prop,"
            "R1:pure_lp,R1:top5,R1:top8,R1:sinkhorn,R1:row,R1:col")


def run_jobs(jobs: list[tuple[list[str], str]], workers: int) -> None:
    """(コマンド, ログ名) を並列に実行する。どれかが失敗したら例外。"""
    LOG.mkdir(parents=True, exist_ok=True)

    def one(job: tuple[list[str], str]) -> int:
        cmd, name = job
        with (LOG / f"{name}.txt").open("w", encoding="utf-8") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    with ThreadPoolExecutor(max_workers=workers) as ex:
        codes = list(ex.map(one, jobs))
    bad = [name for (_c, name), rc in zip(jobs, codes, strict=True) if rc != 0]
    if bad:
        raise RuntimeError(f"失敗したジョブ: {bad[:10]}(全 {len(bad)} 件、ログは {LOG})")
    print(f"  {len(jobs)} 件のジョブが終わった", flush=True)


def py(script: str, *args: str) -> list[str]:
    return ["uv", "run", "python", f"scripts/{script}", *args]


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    steps = args.steps.split(",")
    w = args.workers
    if "costs" in steps:
        run_jobs([(py("stage82_costs.py"), "costs")], 1)
    if "ot" in steps:
        run_jobs([(py("stage82_ot.py", "--cls", c, "--cfg", g, "--pairs", OT_PAIRS),
                   f"ot_{c}_{g}") for c in ("c1", "c2") for g in (*FOLDS, "full")], w)
    if "qp" in steps:
        jobs = []
        for st in D1_SETTINGS.values():
            for c in ("c1", "c2"):
                for g in FOLDS:
                    for i in (0, 1):
                        jobs.append((py("stage82_qp.py", "--cls", c, "--cfg", g, "--group",
                                        st.group, "--cost", st.cost, "--ot", st.ot, "--qp", st.qp,
                                        "--tier", st.tier, "--shard", f"{i}/2"),
                                     f"qp_{st.group}_{c}_{g}_{st.tag}_{i}"))
        run_jobs(jobs, w)
    if "metrics" in steps:
        groups = sorted({st.group for st in D1_SETTINGS.values()})
        run_jobs([(py("metrics_stage39.py", "--gen-dir", str(DATA / grp / f"gen_{g}"), "--out",
                      str(DATA / grp / f"metrics_{g}")), f"metrics_{grp}_{g}")
                  for grp in groups for g in FOLDS], min(w, 8))
    if "evaluate" in steps:
        run_jobs([(py("stage82_evaluate.py"), "evaluate")], 1)
    if "full" in steps:
        sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
        best = EXP2[sel["best"]]
        alld = f"{best.cost}_all"
        if alld not in COSTS:
            raise ValueError(f"苦境全体のコストの定義がない: {alld}")
        run_jobs([(py("stage82_ot.py", "--cls", c, "--cfg", "full", "--pairs",
                      f"{best.cost}:prop,{alld}:prop"), f"ot_full_{c}") for c in ("c1", "c2")], w)
        jobs = []
        for cost, tier in ((best.cost, "t123"), (best.cost, "t12"), (best.cost, "t1"),
                           (alld, "t123")):
            for c in ("c1", "c2"):
                for i in (0, 1):
                    jobs.append((py("stage82_qp.py", "--cls", c, "--cfg", "full", "--group",
                                    "full", "--cost", cost, "--ot", "prop", "--qp", best.qp,
                                    "--tier", tier, "--shard", f"{i}/2"),
                                 f"qp_full_{c}_{cost}_{tier}_{i}"))
        run_jobs(jobs, w)
        run_jobs([(py("metrics_stage39.py", "--gen-dir", str(DATA / "full" / "gen_full"),
                      "--out", str(DATA / "full" / "metrics_full")), "metrics_full")], 1)
    if "final" in steps:
        run_jobs([(py("stage82_final.py"), "final")], 1)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
