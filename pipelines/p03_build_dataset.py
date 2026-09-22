"""層3(accounting) + datasets: 制約付きデータセットを構築するエントリポイント。

`accounts` 設定グループで canonical 勘定科目リストを切り替える
(core: 出現率90%以上 / extended: 出現率50%以上。conf/accounts/ 参照):
    uv run python pipelines/p03_build_dataset.py accounts=extended

ロジックは src/fsgen/accounting/ と src/fsgen/datasets/ に置き、
ここでは設定の読み込みと呼び出しのみを行う。
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    raise NotImplementedError("accounting/datasets layers are not implemented yet (stage 1 skeleton)")


if __name__ == "__main__":
    main()
