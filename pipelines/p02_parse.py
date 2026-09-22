"""層2(parsing): 取得済み XBRL を生テーブルへ変換するエントリポイント。

ロジックは src/fsgen/parsing/ に置き、ここでは設定の読み込みと
呼び出しのみを行う。
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    raise NotImplementedError("parsing layer is not implemented yet (stage 1 skeleton)")


if __name__ == "__main__":
    main()
