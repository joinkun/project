"""層1(acquisition): EDINET から書類を取得するエントリポイント。

ロジックは src/fsgen/acquisition/ に置き、ここでは設定の読み込みと
呼び出しのみを行う。

ドライラン(書類一覧APIのみ叩いて document_list.csv を書いて終了):
    uv run python pipelines/p01_fetch.py data=fy2024 dry_run=true

実取得(ZIPを data/raw/documents/ に保存):
    uv run python pipelines/p01_fetch.py data=fy2024

出力先は ``cfg.data.output_suffix`` で提出期間ごとに分ける
(例: ``_2023_2024`` なら data/raw/documents_2023_2024/ と
index/document_list_2023_2024.csv、index/fetch_log_2023_2024.csv)。
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from fsgen.acquisition.document_list import (
    fetch_document_list,
    fetch_documents,
    write_document_list_csv,
)
from fsgen.acquisition.edinet_client import EdinetClient

log = logging.getLogger(__name__)


def _require_api_key(api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"environment variable {api_key_env} is not set "
            f"(copy .env.example to .env and fill it in)"
        )
    return api_key


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    # API キーはログに出さない。resolved config も key を含まないことを確認済み。
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    api_key = _require_api_key(cfg.edinet.api_key_env)
    date_from = dt.date.fromisoformat(cfg.data.date_from)
    date_to = dt.date.fromisoformat(cfg.data.date_to)
    doc_type_codes = tuple(cfg.data.doc_type_codes)
    ordinance_codes = tuple(cfg.data.ordinance_codes)

    raw_dir = Path(cfg.output.raw_dir)
    index_dir = raw_dir / "index"
    suffix = str(cfg.data.get("output_suffix", ""))

    with EdinetClient(
        api_key=api_key,
        base_url=cfg.edinet.base_url,
        min_interval_seconds=cfg.edinet.min_interval_seconds,
        max_retries=cfg.edinet.max_retries,
        backoff_base_seconds=cfg.edinet.backoff_base_seconds,
    ) as client:
        summary = fetch_document_list(
            client,
            date_from,
            date_to,
            doc_type_codes,
            ordinance_codes,
            require_xbrl=cfg.data.require_xbrl,
        )
        write_document_list_csv(summary, index_dir / f"document_list{suffix}.csv")

        log.info(
            "document list: %d days queried (%d failed), %d raw results, "
            "%d unique documents, %d unique edinetCode, %d edinetCode with duplicates",
            summary.days_queried,
            len(summary.failed_dates),
            summary.raw_result_count,
            summary.unique_document_count,
            summary.unique_edinet_code_count,
            len(summary.duplicate_edinet_codes),
        )
        if summary.failed_dates:
            log.warning("failed to list documents for dates: %s", summary.failed_dates)

        if cfg.dry_run:
            log.info("dry_run=true: stopping after document_list.csv")
            return

        fetch_documents(
            client,
            summary.records,
            raw_dir,
            index_dir,
            estimated_bytes_per_doc=cfg.fetch.estimated_bytes_per_doc,
            assume_yes=cfg.fetch.assume_yes,
            suffix=suffix,
        )


if __name__ == "__main__":
    main()
