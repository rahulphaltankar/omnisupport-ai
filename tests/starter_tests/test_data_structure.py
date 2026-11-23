from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
# Raw primary data and the fine-tune subset are intentionally NOT committed to the public repository (see .gitignore / data/README.md).
# These checks therefore run when the supplied data are present locally and are skipped in CI.
needs_raw = pytest.mark.skipif(not list(RAW.glob('support_records_part_*.parquet')), reason='raw primary data not present (not committed to the repository)')


@needs_raw
def test_primary_dataset_is_parquet_and_has_ten_shards():
    shards = sorted(RAW.glob('support_records_part_*.parquet'))
    assert len(shards) == 10
    assert not list(RAW.glob('support_records_part_*.csv.gz'))


@needs_raw
def test_required_supporting_assets_exist():
    assert (ROOT / 'data' / 'subsets' / 'transformer_finetune_10000.parquet').exists()


def test_committed_supporting_assets_exist():
    assert (ROOT / 'data' / 'agent_store' / 'orders.csv').exists()
    assert (ROOT / '.github' / 'workflows' / 'tests.yml').exists()
    assert (ROOT / 'monitoring' / 'monitoring_summary_template.csv').exists()
