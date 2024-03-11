import tempfile
from pathlib import Path

from hyperbolic_vae.datasets.jerby_arnon import (
    ANNOTATIONS_CSV_GZ_URL,
    JerbyArnonCSVDataModule,
    _download_and_extract_csv_gz,
)


def test_download_and_extract_csv_gz():
    with tempfile.TemporaryDirectory() as tempdir:
        save_path = Path(tempdir) / "annotations.csv"
        _download_and_extract_csv_gz(ANNOTATIONS_CSV_GZ_URL, save_path)
        assert save_path.exists()
        assert save_path.is_file()
        assert save_path.stat().st_size > 0


def test_data_module():
    with tempfile.TemporaryDirectory() as tempdir:
        data_module = JerbyArnonCSVDataModule(data_path=Path(tempdir), batch_size=2, num_workers=0)
        assert data_module.data_path.exists()
        assert data_module.batch_size == 2
        assert data_module.num_workers == 0
        data_module.prepare_data()
        data_module.setup(stage="fit")
