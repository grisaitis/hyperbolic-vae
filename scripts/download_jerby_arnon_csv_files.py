import logging

import hyperbolic_vae
from hyperbolic_vae.datasets.jerby_arnon import (
    ANNOTATIONS_CSV_GZ_URL,
    ANNOTATIONS_CSV_PATH,
    JERBY_ARNON_DATA_PATH,
    TPM_CSV_GZ_URL,
    TPM_CSV_PATH,
    _download_and_extract_csv_gz,
)

if __name__ == "__main__":
    logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
    logging.getLogger().setLevel("INFO")
    hyperbolic_vae.util.configure_handler_for_script()

    JERBY_ARNON_DATA_PATH.mkdir(exist_ok=True)
    _download_and_extract_csv_gz(ANNOTATIONS_CSV_GZ_URL, ANNOTATIONS_CSV_PATH)
    _download_and_extract_csv_gz(TPM_CSV_GZ_URL, TPM_CSV_PATH)
