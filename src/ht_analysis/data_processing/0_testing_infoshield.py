import logging
from pathlib import Path
from typing import Optional, Any

import polars as pl

from src.InfoShield import infoshieldcoarse, infoshieldfine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
RAW_PATH = DATA_PATH / "0_raw"
BRONZE_PATH = DATA_PATH / "1_bronze"
TABLE_PATH = BRONZE_PATH / "ids_and_text_columns.csv"
RESULT_PATH = BRONZE_PATH / "ids_and_text_columns_LSH_labels.csv"


def process_data(data_path: Path, **kwargs: Any) -> Optional[bool]:
    """Import data from the given path directly into InfoShield"""
    if not data_path.is_file():
        logging.error(f"Data file not found at: {data_path}")
        return None

    try:
        id_column = kwargs.get("id_column", "ad_id")
        text_column = kwargs.get("text_column", "body")
        coarse = infoshieldcoarse.InfoShieldCoarse(str(data_path))
        coarse.clustering()
        infoshieldfine.run_infoshieldfine(
            coarse.final_data_filename, id_column, text_column
        )
        return True
    except Exception as e:
        logging.error(f"An unexpected error ocurred: {e}")
        return None


def filter_data(data_path: Path, write_path: Path, **kwargs: Any) -> Optional[bool]:
    """Filter the original raw dataset's id and text columns, then put them in the bronze layer."""
    logging.info(f"Attempting to load dataset: {data_path}")

    if not data_path.is_file():
        logging.error(f"File not found: {data_path}")
        return None

    try:
        df = (
            pl.scan_csv(data_path, **kwargs).select(
                pl.col("ad_id"), pl.col("post_date"), pl.col("body")
            )
        ).collect()
        write_file = write_path / "ids_and_text_columns.csv"
        df.write_csv(write_file)
        logging.info(f"Successfully saved file to {write_file}. Shape = {df.shape}")
    except FileNotFoundError:
        logging.error(f"File not found during read attempt: {data_path}")
        return None

    except UnicodeDecodeError as e:
        logging.error(
            f"Encoding error reading {data_path} with encoding '{kwargs.get('encoding', 'default')}': {e}. "
            f"Verify encoding."
        )
        return None

    except Exception as e:
        logging.error(f"An unexpected error occurred loading {data_path}: {e}")
        return None


def main():
    # filter_data(data_path=RAW_PATH / "ht_data.csv", write_path=BRONZE_PATH)
    # process_data(data_path=TABLE_PATH, id_column="ad_id", text_column="body")

    logging.info(f"Describing dataframe from {str(RESULT_PATH).split('/')[-1]}")
    try:
        df = pl.read_csv(
            RESULT_PATH,
            schema={
                "ad_id": pl.Float64,
                "post_date": pl.String,
                "body": pl.String,
                "LSH label": pl.String,
            },
        ).select(pl.col("LSH label"))

        unique_count = df.select(pl.col("LSH label").n_unique()).item()
        value_counts_df = (
            df.select(pl.col("LSH label").value_counts())
            .unnest("LSH label")
            .sort(by="LSH label")
        )
        value_counts_df.write_csv(BRONZE_PATH / "unique_LSH_counts.csv")

        logging.info(f"Total labels: {unique_count}")
        logging.info(
            f"Count of appearances of each unique value in 'LSH label': {value_counts_df}"
        )

    except FileNotFoundError:
        logging.error(f"File not found during read attempt: {RESULT_PATH}")
        return None

    except Exception as e:
        logging.error(f"Unexpected error ocurred: {e}")
        return None


if __name__ == "__main__":
    main()
