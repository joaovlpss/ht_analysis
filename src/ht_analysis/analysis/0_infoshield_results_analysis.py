import logging
from pathlib import Path
from typing import Optional, Any
import random

import polars as pl
import networkx as nx
from pyvis.network import Network


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
RAW_PATH = DATA_PATH / "0_raw"
BRONZE_PATH = DATA_PATH / "1_bronze"
LSH_LABEL_PATH = BRONZE_PATH / "ids_and_text_columns_LSH_labels.csv"
DATASET_PATH = RAW_PATH / "ht_data.csv"


def load_data(data_path: Path, **kwargs: Any) -> pl.DataFrame:
    """Load a dataset from a given path."""

    assert data_path.is_file()

    try:
        return pl.read_csv(data_path, **kwargs)
    except Exception as e:
        logging.error(f"An unexpected error ocurred: {e}")
        raise e


def add_lsh_labels(dataset: pl.DataFrame, lsh_labels: pl.DataFrame) -> pl.DataFrame:
    """Adds the 'LSH Label' column to the main dataset, substituting its
    'LSH Label' column. Returns the joined dataset."""

    # first we check for column existence
    assert not dataset.is_empty()
    assert not lsh_labels.is_empty()
    try:
        dataset.get_column("ad_id")
        lsh_labels.get_column("ad_id")
        lsh_labels.get_column("LSH label")
    except pl.exceptions.ColumnNotFoundError as e:
        logging.error(f"An unexpected error ocurred: {e}")
        raise e

    # then we just drop LSH Label in the main dataset and add the one from our results
    lsh_labels = lsh_labels.drop(["body", "post_date"])
    return dataset.join(lsh_labels, on="ad_id", how="left")


def main():
    logging.info("Trying to load dataset and LSH Labels...")
    dataset = load_data(DATASET_PATH).drop("LSH label")
    lsh_labels = load_data(LSH_LABEL_PATH)

    logging.info("Successfully loaded dataset and LSH labels! Performing join...")
    dataset_lsh = add_lsh_labels(dataset=dataset, lsh_labels=lsh_labels)

    logging.info("Mounting graph from new dataset...")
    logging.info("Separating columns of interest.")
    dataset_lsh = (
        dataset_lsh.with_columns(
            pl.concat_str(["location_detail", "post_date"], separator="_").alias(
                "trajectory"
            )
        )
        .select(
            pl.col("ad_id"),
            pl.col("data_chat_name").alias("person_id"),
            pl.col("phone"),
            pl.col("email"),
            pl.col("url"),
            pl.col("main_image_url"),
            pl.col("LSH label"),
            pl.col("trajectory"),
        )
        .drop_nulls("LSH label")
    )

    logging.info(
        "Separating rows with LSH labels in the top 10% percentile (of appearances)."
    )
    # here, we count the number of appearances of unique labels,
    # separate the top 10% percentile of labels by appearance,
    # then filter our dataset with only rows which contain such labels in the "LSH label" column.
    counts_df = dataset_lsh["LSH label"].value_counts(sort=True)
    print(counts_df.head(10))

    top_10pct_count = int(0.1 * counts_df.height)
    top_labels = counts_df.head(top_10pct_count)["LSH label"]
    top_10pct_df = dataset_lsh.filter(pl.col("LSH label").is_in(top_labels))

    logging.info("Generating networkx graph.")
    graph = nx.Graph()

    for row in top_10pct_df.iter_rows(named=True):
        person_id = row["person_id"]

        graph.add_node(person_id, type="person")

        for col_name, value in row.items():
            if col_name != "person_id" and value is not None:
                # add the other column's value as a node
                # prefixing with column name to ensure uniqueness if values overlap across columns
                node_id = f"{col_name}_{value}"
                graph.add_node(node_id, type=col_name)
                graph.add_edge(person_id, node_id)


if __name__ == "__main__":
    main()
