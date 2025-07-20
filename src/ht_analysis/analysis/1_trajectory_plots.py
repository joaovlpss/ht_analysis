import logging
from pathlib import Path
from typing import Any, Optional

import folium
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

from folium.plugins import HeatMap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(200)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
RAW_PATH = DATA_PATH / "0_raw"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "graphs" / "infoshield_results"

TOP_10_DATA_PATH = ARTIFACTS_PATH / "top_10_persons_data.csv"
MAIN_DATASET_PATH = RAW_PATH / "ht_data.csv"

TIMESTAMP_PLOTS_PATH = ARTIFACTS_PATH / "timestamp_plots"
PROCESSED_OUTPUT_PATH = ARTIFACTS_PATH / "top_10_persons_with_geo.csv"


def load_data(data_path: Path, **kwargs: Any) -> pl.DataFrame:
    """
    Loads a dataset from a given CSV file path into a polars DataFrame.
    """
    if not data_path.is_file():
        logging.error(f"File not found: {data_path}")
        raise FileNotFoundError(f"The specified data file does not exist: {data_path}")

    try:
        logging.info(f"Loading data from {data_path}...")
        df = pl.read_csv(data_path, **kwargs)
        logging.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {data_path}: {e}")
        raise

def generate_datetime_aggregations(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Creates day, month and year aggregations from a dataframe, and outputs
    the same dataframe with these aggregations as new columns.

    Args:
        df -- A dataframe with a 'datetime' column containing pl.Datetime objects
            formatted as YYYY-MM-DD HH:mm:ss
    """
    if "datetime" not in df.columns or df["datetime"].dtype != pl.Datetime:
        logging.error("DataFrame must contain a 'datetime' column of type pl.Datetime.")
        raise ValueError("Input DataFrame is missing or has incorrect 'datetime' column type.")

    logging.info("Generating day, month, and year aggregations...")
    return df.with_columns(
        year=pl.col("datetime").dt.year(),
        month=pl.col("datetime").dt.strftime("%Y-%m"),
        day=pl.col("datetime").dt.truncate("1d"),
    )


def plot_location_over_time(df: pl.DataFrame, save_path: Path, save_name: str):
    """
    Creates and saves a line plot of location over time for a given chat name.
    """
    if df.is_empty():
        logging.warning(f"No data to plot for {save_name}. Skipping.")
        return

    pandas_df = df.to_pandas().sort_values(by="datetime", axis=0)

    pandas_df["location_count"] = pandas_df.groupby("location")["location"].transform(
        "count"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    sns.scatterplot(
        data=pandas_df,
        x="datetime",
        y="location",
        hue="location_count",
        palette="Oranges_d",
        marker="o",
        ax=ax,
    )

    ax.set_title(f"Location Trajectory for: {save_name}", fontsize=16, weight="bold")
    ax.set_xlabel("Datetime", fontsize=12)
    ax.set_ylabel("Location", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_name != "all":
        safe_filename = "".join(
            c for c in save_name if c.isalnum() or c in (" ", "_")
        ).rstrip()
        plot_filename = save_path / f"{safe_filename}_trajectory.png"
    else:
        plot_filename = save_path / "all_trajectory.png"

    try:
        plt.savefig(plot_filename, dpi=300)
        logging.info(f"Successfully saved plot to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save plot for {save_name}: {e}")
    finally:
        plt.close(fig)


def plot_location_heatmap(df: pl.DataFrame, save_path: Path, save_name: str):
    """
    Generates and saves an interactive, georeferenced heatmap on a world map.

    This function takes a DataFrame, extracts the latitude and longitude columns,
    and plots them as a heatmap layer on an interactive Folium map. The resulting
    map is saved as an HTML file.
    """
    logging.info("Generating georeferenced heatmap...")

    locations_df = df.select(["latitude", "longitude"]).drop_nulls()

    if locations_df.is_empty():
        logging.warning(
            "No valid coordinate data found. Cannot generate georeferenced heatmap."
        )
        return

    mean_lat = locations_df.get_column("latitude").mean()
    mean_lon = locations_df.get_column("longitude").mean()

    m = folium.Map(
        location=[mean_lat, mean_lon], zoom_start=4, tiles="CartoDB positron"
    )

    heat_data = locations_df.to_numpy().tolist()

    HeatMap(heat_data, radius=15, blur=12).add_to(m)

    heatmap_filename = save_path / f"{save_name}_georeferenced_heatmap.html"
    try:
        m.save(str(heatmap_filename))
        logging.info(f"Successfully saved interactive heatmap to {heatmap_filename}")
    except Exception as e:
        logging.error(f"Failed to save georeferenced heatmap: {e}")


def main():
    top_10_df = load_data(TOP_10_DATA_PATH)

    geo_data_df = load_data(
        MAIN_DATASET_PATH, columns=["ad_id", "latitude", "longitude"]
    )

    logging.info("Joining the top 10 persons data with geo-coordinates...")
    combined_df = top_10_df.join(geo_data_df, on="ad_id", how="left")
    logging.info("Join successful.")

    logging.info("Processing the 'trajectory' column...")
    processed_df = (
        combined_df.with_columns(_trajectory_split=pl.col("trajectory").str.split("_"))
        .with_columns(
            location=pl.col("_trajectory_split").list.get(0),
            datetime_str=pl.col("_trajectory_split").list.get(1),
        )
        .with_columns(
            datetime=pl.col("datetime_str").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
        .drop(
            "trajectory",
            "_trajectory_split",
            "datetime_str"
        )
    )

    logging.info("Successfully split 'trajectory' into 'location' and 'datetime'.")

    final_df = processed_df.select(
        "data_chat_name",
        "ad_id",
        "phone",
        "email",
        "url",
        "location",
        "datetime",
        "latitude",
        "longitude",
        "body",
    )

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving processed data to {PROCESSED_OUTPUT_PATH}...")
    final_df.write_csv(PROCESSED_OUTPUT_PATH)

    logging.info("Starting plot generation...")

    plot_location_over_time(final_df, TIMESTAMP_PLOTS_PATH, "all")
    plot_location_heatmap(final_df, TIMESTAMP_PLOTS_PATH, "all")

    unique_chat_names = final_df.get_column("data_chat_name").unique().to_list()

    for name in unique_chat_names:
        logging.info(f"Generating plot for '{name}'...")
        person_specific_df = final_df.filter(pl.col("data_chat_name") == name)

        plot_location_over_time(person_specific_df, TIMESTAMP_PLOTS_PATH, name)
        plot_location_heatmap(person_specific_df, TIMESTAMP_PLOTS_PATH, name)

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()
