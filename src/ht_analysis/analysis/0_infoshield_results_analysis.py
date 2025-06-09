import logging
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / "data"
RAW_PATH = DATA_PATH / "0_raw"
BRONZE_PATH = DATA_PATH / "1_bronze"
LSH_LABEL_PATH = BRONZE_PATH / "ids_and_text_columns_LSH_labels.csv"
DATASET_PATH = RAW_PATH / "ht_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "graphs" / "infoshield_results"


# Statically define the color map for node types
STATIC_COLOR_MAP = {
    "ad_id": "#FF6347",  # Tomato
    "person_id": "#4682B4",  # SteelBlue
    "phone": "#32CD32",  # LimeGreen
    "email": "#FFD700",  # Gold
    "url": "#6A5ACD",  # SlateBlue
    "main_image_url": "#8A2BE2",  # BlueViolet
    "LSH label": "#D2691E",  # Chocolate
    "trajectory": "#00CED1",  # DarkTurquoise
    "person": "#4682B4", # person_id is aliased to person, so we should use the same color
}


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

    print(f"Shape antes = {dataset_lsh.shape}")

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

    print(f"Shape depois = {dataset_lsh.shape}")

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

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created at: {OUTPUT_PATH}")

    unique_lsh_labels = top_10pct_df["LSH label"].unique().to_list()
    logging.info(
        f"Found {len(unique_lsh_labels)} unique 'LSH label' categories to process."
    )

    for lsh_label in unique_lsh_labels:
        logging.info(f"Processing graph for LSH label: {lsh_label}")

        label_df = top_10pct_df.filter(pl.col("LSH label") == lsh_label)

        graph = nx.Graph()
        for row in label_df.iter_rows(named=True):
            person_id = row["person_id"]
            if person_id is None:
                continue

            graph.add_node(person_id, type="person")

            for col_name, value in row.items():
                if col_name != "person_id" and value is not None:
                    # Prefix with column name to ensure uniqueness
                    node_id = f"{col_name}_{value}"
                    graph.add_node(node_id, type=col_name)
                    graph.add_edge(person_id, node_id)

        logging.info(
            f"Graph for '{lsh_label}' created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )

        if graph.number_of_nodes() == 0:
            logging.warning(f"Graph for LSH label '{lsh_label}' is empty. Skipping visualization.")
            continue

        logging.info("Generating interactive graph plot with Plotly...")

        pos = nx.spring_layout(graph, seed=42)

        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])


        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # create node traces
        node_x, node_y, node_text, node_colors = [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_type = graph.nodes[node].get("type", "unknown")
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"ID: {node}<br>Type: {node_type}")
            node_colors.append(STATIC_COLOR_MAP.get(node_type, "#808080")) # default to grey

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(
                showscale=False,
                size=10,
                color=node_colors,
                line_width=2,
            ),
        )
        node_trace.text = node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"Network Graph for LSH Label: {lsh_label}",
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        output_filename = OUTPUT_PATH / f"graph_for_LSH_label_{lsh_label}.html"
        try:
            fig.write_html(str(output_filename))
            logging.info(f"Interactive graph saved to: {output_filename}")
        except Exception as e:
            logging.error(f"Error generating or saving Plotly graph: {e}")
            raise e

if __name__ == "__main__":
    main()
