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
    "ad_id": "#FF0000",              # Red
    "person_id": "#0000FF",           # Blue
    "phone": "#00C000",              # Bright Green
    "email": "#FFA500",              # Bright Orange
    "url": "#000000",              # Black
    "main_image_url": "#00CED1",      # Dark Turquoise / Cyan
    "LSH label": "#FF00FF",           # Magenta / Fuchsia
    "trajectory": "#A52A2A",         # Brown
    "person": "#0000FF",              # Blue (aliased with person_id)
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


def plot_and_save_graph(graph: nx.Graph, file_path: Path, title: str):
    """Generates and saves an interactive plot of a graph."""
    if graph.number_of_nodes() == 0:
        logging.warning(f"Skipping plot for empty graph: {title}")
        return

    # Create output directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(graph, seed=42)

    # Edge trace
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Node trace
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = graph.nodes[node].get('type', 'unknown')
        node_text.append(f"{node}<br>Type: {node_type}")
        node_color.append(STATIC_COLOR_MAP.get(node_type, "#808080")) # Default to grey

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    logging.info(f"Saving graph to {file_path}")
    fig.write_html(file_path)


def main():
    logging.info("Trying to load dataset and LSH Labels...")
    dataset = load_data(DATASET_PATH, infer_schema_length=10000).drop("LSH label")

    logging.info("Successfully loaded dataset and LSH labels! Performing join...")

    logging.info("Preparing data for graph construction...")
    graph_df = (
        dataset.with_columns(
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
            pl.col("trajectory"),
        )
    )

    logging.info("Mounting graph from the dataset...")
    graph = nx.Graph()
    for row in graph_df.iter_rows(named=True):
        person_id = row.get("person_id")
        if person_id is None:
            continue

        # Add person node, aliasing its type to 'person' for consistency
        graph.add_node(person_id, type="person")

        for col_name, value in row.items():
            if col_name != "person_id" and value is not None:
                # Prefix with column name to ensure uniqueness of nodes across types
                node_id = f"{col_name}_{value}"
                graph.add_node(node_id, type=col_name)
                graph.add_edge(person_id, node_id)

    logging.info(f"Graph mounted successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    logging.info("Calculating the main core of the graph...")
    main_core = nx.k_core(graph)
    logging.info(f"Main core contains {main_core.number_of_nodes()} nodes.")

    logging.info("Identifying top 10 most connected 'person' nodes in the core...")
    core_degrees = dict(main_core.degree())

    person_nodes = [
        n for n, attr in main_core.nodes(data=True) if attr.get("type") == "person"
    ]

    top_persons = sorted(
        person_nodes,
        key=lambda n: core_degrees.get(n, 0),
        reverse=True
    )[:10]

    logging.info(f"Top 10 persons identified: {top_persons}")

    logging.info("Generating and saving egonet plots for top persons...")
    for person_node in top_persons:
        person_output_path = OUTPUT_PATH / str(person_node)

        for radius in [1, 2]:
            logging.info(f"Generating egonet for '{person_node}' with radius {radius}...")
            ego_graph = nx.ego_graph(graph, person_node, radius=radius)

            file_name = f"egonet_radius_{radius}.html"
            output_file_path = person_output_path / file_name

            plot_title = f"EgoNet for {person_node} (Radius: {radius})"
            plot_and_save_graph(ego_graph, output_file_path, title=plot_title)

    logging.info("Creating and saving dataset for top 10 persons...")

    # Add the 'trajectory' column to the original dataset. This is done here to
    # avoid altering the existing workflow above.
    dataset_with_trajectory = dataset.with_columns(
        pl.concat_str(["location_detail", "post_date"], separator="_").alias(
            "trajectory"
        )
    )

    # Filter the dataset for rows where 'data_chat_name' is in the top_persons list
    # and select the specified columns.
    top_10_table = dataset_with_trajectory.filter(
        pl.col("data_chat_name").is_in(top_persons)
    ).select(
        "data_chat_name",
        "ad_id",
        "phone",
        "email",
        "url",
        "trajectory",
        "body",
    ).sort(by="data_chat_name")

    output_csv_path = OUTPUT_PATH / "top_10_persons_data.csv"
    logging.info(f"Saving top 10 persons data to {output_csv_path}")
    top_10_table.write_csv(output_csv_path)

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()
