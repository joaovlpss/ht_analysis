import pandas as pd
import plotly.express as px

# config
CSV_FILE_PATH = "./data/ht_data.csv"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"
LABEL_COL = "LSH label"
TIME_COL = "post_date"
HOVER_DATA_COLS = ["title", "user_name", "_id"]  # columns to show on hover


def load_and_preprocess_data(file_path, time_col, lat_col, lon_col, label_col):
    """
    Loads data from a CSV file and performs basic preprocessing.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # ensure columns exist
    required_cols = [time_col, lat_col, lon_col, label_col]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.")
            return None

    # convert post_date to datetime
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        print(
            f"Error converting '{time_col}' to datetime: {e}. Ensure format is recognizable."
        )
        # we might need to preprocess or specify
        return None

    # convert lat/lon to numeric, coercing errors to NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # handle missing or invalid geo-coordinates and labels
    df.dropna(subset=[lat_col, lon_col, label_col], inplace=True)
    if df.empty:
        print("Error: No valid data remaining after cleaning lat/lon/label.")
        return None

    # sort by date for animation
    df = df.sort_values(by=time_col)

    # create a simplified time string for animation frames (e.g., YYYY-MM-DD)
    # group by day
    # Sendingdf["animation_group"] = df[time_col].dt.strftime("%Y-%m-%d")
    # group by week
    # df['animation_group'] = df[time_col].dt.to_period('W').astype(str)
    # group by month
    # df['animation_group'] = df[time_col].dt.to_period('M').astype(str)

    # ensure label column is of string type for consistent coloring
    df[label_col] = df[label_col].astype(str)

    return df


def create_timelapse_map(
    df, lat_col, lon_col, label_col, time_col_for_animation, hover_cols
):
    """
    Creates an animated scatter plot on a map.
    """
    if df is None or df.empty:
        print("Cannot create map: DataFrame is empty or None.")
        return None

    # determine map center (plotly should do this automatically but just in case)
    map_center_lat = df[lat_col].mean()
    map_center_lon = df[lon_col].mean()

    print(f"Found {len(df)} data points to plot.")
    print(f"Unique labels for '{label_col}': {df[label_col].nunique()}")
    print(
        f"Unique animation frames for '{time_col_for_animation}': {df[time_col_for_animation].nunique()}"
    )

    fig = px.scatter_map(
        df,
        lat=lat_col,
        lon=lon_col,
        # color points by label
        color=label_col,
        # create a timelapse
        animation_frame=time_col_for_animation,
        # optionally, trace label paths
        animation_group=LABEL_COL,
        # shows up on hover
        hover_name="_id",
        # additional data on hover
        hover_data=hover_cols + [LABEL_COL],
        # max size of markers, if we ever use a "size" argument
        size_max=15,
        # we could add a 'size' argument if we ever get to have a metric like 'views'
        # size='some_numeric_column'
        zoom=3,
        center={"lat": map_center_lat, "lon": map_center_lon},
        height=700,
        title="Ad Movements Over Time by Label",
    )

    fig.update_layout(
        mapbox_style="open-street-map",  # or "carto-positron", "stamen-terrain", etc.
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title_text=f"{label_col} Groups",
    )

    # potentially improve animation slider
    # slower animation:
    # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000 # milliseconds
    # fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

    return fig


if __name__ == "__main__":
    print("Starting data loading and preprocessing...")
    data_df = load_and_preprocess_data(
        CSV_FILE_PATH, TIME_COL, LATITUDE_COL, LONGITUDE_COL, LABEL_COL
    )

    if data_df is not None and not data_df.empty:
        print("Preprocessing complete. Creating visualization...")
        timelapse_fig = create_timelapse_map(
            data_df,
            LATITUDE_COL,
            LONGITUDE_COL,
            LABEL_COL,
            "animation_group",
            HOVER_DATA_COLS,
        )
        if timelapse_fig:
            print("Visualization created. Showing map...")
            timelapse_fig.show()
            # save as an HTML file:
            timelapse_fig.write_html("ad_timelapse_map.html")
            print("Map saved to ad_timelapse_map.html")
    else:
        print("Could not proceed due to data loading/preprocessing errors.")
