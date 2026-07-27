import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep

set_config(transform_output="pandas")

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Uber Demand — NYC",
    page_icon="🚕",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders — models/data only load once per session, not on every rerun
# ---------------------------------------------------------------------------
root_path = Path(__file__).parent


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    scaler = joblib.load(root_path / "models/scaler.joblib")
    encoder = joblib.load(root_path / "models/encoder.joblib")
    model = joblib.load(root_path / "models/model.joblib")
    kmeans = joblib.load(root_path / "models/mb_kmeans.joblib")
    return scaler, encoder, model, kmeans


@st.cache_data(show_spinner="Loading data...")
def load_data():
    df_plot = pd.read_csv(root_path / "data/external/plot_data.csv")
    df = pd.read_csv(
        root_path / "data/processed/test.csv", parse_dates=["tpep_pickup_datetime"]
    ).set_index("tpep_pickup_datetime")
    return df_plot, df


scaler, encoder, model, kmeans = load_models()
df_plot, df = load_data()

pipe = Pipeline([("encoder", encoder), ("reg", model)])

COLORS = [
    "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
    "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
    "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
    "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
    "#FF00FF", "#FF1493", "#C71585", "#FF6347", "#FFA07A",
    "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA", "#DAA520",
]
region_colors = {r: COLORS[i] for i, r in enumerate(df_plot["region"].unique().tolist())}
df_plot["color"] = df_plot["region"].map(region_colors)


def assign_region(lat: float, long: float) -> int:
    """Scale a raw lat/long pair and predict which k-means region it falls into."""
    raw = pd.DataFrame({"pickup_latitude": [lat], "pickup_longitude": [long]})
    raw = raw[scaler.feature_names_in_]
    scaled = scaler.transform(raw)
    
    prediction = np.asarray(kmeans.predict(np.asarray(scaled)))
    return int(prediction.ravel()[0])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚕 Uber Demand in New York City")
st.caption("Interactive demand forecasting across NYC pickup regions")

# ---------------------------------------------------------------------------
# Sidebar — all inputs live here so the main area is dedicated to results
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")

    map_type = st.radio(
        "Map scope",
        options=["Complete NYC Map", "Only for Neighborhood Regions"],
        index=1,
    )

    st.divider()
    st.subheader("📅 When")
    date = st.date_input(
        "Date",
        value=dt.date(2016, 3, 1),
        min_value=dt.date(2016, 3, 1),
        max_value=dt.date(2016, 3, 31),
    )
    time = st.time_input("Time", value=dt.time(12, 0))

    st.divider()
    st.subheader("📍 Where")

    location_mode = st.radio(
        "Location source",
        options=["Random sample", "Enter coordinates"],
        help="Enter your own latitude/longitude, or sample a random NYC pickup point.",
    )

    if location_mode == "Random sample":
        # keep the sampled location stable across reruns; only resample on click
        if "sample_loc" not in st.session_state:
            st.session_state.sample_loc = df_plot.sample(1).reset_index(drop=True)

        if st.button("🔄 Get New Random Location", width="stretch"):
            st.session_state.sample_loc = df_plot.sample(1).reset_index(drop=True)

        sample_loc = st.session_state.sample_loc
        lat = sample_loc["pickup_latitude"].item()
        long = sample_loc["pickup_longitude"].item()
        region = sample_loc["region"].item()

    else:
        col_lat, col_long = st.columns(2)
        lat = col_lat.number_input("Latitude", value=40.7128, format="%.6f")
        long = col_long.number_input("Longitude", value=-74.0060, format="%.6f")

        # basic sanity bounds so a wildly out-of-range value doesn't silently
        # get assigned to some meaningless region
        if not (40.4 <= lat <= 41.0 and -74.3 <= long <= -73.6):
            st.warning("These coordinates look outside the NYC dataset's coverage area. "
                       "Results may not be meaningful.")

        region = assign_region(lat, long)
        sample_loc = pd.DataFrame({
            "pickup_latitude": [lat],
            "pickup_longitude": [long],
            "region": [region],
        })

    st.metric("Region ID", region)
    col_a, col_b = st.columns(2)
    col_a.metric("Latitude", f"{lat:.4f}")
    col_b.metric("Longitude", f"{long:.4f}")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if date and time:

    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime(
        year=date.year, month=date.month, day=date.day,
        hour=time.hour, minute=time.minute,
    ) + delta
    index = pd.Timestamp(f"{date} {next_interval.time()}")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Forecast window", next_interval.time().strftime("%H:%M"))
    info_col2.metric("Timestamp used", index.strftime("%Y-%m-%d %H:%M"))
    info_col3.metric("Current region", region)

    tab_map, tab_chart, tab_data = st.tabs(["🗺️ Map", "📊 Demand Chart", "📋 Raw Predictions"])

    # -- compute predictions once, reused across all tabs -------------------
    try:
        if map_type == "Complete NYC Map":
            plot_df = df_plot
            input_data = df.loc[index, :].sort_values("region")
            
            region_ids = input_data["region"].tolist()
        else:
            scaled_cord = scaler.transform(sample_loc[scaler.feature_names_in_])
            distances = np.asarray(kmeans.transform(np.asarray(scaled_cord))).ravel().tolist()
            distances = list(enumerate(distances))
            sorted_distances = sorted(distances, key=lambda x: x[1])[:9]
            region_ids = sorted([r[0] for r in sorted_distances])
            plot_df = df_plot[df_plot["region"].isin(region_ids)]
            input_data = df.loc[index, :]
            input_data = input_data.loc[input_data["region"].isin(region_ids), :].sort_values("region")

        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))
        results_df = pd.DataFrame({
            "region": region_ids,
            "predicted_demand": predictions.astype(int),
            "color": [region_colors[r] for r in region_ids],
        })
        results_df["is_current_region"] = results_df["region"] == region

        prediction_success = True
    except KeyError:
        st.warning(
            f"No data available for {index}. Try a different date/time — "
            "this dataset only covers March 2016."
        )
        prediction_success = False

    if prediction_success:
        with tab_map:
            progress_bar = st.progress(0, text="Rendering map...")
            for pct in range(100):
                sleep(0.01)
                progress_bar.progress(pct + 1, text="Rendering map...")
            progress_bar.empty()

            st.map(data=plot_df, latitude="pickup_latitude",
                   longitude="pickup_longitude", size=0.01, color="color")

            with st.expander("🗺️ Map Legend", expanded=False):
                legend_cols = st.columns(3)
                for i, row in results_df.iterrows():
                    with legend_cols[i % 3]:
                        label = f"Region {row['region']}"
                        if row["is_current_region"]:
                            label += " ⭐ (You)"
                        st.markdown(
                            f'<div style="display:flex;align-items:center;">'
                            f'<div style="background-color:{row["color"]};width:16px;'
                            f'height:16px;margin-right:8px;border-radius:3px;"></div>'
                            f'{label} — <b>{row["predicted_demand"]}</b> rides</div>',
                            unsafe_allow_html=True,
                        )

        with tab_chart:
            st.subheader("Predicted Demand by Region")

            chart_df = results_df.copy()
            chart_df = chart_df.sort_values("region")

            
            category_order = chart_df["region"].astype(str).tolist()

            fig = px.bar(
                chart_df,
                x=chart_df["region"].astype(str),
                y="predicted_demand",
                color="is_current_region",
                color_discrete_map={True: "#FFD700", False: "#1E90FF"},
                labels={"x": "Region", "predicted_demand": "Predicted Rides", "is_current_region": "Your region"},
                text="predicted_demand",
            )
            fig.update_xaxes(categoryorder="array", categoryarray=category_order)
            fig.update_layout(showlegend=True, xaxis_title="Region ID", yaxis_title="Predicted Demand")
            st.plotly_chart(fig, width="stretch")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total predicted demand", int(results_df["predicted_demand"].sum()))
            m2.metric("Busiest region", int(results_df.loc[results_df["predicted_demand"].idxmax(), "region"]))
            m3.metric(
                "Your region's demand",
                int(results_df.loc[results_df["is_current_region"], "predicted_demand"].iloc[0]),
            )

        with tab_data:
            st.subheader("Prediction Table")
            st.dataframe(
                results_df[["region", "predicted_demand", "is_current_region"]]
                .rename(columns={"is_current_region": "your_region"})
                .sort_values("predicted_demand", ascending=False),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name=f"demand_predictions_{index.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )