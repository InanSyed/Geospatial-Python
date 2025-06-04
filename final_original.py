"""Geospatial analysis of cities near Holocene volcanoes.

This script demonstrates a full workflow for analysing cities in close
proximity to active (Holocene) volcanoes.  The code loads volcano, city and
country datasets, performs spatial joins and distance calculations and finally
visualises the results.  It produces both a static map with ``matplotlib`` and
an interactive ``folium`` map.  All file paths are resolved relative to the
repository so the script can be executed from any directory.
"""

from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import mapping

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
# Root directory where all datasets live relative to this repository
BASE_DIR = Path("data")

# Path to the shapefile containing point locations of Holocene volcanoes
VOLCANO_SHP = (
    BASE_DIR
    / "Global_2013_HoloceneVolcanoes_SmithsonianVOTW"
    / "Smithsonian_VOTW_Holocene_VolcanoesPoint.shp"
)
# Path to the world populated places dataset from Natural Earth
CITIES_SHP = BASE_DIR / "ne_10m_populated_places" / "ne_10m_populated_places.shp"
# Path to the country boundaries dataset from Natural Earth
COUNTRIES_SHP = BASE_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"

# Colours used for different hazard tiers when plotting
COLOR = {
    "High": "#d73027",  # within 10 km
    "Med-High": "#fc8d59",  # 10–30 km
    "Medium": "#fee08b",  # 30–50 km
    "Low": "#d9ef8b",  # 50–100 km
}

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def load_layer(path: Path, name: str) -> gpd.GeoDataFrame:
    """Load a vector layer from *path* with some basic sanity checks."""

    # Make sure the file exists before attempting to read it.  This provides a
    # friendlier error message than letting ``geopandas`` fail later on.
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path}")

    # Use GeoPandas to read the vector data (shapefile, GeoJSON, ...).
    gdf = gpd.read_file(path)

    # Abort if the dataset unexpectedly contains no records.
    if gdf.empty:
        raise ValueError(f"{name} loaded but is empty")

    # Drop records that do not have a valid geometry to avoid issues later.
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    return gdf


def inspect(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Print a short overview of a GeoDataFrame."""

    # Display some high-level information about the dataset.  This is useful
    # when running the script interactively to understand what each layer
    # contains without printing the entire table.
    print(f"\n{name}:")
    print(f"  • Records : {len(gdf):,}")
    print(f"  • CRS     : {gdf.crs}")
    print(f"  • Columns : {list(gdf.columns)[:10]} ...")
    print("  • Sample:")
    print(gdf.head(3))


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------


def compute_hazard_rings(volcanoes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return polygons representing hazard distance bands around volcanoes."""

    # Distance bands we want to create.  Each tuple is (inner_km, outer_km,
    # label).  The first one starts at 0 km so it represents the area closest
    # to the vent.
    tiers = [
        (0, 10, "High"),
        (10, 30, "Med-High"),
        (30, 50, "Medium"),
        (50, 100, "Low"),
    ]

    rings = []
    for low, high, name in tiers:
        # Buffer the points by the outer distance to create a disc
        outer = volcanoes.geometry.buffer(high * 1000)
        ring = gpd.GeoSeries(outer).unary_union

        # If there is an inner radius, subtract the inner disc to form a ring
        if low > 0:
            inner = volcanoes.geometry.buffer(low * 1000)
            ring = ring.difference(gpd.GeoSeries(inner).unary_union)

        rings.append({"hazard": name, "geometry": ring})

    # Combine rings into a GeoDataFrame preserving the CRS of the input
    return gpd.GeoDataFrame(rings, crs=volcanoes.crs)


def classify_cities(
    cities: gpd.GeoDataFrame, volcanoes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Assign a hazard tier to each city based on its nearest volcano."""

    # Use a spatial join to find the closest volcano to each city and measure
    # the distance between them.  ``sjoin_nearest`` returns the index of the
    # nearest volcano along with the computed distance in the units of the
    # layer's CRS (metres for Web Mercator).
    nearest = gpd.sjoin_nearest(
        cities, volcanoes[["geometry"]], how="left", distance_col="dist"
    ).drop(columns="index_right")

    cities = nearest
    # Convert the distance column from metres to kilometres for readability
    cities["dist_km"] = cities.pop("dist") / 1000.0

    def tier(d: float) -> str:
        """Return the hazard classification for a given distance in km."""

        if d <= 10:
            return "High"
        if d <= 30:
            return "Med-High"
        if d <= 50:
            return "Medium"
        return "Low"

    # Apply the helper to categorise every city
    cities["hazard"] = cities["dist_km"].apply(tier)
    return cities


def plot_static(
    world: gpd.GeoDataFrame,
    volcanos: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
) -> None:
    """Render a static world map using the Robinson projection."""

    # The Robinson projection gives a pleasing view of the entire world while
    # preserving relative shapes fairly well.
    rob = "ESRI:54030"
    world_robin = world.to_crs(rob)
    volc_robin = volcanos.to_crs(rob)
    cities_robin = cities.to_crs(rob)

    # Create the plot and draw the world map as a light grey background
    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)
    world_robin.plot(ax=ax, color="#f7f7f7", edgecolor="lightgray", linewidth=0.4)
    # Plot volcanoes as red triangles
    volc_robin.plot(
        ax=ax,
        marker="^",
        markersize=10,
        color="red",
        alpha=0.7,
        label="Holocene volcano",
    )
    # Plot at-risk cities as blue circles
    cities_robin.plot(
        ax=ax,
        marker="o",
        markersize=20,
        color="dodgerblue",
        alpha=0.6,
        label="City (>100k) ≤100km",
    )
    # Finish off the plot with a title and legend then display it
    ax.set_title(
        "Major Cities within 100 km of an Active (Holocene) Volcano",
        fontsize=15,
    )
    ax.legend(frameon=False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def create_interactive_map(
    volcanoes: gpd.GeoDataFrame,
    hazard_rings: gpd.GeoDataFrame,
    cities: gpd.GeoDataFrame,
    tier_tbl: pd.DataFrame,
) -> folium.Map:
    """Return a Folium map visualising volcanoes and at-risk cities."""

    # Convert layers to geographic coordinates for use with Folium/Leaflet
    rings_wgs = hazard_rings.to_crs(4326)
    cities_wgs = cities.to_crs(4326)

    # Create the base map centred on the equator using a neutral basemap
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")

    # Add volcano locations as a separate layer so users can toggle them
    volc_layer = folium.FeatureGroup(name="Volcanoes")
    for _, r in volcanoes.iterrows():
        folium.CircleMarker(
            [r.Latitude, r.Longitude],
            radius=2,
            color="black",
            fill=True,
            fill_opacity=1,
            tooltip=r.Volcano_Na,
        ).add_to(volc_layer)
    m.add_child(volc_layer)

    def style_ring(feat: dict) -> dict:
        """Return style dict for each hazard ring polygon."""

        hz = feat["properties"]["hazard"]
        return {
            "color": COLOR[hz],
            "weight": 0,
            "fillColor": COLOR[hz],
            "fillOpacity": 0.25,
        }

    for _, r in rings_wgs.iterrows():
        # Build a GeoJSON feature for each hazard ring and add it to the map
        feat = {
            "type": "Feature",
            "geometry": mapping(r.geometry),
            "properties": {"hazard": r.hazard},
        }
        pop = tier_tbl.loc[r.hazard, "Pop"]
        folium.GeoJson(
            feat,
            style_function=style_ring,
            tooltip=f"{r.hazard} ring • Pop: {pop}",
        ).add_to(m)

    # Add each city as a coloured circle marker sized by population
    city_layer = folium.FeatureGroup(name="Cities (>100k)")
    for _, r in cities_wgs.iterrows():
        folium.CircleMarker(
            [r.geometry.y, r.geometry.x],
            radius=4,
            color=COLOR[r.hazard],
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{r.NAME} ({int(r.POP_MAX):,}) – {r.hazard}",
        ).add_to(city_layer)
    m.add_child(city_layer)

    folium.LayerControl().add_to(m)

    # Add a custom legend explaining the colour scheme
    from branca.element import Template, MacroElement

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <style>
      .volc-legend {{
        position: fixed;
        bottom: 15px;
        right: 15px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border-radius: 4px;
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
        font-size: 12px;
        line-height: 18px;
      }}
      .volc-legend i {{
        width: 12px;
        height: 12px;
        float: left;
        margin-right: 6px;
        opacity: 0.8;
      }}
    </style>
    <div class="volc-legend">
      <div><i style="background:black"></i> Volcano vent</div>
      <div><i style="background:{COLOR['High']}"></i> High (≤10 km)</div>
      <div><i style="background:{COLOR['Med-High']}"></i> Med-High (10–30 km)</div>
      <div><i style="background:{COLOR['Medium']}"></i> Medium (30–50 km)</div>
      <div><i style="background:{COLOR['Low']}"></i> Low (50–100 km)</div>
      <div><i style="background:dodgerblue"></i> City >100k</div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)
    return m


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------


def main() -> None:
    """Run the analysis and produce the maps."""

    # Load raw datasets from disk
    volcanoes = load_layer(VOLCANO_SHP, "Holocene Volcanoes")
    cities = load_layer(CITIES_SHP, "Populated Places")

    # Print quick summaries so we know what was loaded
    inspect(volcanoes, "Volcano layer")
    inspect(cities, "Cities layer")

    # Reproject to Web Mercator which uses metres – handy for distance buffers
    volcanoes_merc = volcanoes.to_crs(3857)
    cities_merc = cities.to_crs(3857)

    # Limit to large cities only (population > 100k)
    cities_100k = cities_merc.loc[cities_merc["POP_MAX"].fillna(0) > 100_000].copy()
    print(f"Cities >100k : {len(cities_100k):,} of {len(cities_merc):,}")

    # Create a 100 km buffer around each volcano for a simple at-risk search
    buffer_met = 100 * 1000
    volc_buf = volcanoes_merc.copy()
    volc_buf["geometry"] = volcanoes_merc.geometry.buffer(buffer_met)

    # Spatial join to find cities within the buffer polygons
    at_risk = gpd.sjoin(
        cities_100k, volc_buf[["geometry"]], how="inner", predicate="within"
    ).drop_duplicates(subset="NAME")
    at_risk = at_risk.drop(columns="index_right")
    print(f"At-risk cities: {len(at_risk):,}")

    # Build distance rings and classify each city by hazard tier
    hazard_rings = compute_hazard_rings(volcanoes_merc)
    cities_tiered = classify_cities(at_risk, volcanoes_merc)

    # Summarise total population and number of cities per tier
    tier_tbl = cities_tiered.groupby("hazard")["POP_MAX"].agg(Pop="sum", Cities="size")
    print(tier_tbl)

    # Load a base world map for context
    world = load_layer(COUNTRIES_SHP, "Admin-0 countries")

    # ╔═ Country-level exposure summary ═════════════════════════╗
    # 1. Aggregate at-risk cities by country
    exposure_by_ctry = (
        cities_tiered
        .groupby("ADM0NAME")
        .agg(exposed_pop=("POP_MAX", "sum"),
             at_risk_cities=("NAME", "size"))
        .sort_values("exposed_pop", ascending=False)
    )

    # 2. Attach to country polygons (left join keeps all countries for mapping)
    world_w_exposure = world.to_crs(4326).merge(
        exposure_by_ctry,
        how="left",
        left_on="NAME",
        right_index=True,
        validate="1:1",
    )

    # 3. Fill NaN with 0 for mapping
    world_w_exposure["exposed_pop"] = world_w_exposure["exposed_pop"].fillna(0)
    world_w_exposure["at_risk_cities"] = world_w_exposure["at_risk_cities"].fillna(0)

    # ------------------ Static bar of top 15 countries ------------------
    top15 = exposure_by_ctry.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top15.index[::-1], top15["exposed_pop"][::-1] / 1e6, color="#fc8d59")
    plt.xlabel("Population exposed (millions)")
    plt.title("Top 15 Countries by Population in ≤100 km of a Volcano")
    plt.tight_layout()
    plt.show()

    # ------------------ Folium choropleth ------------------
    chor_map = folium.Map(location=[0, 20], zoom_start=2, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=world_w_exposure,
        name="Country exposure",
        data=world_w_exposure,
        columns=["NAME", "exposed_pop"],
        key_on="feature.properties.NAME",
        fill_color="YlOrRd",
        nan_fill_color="lightgray",
        fill_opacity=0.8,
        legend_name="Population in cities ≤100 km of a volcano",
        bins=[0, 1e6, 5e6, 10e6, 25e6, 50e6, 100e6, 200e6],
    ).add_to(chor_map)

    # Add simple pop-up on click
    style_no_border = {"weight": 0.3, "color": "gray", "fillOpacity": 0}
    folium.GeoJson(
        world_w_exposure,
        style_function=lambda *_: style_no_border,
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "exposed_pop", "at_risk_cities"],
            aliases=["Country", "Exposed pop", "At-risk cities"],
            localize=True,
        ),
    ).add_to(chor_map)

    folium.LayerControl().add_to(chor_map)
    display(chor_map)
    # ╚══════════════════════════════════════════════════════════╝

    # Use Mercator version for static plotting
    world_merc = world.to_crs(3857)
    # Show a matplotlib overview
    plot_static(world_merc, volcanoes_merc, at_risk)

    # Build and display the interactive web map (in Jupyter this shows inline)
    m = create_interactive_map(volcanoes, hazard_rings, cities_tiered, tier_tbl)
    display(m)

    # Finally plot a bar chart of the ten largest at-risk cities
    plt.figure(figsize=(8, 6))
    top10 = at_risk.nlargest(10, "POP_MAX").sort_values("POP_MAX")
    plt.barh(top10["NAME"], top10["POP_MAX"] / 1_000_000, color="steelblue")
    plt.xlabel("Population (millions)")
    plt.title("Ten Most Populous Cities within 100 km of a Volcano")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
