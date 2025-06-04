from __future__ import annotations
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import mapping

# ------------------ Paths & constants ------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

VOLCANO_SHP = DATA_DIR / "Global_2013_HoloceneVolcanoes_SmithsonianVOTW" / "Smithsonian_VOTW_Holocene_VolcanoesPoint.shp"
CITIES_SHP = DATA_DIR / "ne_10m_populated_places" / "ne_10m_populated_places.shp"
COUNTRIES_SHP = DATA_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"

COLOR = {
    "High": "#d73027",
    "Med-High": "#fc8d59",
    "Medium": "#fee08b",
    "Low": "#d9ef8b",
}

# ------------------ Utility functions ------------------

def load_layer(path: Path, name: str) -> gpd.GeoDataFrame:
    """Load a vector layer and perform basic checks."""
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{name} loaded but is empty")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    return gdf


def inspect(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Pretty-print basic GeoDataFrame info."""
    print(f"{name}:")
    print(f"  Records : {len(gdf):,}")
    print(f"  CRS     : {gdf.crs}")
    print(f"  Columns : {list(gdf.columns)[:10]} ...")
    print(gdf.head(3))


# ------------------ Analysis helpers ------------------

def compute_hazard_rings(volcanoes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return polygons for distance-based hazard rings."""
    tiers = [
        (0, 10, "High"),
        (10, 30, "Med-High"),
        (30, 50, "Medium"),
        (50, 100, "Low"),
    ]
    rings = []
    for low, high, name in tiers:
        outer = volcanoes.geometry.buffer(high * 1000)
        ring = gpd.GeoSeries(outer).unary_union
        if low > 0:
            inner = volcanoes.geometry.buffer(low * 1000)
            ring = ring.difference(gpd.GeoSeries(inner).unary_union)
        rings.append({"hazard": name, "geometry": ring})
    return gpd.GeoDataFrame(rings, crs=volcanoes.crs)


def classify_cities(cities: gpd.GeoDataFrame, volcanoes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach hazard tier based on distance to nearest volcano."""
    nearest = gpd.sjoin_nearest(cities, volcanoes[["geometry"]], how="left", distance_col="dist")
    cities = nearest.drop(columns="index_right")
    cities["dist_km"] = cities.pop("dist") / 1000.0

    def tier(d: float) -> str:
        if d <= 10:
            return "High"
        if d <= 30:
            return "Med-High"
        if d <= 50:
            return "Medium"
        return "Low"

    cities["hazard"] = cities["dist_km"].apply(tier)
    return cities


def plot_static(world: gpd.GeoDataFrame, volcanos: gpd.GeoDataFrame, cities: gpd.GeoDataFrame) -> None:
    ROB = "ESRI:54030"
    world_robin = world.to_crs(ROB)
    volc_robin = volcanos.to_crs(ROB)
    cities_robin = cities.to_crs(ROB)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)
    world_robin.plot(ax=ax, color="#f7f7f7", edgecolor="lightgray", linewidth=0.4)
    volc_robin.plot(ax=ax, marker="^", markersize=10, color="red", alpha=0.7, label="Holocene volcano")
    cities_robin.plot(ax=ax, marker="o", markersize=20, color="dodgerblue", alpha=0.6, label="City (>100k) ≤100km")
    ax.set_title("Major Cities within 100 km of an Active (Holocene) Volcano", fontsize=15)
    ax.legend(frameon=False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def create_interactive_map(volcanoes: gpd.GeoDataFrame, hazard_rings: gpd.GeoDataFrame, cities: gpd.GeoDataFrame, tier_tbl: pd.DataFrame) -> folium.Map:
    rings_wgs = hazard_rings.to_crs(4326)
    cities_wgs = cities.to_crs(4326)

    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")

    volc_layer = folium.FeatureGroup(name="Volcanoes")
    for _, r in volcanoes.iterrows():
        folium.CircleMarker([r.Latitude, r.Longitude], radius=2, color="black", fill=True, fill_opacity=1, tooltip=r.Volcano_Na).add_to(volc_layer)
    m.add_child(volc_layer)

    def style_ring(feat):
        hz = feat["properties"]["hazard"]
        return {"color": COLOR[hz], "weight": 0, "fillColor": COLOR[hz], "fillOpacity": 0.25}

    for _, r in rings_wgs.iterrows():
        feat = {"type": "Feature", "geometry": mapping(r.geometry), "properties": {"hazard": r.hazard}}
        pop = tier_tbl.loc[r.hazard, "Pop"]
        folium.GeoJson(feat, style_function=style_ring, tooltip=f"{r.hazard} ring • Pop: {pop}").add_to(m)

    city_layer = folium.FeatureGroup(name="Cities (>100k)")
    for _, r in cities_wgs.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color=COLOR[r.hazard], fill=True, fill_opacity=0.7, tooltip=f"{r.NAME} ({int(r.POP_MAX):,}) – {r.hazard}").add_to(city_layer)
    m.add_child(city_layer)

    folium.LayerControl().add_to(m)

    from branca.element import Template, MacroElement
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <style>
      .volc-legend {{
        position: fixed;
        bottom: 15px; right: 15px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border-radius: 4px;
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
        font-size: 12px;
        line-height: 18px;
      }}
      .volc-legend i {{
        width: 12px; height: 12px;
        float: left; margin-right: 6px;
        opacity: 0.8;
      }}
    </style>
    <div class="volc-legend">
      <div><i style="background:black"></i> Volcano vent</div>
      <div><i style="background:{COLOR['High']}"></i> High (≤10 km)</div>
      <div><i style="background:{COLOR['Med-High']}"></i> Med‑High (10–30 km)</div>
      <div><i style="background:{COLOR['Medium']}"></i> Medium (30–50 km)</div>
      <div><i style="background:{COLOR['Low']}"></i> Low (50–100 km)</div>
      <div><i style="background:dodgerblue"></i> City >100 k</div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)
    return m


# ------------------ Main workflow ------------------

def main() -> None:
    volcanoes = load_layer(VOLCANO_SHP, "Holocene Volcanoes")
    cities = load_layer(CITIES_SHP, "Populated Places")
    inspect(volcanoes, "Volcano layer")
    inspect(cities, "Cities layer")

    volcanoes_merc = volcanoes.to_crs(3857)
    cities_merc = cities.to_crs(3857)

    cities_100k = cities_merc.loc[cities_merc["POP_MAX"].fillna(0) > 100_000].copy()
    print(f"Cities >100k : {len(cities_100k):,} of {len(cities_merc):,}")

    buffer_met = 100 * 1000
    volc_buf = volcanoes_merc.copy()
    volc_buf["geometry"] = volcanoes_merc.geometry.buffer(buffer_met)

    at_risk = gpd.sjoin(cities_100k, volc_buf[["geometry"]], how="inner", predicate="within").drop_duplicates(subset="NAME")
    print(f"At-risk cities: {len(at_risk):,}")

    hazard_rings = compute_hazard_rings(volcanoes_merc)
    cities_tiered = classify_cities(at_risk, volcanoes_merc)

    tier_tbl = cities_tiered.groupby("hazard")["POP_MAX"].agg(Pop="sum", Cities="size")
    print(tier_tbl)

    world = load_layer(COUNTRIES_SHP, "Admin-0 countries").to_crs(3857)

    plot_static(world, volcanoes_merc, at_risk)

    m = create_interactive_map(volcanoes, hazard_rings, cities_tiered, tier_tbl)
    m.save("volcano_cities_map.html")
    print("Interactive map saved to volcano_cities_map.html")

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
