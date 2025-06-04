# Geospatial Python Exercises

This repository contains notebooks and scripts used for basic geospatial data processing in Python. The main script of interest is `final_original.py` which performs an analysis of volcano hazards and nearby cities.

## Project goal

`final_original.py` loads volcano, city and country datasets, identifies cities with a population greater than 100,000 that lie within 100 km of an active (Holocene) volcano and visualises the results. A static map is produced with `matplotlib` and an interactive map is created with `folium`.

## Data sources

The script expects three vector datasets in ESRI Shapefile format:

- **Holocene volcanoes** – "Global 2013 Holocene Volcanoes" from the Smithsonian Volcanoes of the World catalogue.
- **Populated places** – Natural Earth 1:10m populated places dataset.
- **Admin 0 countries** – Natural Earth 1:50m administrative boundaries.

These datasets are not included in the repository. They must be downloaded from their respective providers and placed in the `data/` directory described below.

## Preparing the data

Create a folder named `data/` in the project root with the following structure:

```
Geospatial-Python/
├── data/
│   ├── Global_2013_HoloceneVolcanoes_SmithsonianVOTW/
│   │   └── Smithsonian_VOTW_Holocene_VolcanoesPoint.shp
│   ├── ne_10m_populated_places/
│   │   └── ne_10m_populated_places.shp
│   └── ne_50m_admin_0_countries/
│       └── ne_50m_admin_0_countries.shp
```

Each folder should contain the full set of files that make up a Shapefile (`.shp`, `.dbf`, `.shx`, etc.).

## Installation

Create a Python environment and install the requirements:

```bash
pip install -r requirements.txt
```

`geopandas` has several compiled dependencies, so consider using `conda` if you encounter installation issues.

## Running `final_original.py`

Run the script from the project root so that the relative `data/` paths resolve correctly:

```bash
python final_original.py
```

The script prints summary information, displays a static map with `matplotlib` and opens an interactive map using `folium`. When executed in a Jupyter environment, the map will display inline; otherwise you can save it manually (e.g. `m.save("map.html")`).

