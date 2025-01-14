#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import rasterio
from rasterio.io import MemoryFile
import time
from owslib.wms import WebMapService
from requests.exceptions import HTTPError
from PIL import Image
from tqdm import tqdm

###############################################################################
# Description:
# This script:
# 1. Reads a polygon of interest from a shapefile.
# 2. Filters the roofs dataset (GPKG) using a bounding box and precise spatial filtering.
# 3. Removes non-building categories based on a specified column.
# 4. Factorizes and renames columns to prepare the dataset.
# 5. Downloads both TIFF and PNG images from a WMS for each building.
###############################################################################

base_dir = os.path.dirname(os.path.realpath(__file__))

# Paths to input and output files
roofs_gpkg = os.path.join(base_dir, 'data', 'SOLKAT_DACH.gpkg')  # Path to the large roofs dataset (GeoPackage)
polygon_path = os.path.join(base_dir, 'data', 'area_of_interest.shp')  # Path to the polygon of interest

# Path to save the filtered shapefile
filtered_shp_path = os.path.join(base_dir, 'output', 'filtered_shapefile', 'filtered.shp')

# Directories to save generated images
tif_dir = os.path.join(base_dir, 'data', 'test_images_tif')  # Directory for TIFF images
png_dir = os.path.join(base_dir, 'data', 'test_images')  # Directory for PNG images

# Image settings for downloading from WMS
img_size_px = 256  # Size of the image in pixels
img_res_m_per_px = 0.2  # Image resolution in meters per pixel
img_size_m = img_size_px * img_res_m_per_px  # Total image size in meters

# Create necessary directories
os.makedirs(os.path.dirname(filtered_shp_path), exist_ok=True)
os.makedirs(tif_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

if __name__ == "__main__":
    # 1. Read polygon of interest
    print("Reading polygon of interest...")
    area_polygon = gpd.read_file(polygon_path)  # Load the shapefile containing the polygon of interest
    print("Polygon of interest loaded.")

    # Ensure CRS alignment between the roofs dataset and the polygon
    temp = gpd.read_file(roofs_gpkg, rows=1)  # Load a small portion of the roofs dataset to check CRS
    if temp.crs != area_polygon.crs:
        area_polygon = area_polygon.to_crs(temp.crs)  # Align CRS

    # 2. Use bbox filtering to load only data within the bounding box of the polygon
    bbox_array = area_polygon.total_bounds  # Calculate the bounding box of the polygon
    bbox = tuple(bbox_array)  # Convert bounding box to a tuple (minx, miny, maxx, maxy)

    print("Reading roofs dataset using bbox to limit data...")
    roofs_bbox = gpd.read_file(roofs_gpkg, bbox=bbox)  # Load only features within the bounding box
    print(f"Number of features after bbox filtering: {len(roofs_bbox)}")

    # Apply precise filtering using the polygon geometry
    print("Applying precise spatial filter (within) with progress bar...")
    area_poly_union = area_polygon.unary_union  # Combine all polygons into a single geometry
    filtered_features = [
        row for idx, row in tqdm(roofs_bbox.iterrows(), total=len(roofs_bbox), desc="Filtering")
        if row.geometry.within(area_poly_union)
    ]

    # Create a GeoDataFrame from the filtered features
    roofs_subset = gpd.GeoDataFrame(filtered_features, crs=roofs_bbox.crs)
    print(f"Number of roofs after precise filtering: {len(roofs_subset)}")

    # 3. Remove non-building categories based on a specific column
    if 'SB_OBJEKTART' in roofs_subset.columns:
        print("Filtering non-building categories...")
        non_building_categories = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]
        roofs_subset = roofs_subset[~roofs_subset['SB_OBJEKTART'].isin(non_building_categories)]
        print(f"Filtered non-buildings. Remaining buildings: {len(roofs_subset)}")
    else:
        print("[WARNING] Column 'SB_OBJEKTART' not found. No filtering applied.")

    # 4. Rename and save filtered data
    columns_to_use = ['DF_UID', 'DF_NUMMER', 'SB_UUID', 'SB_OBJEKTART', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'FLAECHE', 'geometry']
    available_columns = [c for c in columns_to_use if c in roofs_subset.columns]
    gen = roofs_subset[available_columns].copy()

    # Factorize the SB_UUID column and rename columns
    if 'SB_UUID' in gen.columns:
        gen['SB_UUID_Num'], unique = pd.factorize(gen['SB_UUID'])
        print(f"SB_UUID factorization completed: {len(unique)} unique IDs.")

    rename_dict = {
        'SB_UUID_Num': 'b_id',
        'DF_UID': 'sg_id',
        'DF_NUMMER': 'sg_no',
        'AUSRICHTUN': 'azimuth',
        'NEIGUNG': 'slope',
        'MSTRAHLUNG': 'irr',
        'FLAECHE': 'real_area'
    }
    segments = gen.rename(columns=rename_dict)

    # Save the filtered and renamed dataset as a shapefile
    final_columns = ['b_id', 'sg_id', 'sg_no', 'azimuth', 'slope', 'irr', 'real_area', 'geometry']
    final_columns = [c for c in final_columns if c in segments.columns]
    final_gdf = segments[final_columns]
    final_gdf.to_file(filtered_shp_path)
    print(f"Filtered and renamed roofs saved to {filtered_shp_path}")

    # 5. Dissolve segments by building ID and create square buffers for WMS requests
    print("Dissolving segments by building ID...")
    buildings = segments[['b_id', 'geometry']].dissolve(by='b_id', as_index=False)
    buildings['centroid'] = buildings.centroid
    buildings['buffer'] = buildings['centroid'].buffer(img_size_m / 2, cap_style=3)

    # 6. Download images for each building
    print("Connecting to WMS...")
    wms = WebMapService('https://wms.geo.admin.ch/?', version='1.3.0')
    print("Downloading building images...")

    for index, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Processing buildings"):
        retries = 5
        tile = None
        for attempt in range(retries):
            try:
                # Request the image from WMS
                tile = wms.getmap(
                    layers=['ch.swisstopo.swissimage'],
                    styles=['default'],
                    srs='EPSG:2056',
                    bbox=row['buffer'].bounds,
                    size=(img_size_px, img_size_px),
                    format='image/tiff',
                    transparent=True,
                    timeout=120
                )
                break
            except HTTPError as e:
                print(f"HTTP error {e.response.status_code}. Attempt {attempt + 1}/{retries}")
                time.sleep(5)
            except ConnectionError as e:
                print(f"Connection error: {e}. Attempt {attempt + 1}/{retries}")
                time.sleep(5)

        if tile is None:
            continue

        # Save TIFF
        b_id = row['b_id']
        tif_filepath = os.path.join(tif_dir, f"{b_id}.tif")
        with open(tif_filepath, "wb") as f:
            f.write(tile.read())

        # Convert TIFF to PNG
        with rasterio.open(tif_filepath) as src:
            img_array = src.read([1, 2, 3])
            img_array = np.transpose(img_array, (1, 2, 0))
        png_filepath = os.path.join(png_dir, f"{b_id}.png")
        img = Image.fromarray(img_array)
        img.save(png_filepath)

    print("Dataset creation and image generation completed.")
    sys.exit(0)

