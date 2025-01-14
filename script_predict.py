#!/usr/bin/env python
# coding: utf-8

import sys
import os
import torch
import pathlib
import pandas as pd
import geopandas as gpd
import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon
from tqdm import tqdm

from utils import divide_img_tiles, segment_tile, compose_tiles, segment_img, extract_polygons, dissolve_and_buffer

###############################################################################
# Description:
# This script:
# 1. Loads a trained segmentation model (superstructures.pth).
# 2. Processes PNG and corresponding TIFF images to generate segmentation predictions.
# 3. Extracts polygons for superstructures (class_value=2) and saves them as predictions.shp.
# 4. Updates the roof dataset with superstructure information.
# 5. Calculates total solar energy (GWh), energy from roofs without superstructures, and lost energy.
# 6. Saves the updated GeoDataFrame with a `has_superstructures` column.
###############################################################################

base_dir = os.path.dirname(os.path.realpath(__file__))

# Define paths
models_folder_path = os.path.join(base_dir, 'models')
model_path = os.path.join(models_folder_path, 'superstructures.pth')

png_folder = os.path.join(base_dir, 'data', 'test_images')
tif_folder = os.path.join(base_dir, 'data', 'test_images_tif')

predictions_folder = os.path.join(base_dir, 'output', 'predictions_shapefile')
os.makedirs(predictions_folder, exist_ok=True)
predictions_output_path = os.path.join(predictions_folder, 'predictions.shp')

# Path to the filtered roofs shapefile
filtered_shp_path = os.path.join(base_dir, 'output', 'filtered_shapefile', 'filtered.shp')

# Path to the updated roofs shapefile
updated_roofs_path = os.path.join(base_dir, 'output', 'filtered_shapefile', 'updated_roofs.shp')

# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_test = torch.load(model_path, map_location=device)

# Prepare an empty GeoDataFrame for superstructures
sovrastrutture_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:2056")

def generate_predictions_shapefile(model, png_folder, tif_folder, output_path):
    """
    Generate predictions for superstructures, save them as a shapefile.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        png_folder (str): Folder containing PNG images.
        tif_folder (str): Folder containing corresponding TIFF images.
        output_path (str): Path to save the shapefile of predicted superstructures.
    """
    global sovrastrutture_gdf
    image_files = list(pathlib.Path(png_folder).glob('*.png'))

    for filename in tqdm(image_files, desc="Processing images"):
        image = cv2.imread(str(filename))
        if image is None:
            print(f"[ERROR] Cannot read image {filename}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tif_path = pathlib.Path(tif_folder) / (filename.stem + ".tif")
        if not tif_path.exists():
            print(f"[ERROR] No TIFF found for {filename.stem}")
            continue

        with rasterio.open(tif_path) as src:
            transform = src.transform
            bounds = src.bounds

        pred = segment_img(image, model, tile_size=256, stride=256, background_class=0)
        if pred is None:
            print(f"[ERROR] Prediction returned None for image: {filename}")
            continue

        central_polygon = Polygon([
            (bounds.left, bounds.top),
            (bounds.left, bounds.bottom),
            (bounds.right, bounds.bottom),
            (bounds.right, bounds.top),
            (bounds.left, bounds.top)
        ])

        polygons = extract_polygons(pred, class_value=2, transform=transform, central_polygon=central_polygon)
        if polygons:
            sovrastrutture_gdf = pd.concat([sovrastrutture_gdf, gpd.GeoDataFrame(geometry=polygons, crs="EPSG:2056")], ignore_index=True)

    if not sovrastrutture_gdf.empty:
        predicted_buffered = dissolve_and_buffer(sovrastrutture_gdf, buffer_distance=-1.0)
        predicted_buffered.to_file(output_path, driver='ESRI Shapefile')
        print(f"[INFO] Buffered and dissolved superstructures saved at: {output_path}")
    else:
        print("[INFO] No polygons extracted. Shapefile not created.")

if __name__ == "__main__":
    # Step 1: Generate predictions and save the shapefile
    generate_predictions_shapefile(model_test, png_folder, tif_folder, predictions_output_path)

    # Step 2: Load roofs
    roofs_gdf = gpd.read_file(filtered_shp_path).to_crs("EPSG:2056")
    print("[INFO] Loaded filtered roofs.")

    # Ensure unique IDs for roof segments
    roofs_gdf['sg_id'] = roofs_gdf.index
    roofs_gdf.drop_duplicates(subset=['sg_id'], inplace=True)

    # Step 3: Load processed predictions
    if os.path.exists(predictions_output_path):
        predictions_gdf = gpd.read_file(predictions_output_path).to_crs("EPSG:2056")
    else:
        predictions_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:2056")

    # Step 4: Determine roofs with superstructures
    if not predictions_gdf.empty:
        intersection = gpd.overlay(roofs_gdf[['sg_id', 'geometry']], predictions_gdf, how='intersection')
        intersection['area_superstructures'] = intersection.area
        area_per_segment = intersection.groupby('sg_id', as_index=False)['area_superstructures'].sum()
        roofs_gdf = roofs_gdf.merge(area_per_segment, on='sg_id', how='left')
        roofs_gdf['area_superstructures'] = roofs_gdf['area_superstructures'].fillna(0)
        roofs_gdf['has_superstructures'] = (roofs_gdf['area_superstructures'] >= 1.0).astype(int)
    else:
        roofs_gdf['has_superstructures'] = 0
        roofs_gdf['area_superstructures'] = 0

    # Step 5: Calculate energy statistics
    roofs_gdf['total_energy_kwh'] = roofs_gdf['irr'] * roofs_gdf['real_area']
    total_energy_kwh = roofs_gdf['total_energy_kwh'].sum()
    no_super_energy_kwh = roofs_gdf.loc[roofs_gdf['has_superstructures'] == 0, 'total_energy_kwh'].sum()
    lost_energy_kwh = total_energy_kwh - no_super_energy_kwh
    total_energy_gwh = total_energy_kwh / 1_000_000
    lost_energy_gwh = lost_energy_kwh / 1_000_000
    lost_energy_percentage = (lost_energy_kwh / total_energy_kwh) * 100 if total_energy_kwh > 0 else 0.0

    # Print summary statistics
    print("[INFO] Statistics:")
    print(f"Total roof segments: {len(roofs_gdf)}")
    print(f"Roof segments with superstructures: {roofs_gdf['has_superstructures'].sum()}")
    print(f"Roof segments without superstructures: {len(roofs_gdf) - roofs_gdf['has_superstructures'].sum()}")
    print(f"Total energy: {total_energy_gwh:.6f} GWh/year")
    print(f"Lost energy: {lost_energy_gwh:.6f} GWh/year ({lost_energy_percentage:.2f}% of total)")

    # Step 6: Save the updated GeoDataFrame
    roofs_gdf.to_file(updated_roofs_path, driver='ESRI Shapefile')
    print(f"[INFO] Updated roofs saved at: {updated_roofs_path}")
