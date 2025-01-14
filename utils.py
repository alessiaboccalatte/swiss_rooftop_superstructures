import numpy as np
import torch
import torchvision
from shapely.geometry import Polygon
from skimage import measure
import rasterio
import geopandas as gpd

def divide_img_tiles(image, tile_size, stride):
    """Divide the image into tiles."""
    img_size_y, img_size_x, _ = image.shape
    tiles = []
    for h in range(0, img_size_y, stride):
        for w in range(0, img_size_x, stride):
            tile = image[h:h+tile_size, w:w+tile_size]
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            tiles.append(tile)
    return np.array(tiles)

def segment_tile(tile, model, tile_size, background_class):
    """Segment a single tile."""
    trans = torchvision.transforms.ToTensor()
    tile_tensor = trans(tile).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        pred = model(tile_tensor).argmax(dim=1).cpu().numpy().squeeze()
    pred[pred == background_class] = 0
    return pred

def compose_tiles(tiles, img_shape, stride):
    """Reconstruct the image from tiles."""
    img_size_y, img_size_x = img_shape
    final_img = np.zeros((img_size_y, img_size_x))
    tile_idx = 0
    for h in range(0, img_size_y, stride):
        for w in range(0, img_size_x, stride):
            tile = tiles[tile_idx]
            final_img[h:h+tile.shape[0], w:w+tile.shape[1]] = tile
            tile_idx += 1
    return final_img

def segment_img(image, model, tile_size, stride, background_class):
    """Segment an entire image."""
    tiles = divide_img_tiles(image, tile_size, stride)
    seg_tiles = [segment_tile(tile, model, tile_size, background_class) for tile in tiles]
    return compose_tiles(seg_tiles, image.shape[:2], stride)

def extract_polygons(prediction_mask, class_value, transform, central_polygon):
    """Extract polygons of the specified class."""
    mask = (prediction_mask == class_value).astype(np.uint8)
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        xs, ys = rasterio.transform.xy(transform, contour[:, 0], contour[:, 1])
        coords = list(zip(xs, ys))
        if len(coords) >= 3:
            poly = Polygon(coords)
            if poly.is_valid and poly.intersects(central_polygon):
                polygons.append(poly)
    return polygons

def dissolve_and_buffer(gdf, buffer_distance):
    """Dissolve polygons and apply buffer."""
    dissolved = gdf.dissolve()
    dissolved['geometry'] = dissolved.buffer(buffer_distance)
    return dissolved.explode(index_parts=False).reset_index(drop=True)
