import os
import numpy as np
import rasterio
from rasterio.transform import rowcol
import geopandas as gpd

# ============================
# Cargar imágenes y coordenadas
# ============================
def load_tiff_images_with_coords(folder):
    images, x_coords_list, y_coords_list = [], [], []
    files = sorted(os.listdir(folder))

    for file in files:
        if file.endswith(".tiff") or file.endswith(".tif"):
            file_path = os.path.join(folder, file)
            with rasterio.open(file_path) as src:
                image = src.read(1).astype(np.float64)
                image = np.round(image, 4)
                image = np.expand_dims(image, axis=-1)
                images.append(image)

                # Obtener coordenadas X e Y
                height, width = src.height, src.width
                transform = src.transform
                x_coords = np.array([transform * (x, 0) for x in range(width)])[:, 0]
                x_coords = np.tile(x_coords[np.newaxis, :], (height, 1))
                y_coords = np.array([[transform * (0, y)][0][1] for y in range(height)])
                y_coords = np.tile(y_coords[:, np.newaxis], (1, width))

                x_coords_list.append(x_coords)
                y_coords_list.append(y_coords)

    return (
        np.array(images, dtype=np.float64),
        np.array(x_coords_list, dtype=np.float64),
        np.array(y_coords_list, dtype=np.float64)
    )

def extract_observations_from_shapefile(shp_file, tiff_file, X_data, depth_true):
    """
    Extracts coordinates and true depth values from point locations in a shapefile.

    Parameters:
    - shp_file: path to the shapefile (.shp) containing geometries and an 'id' field.
    - tiff_file: reference geospatial raster file.
    - X_data: array of shape (1, H, W, 3) containing spatial information (e.g., x, y coordinates).
    - depth_true: array of shape (H, W) with true depth values.

    Returns:
    - pixel_x: array with row indices corresponding to each shapefile point in the raster.
    - pixel_y: array with column indices corresponding to each shapefile point in the raster.
    - x_coords_out: array of X coordinates extracted from X_data.
    - y_coords_out: array of Y coordinates extracted from X_data.
    - depth_values: array of depth values from depth_true.
    """
    gdf = gpd.read_file(shp_file)
    shp_x_coords, shp_y_coords = gdf.geometry.x, gdf.geometry.y

    with rasterio.open(tiff_file) as src:
        rows_cols = [rowcol(src.transform, x, y) for x, y in zip(shp_x_coords, shp_y_coords)]

    x_coords_out = []
    y_coords_out = []
    depth_values = []

    pixel_x = []
    pixel_y = []

    for row, col in rows_cols:
        try:
            x = X_data[row, col, 1]  # X coordinate from X_data
            y = X_data[row, col, 2]  # Y coordinate from X_data
            d_true = depth_true[row, col]  # True depth from depth_true

            x_coords_out.append(x)
            y_coords_out.append(y)
            pixel_x.append(row)
            pixel_y.append(col)
            depth_values.append(d_true)
        except IndexError:
            continue

    return np.array(pixel_x), np.array(pixel_y), np.array(x_coords_out), np.array(y_coords_out), np.array(depth_values)
