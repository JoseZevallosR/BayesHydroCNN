# ============================
# Required Libraries
# ============================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rasterio
from rasterio.plot import show
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from PIL import Image

# ============================
# Plot subplots: Friction vs Depth and Emulator vs Simulator
# ============================
def plot_subplots(data, output_path='subplots_depth_vs_friction.png'):
    """
    Generates subplots for each shapefile point showing:
    - Friction (n) vs Real Depth
    - Friction (n) vs Predicted Depth
    Includes:
    - Correlation (r) between n and depth, and n and predicted depth
    - RMSE between simulated and predicted depth
    """
    unique_points = list({(row, col, pid) for row, col, pid, *_ in data})
    num_points = len(unique_points)
    ncols = 4
    nrows = int(np.ceil(num_points / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (row, col, pid) in enumerate(unique_points):
        point_data = [(n, depth, depth_pred) for r, c, p, *_ , n, depth, depth_pred in data
                      if r == row and c == col and p == pid]
        
        if point_data:
            n_vals, depth_vals, depth_pred_vals = zip(*point_data)

            # Plot real and predicted depths
            axes[idx].scatter(n_vals, depth_vals, alpha=0.7, edgecolors='k', label="Simulator", color='blue')
            axes[idx].scatter(n_vals, depth_pred_vals, alpha=0.7, edgecolors='k', label="Emulator", color='red')

            # Compute correlation and RMSE
            try:
                r_real, _ = pearsonr(n_vals, depth_vals)
                r_pred, _ = pearsonr(n_vals, depth_pred_vals)
            except Exception:
                r_real, r_pred = np.nan, np.nan

            rmse = np.sqrt(mean_squared_error(depth_vals, depth_pred_vals))

            # Title and metrics
            axes[idx].set_title(f'ID {pid} RMSE(sim-emu)={rmse:.2f}\n'
                                f'r(n,sim)={r_real:.2f}, r(n,emu)={r_pred:.2f}')
            axes[idx].set_xlabel('Friction (n)')
            axes[idx].set_ylabel('Depth')
            axes[idx].legend()

    # Remove unused subplots
    for ax in axes[len(unique_points):]:
        fig.delaxes(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.show()

# ============================
# Plot points on map with their IDs
# ============================
def plot_points_on_map_with_ids(tiff_path, shapefile_path, output_path='points_on_map.png'):
    """
    Plots shapefile observation points over a GeoTIFF raster background,
    labeling each with its point ID and including a scalebar.

    Parameters:
        tiff_path (str): Path to GeoTIFF raster file.
        shapefile_path (str): Path to shapefile with observation points.
        output_path (str): Destination file for the saved plot (PNG + TIFF).
    """
    gdf = gpd.read_file(shapefile_path)

    with rasterio.open(tiff_path) as src:
        fig, ax = plt.subplots(figsize=(10, 8))

        show(src, ax=ax, cmap='terrain')
        gdf.plot(ax=ax, color='red', markersize=20, zorder=3)
        plt.rcParams['font.family'] = 'DejaVu Sans'

        for idx, row in gdf.iterrows():
            x, y = row.geometry.x, row.geometry.y
            point_id = row['id']
            ax.text(x, y, str(point_id), fontsize=12, ha='left', va='bottom', color='black', zorder=4)

        ax.set_title("(a) Observation points used for model validation", fontsize=14)

        try:
            resolution = src.res[0]
            scalebar = ScaleBar(dx=resolution, units='m', location='lower right', box_alpha=0.5)
            ax.add_artist(scalebar)
        except Exception as e:
            print("Warning: Scale bar could not be added:", e)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"PNG map saved to: {os.path.abspath(output_path)}")

        tiff_output = output_path.replace(".png", ".tiff")
        fig.savefig(tiff_output, dpi=600, format='tiff', bbox_inches='tight')
        print(f"TIFF version saved to: {os.path.abspath(tiff_output)}")

        plt.show()

# ============================
# Compute and plot correlation heatmap
# ============================
def calculate_pixel_correlation(X_train, y_train, output_path='correlation_heatmap.png'):
    """
    Computes Pearson correlation between friction (n) and depth at each pixel location.
    Outputs a high-resolution heatmap image.

    Parameters:
        X_train (np.ndarray): Input data (roughness), shape (samples, height, width, channels)
        y_train (np.ndarray): Target data (depth), same shape
        output_path (str): File path to save the correlation heatmap
    """
    num_images, height, width, _ = X_train.shape
    correlation_matrix = np.full((height, width), np.nan)

    for row in range(height):
        for col in range(width):
            roughness_values = X_train[:, row, col, 0]
            depth_values = y_train[:, row, col, 0]
            if np.std(roughness_values) > 0 and np.std(depth_values) > 0:
                correlation_matrix[row, col] = np.corrcoef(roughness_values, depth_values)[0, 1]

    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    ax = sns.heatmap(
        correlation_matrix,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Pearson correlation (r)'},
        square=True,
        xticklabels=False,
        yticklabels=False
    )

    plt.title("(b) Spatial correlation between Manning’s n and water depth", fontsize=14)
    plt.xlabel("Column index (grid cell)", fontsize=12)
    plt.ylabel("Row index (grid cell)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Correlation heatmap saved at: {os.path.abspath(output_path)}")
    plt.show()

    return correlation_matrix

# ============================
# Extract most frequent friction values
# ============================
def get_n1_n2(X_val, indices):
    """
    Returns the two most frequent friction values (n1, n2) for each image index in the input set.

    Parameters:
        X_val (np.ndarray): Validation images (num_samples, height, width, channels)
        indices (list or np.ndarray): Indices of images to process

    Returns:
        np.ndarray: Array of (n1, n2) tuples for each selected image
    """
    rugosities = []

    for idx in indices:
        friction_values = X_val[idx, :, :, 0].flatten()
        values, counts = np.unique(friction_values, return_counts=True)
        top_indices = np.argsort(-counts)[:2]
        n1, n2 = values[top_indices] if len(top_indices) == 2 else (values[0], values[0])
        rugosities.append((n1, n2))

    return np.array(rugosities)

# ============================
# Combine two images horizontally
# ============================
def combine_images_side_by_side_same_size(img1_path, img2_path, output_path='combined_side_by_side.png'):
    """
    Combines two images horizontally (side-by-side), resizing the second to match the first's height if needed.

    Parameters:
        img1_path (str): First image path
        img2_path (str): Second image path
        output_path (str): Path to save the combined image
    """
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    if img1.size[1] != img2.size[1]:
        new_height = img1.size[1]
        new_width = int((new_height / img2.size[1]) * img2.size[0])
        img2 = img2.resize((new_width, new_height), Image.Resampling.LANCZOS)

    total_width = img1.size[0] + img2.size[0]
    max_height = max(img1.size[1], img2.size[1])
    combined_img = Image.new('RGB', (total_width, max_height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.size[0], 0))
    combined_img.save(output_path)

    print(f"Combined image saved to: {os.path.abspath(output_path)}")

# ============================
# Combine subplot images vertically (e.g., test and train)
# ============================
def combine_subplots_only(test_img_path, train_img_path, output_path='combined_subplots_only.png'):
    """
    Combines subplot images from test and training sets into a vertically stacked figure.

    Parameters:
        test_img_path (str): Path to test image
        train_img_path (str): Path to training image
        output_path (str): Destination file for combined image
    """
    test_img = Image.open(test_img_path)
    train_img = Image.open(train_img_path)

    fig, axes = plt.subplots(2, 1, figsize=(20, 16))

    axes[0].imshow(test_img)
    axes[0].axis('off')
    axes[0].set_title('Test set', fontsize=18)

    axes[1].imshow(train_img)
    axes[1].axis('off')
    axes[1].set_title('Training set', fontsize=18)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to: {os.path.abspath(output_path)}")
    plt.show()

# ============================
# Save pixel-wise error maps
# ============================
def save_error_maps(mae_pixel, rmse_pixel, medae_pixel, std_pixel, output_path="error_maps.png"):
    """
    Visualizes and saves maps of different pixel-wise error metrics.

    Parameters:
        mae_pixel (np.ndarray): Mean Absolute Error map
        rmse_pixel (np.ndarray): Root Mean Squared Error map
        medae_pixel (np.ndarray): Median Absolute Error map
        std_pixel (np.ndarray): Standard Deviation of Error map
        output_path (str): Output image path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [
        ("MAE (Mean Absolute Error)", mae_pixel, "inferno"),
        ("RMSE (Root Mean Squared Error)", rmse_pixel, "magma"),
        ("MedAE (Median Absolute Error)", medae_pixel, "plasma"),
        ("Global Std Dev of Error", std_pixel, "viridis")
    ]

    vmax_common = max(np.max(mae_pixel), np.max(rmse_pixel), np.max(medae_pixel))

    for ax, (title, data, cmap) in zip(axes.flat, metrics):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax_common if "STD" not in title else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Error Value", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=600)
    plt.close()

def generate_samples_from_shp(X_train, y_train, y_pred, shapefile_path, num_images=41):
    gdf = gpd.read_file(shapefile_path)
    x_coords, y_coords = gdf.geometry.x, gdf.geometry.y
    ids = gdf['id'].tolist()

    with rasterio.open("D:/depths/TIF/DEPTHS/depth_0.010_0.055.tiff") as src:
        rows, cols = zip(*[rowcol(src.transform, x, y) for x, y in zip(x_coords, y_coords)])

    point_info = list(zip(rows, cols, ids))  # row, col, id
    sampled_indices = np.random.choice(range(X_train.shape[0]), size=num_images, replace=False)
    
    data = []
    for img_idx in sampled_indices:
        for row, col, pid in point_info:
            try:
                x = X_train[img_idx, row, col, 1]
                y = X_train[img_idx, row, col, 2]
                n = X_train[img_idx, row, col, 0]
                depth = y_train[img_idx, row, col, 0]
                depth_pred = y_pred[img_idx, row, col, 0]
                data.append((row, col, pid, x, y, n, depth,depth_pred))
            except IndexError:
                continue

    return data, sampled_indices, point_info