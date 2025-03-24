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
