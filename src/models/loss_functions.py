def correlation_weigths(X_train, y_train):
    """
    Calcula la correlación entre la rugosidad (n) y la profundidad (depth) para cada píxel y genera un mapa de calor.
    """
    # Obtener dimensiones (número de imágenes, altura, anchura)
    num_images, height, width, _ = X_train.shape

    # Matriz de correlación para cada píxel
    correlation_matrix = np.full((height, width), np.nan)  # Inicializar con NaN para evitar errores

    for row in range(height):
        for col in range(width):
            # Extraer valores de rugosidad (n) y profundidad (depth) en el mismo píxel a lo largo de todas las imágenes
            roughness_values = X_train[:, row, col, 0]
            depth_values = y_train[:, row, col, 0]

            # Verificar que haya variabilidad en los datos antes de calcular correlación
            if np.std(roughness_values) > 0 and np.std(depth_values) > 0:
                correlation_matrix[row, col] = np.corrcoef(roughness_values, depth_values)[0, 1]

    return correlation_matrix  # Devuelve la matriz de correlación por si la necesitas para otros cálculos

# ============================
# Crear el mapa de pesos basado en la correlación
# ============================
def create_weight_map(correlation_map, min_weight=0.1, max_weight=1.0):
    """
    Crea un mapa de pesos donde los valores con menor correlación tienen menor peso.
    """
    weight_map = min_weight + (max_weight - min_weight) * jnp.abs(correlation_map)  # Normalización
    return weight_map

# ============================
# Definir la función de pérdida ponderada basada en correlación
# ============================
def weighted_mse_loss(y_true, y_pred, weight_map):
    return jnp.mean(jnp.square(y_true - y_pred) * weight_map)

def loss_fn(params, apply_fn, x, y, correlation_map):
    predictions = apply_fn(params, x, correlation_map)
    weight_map = create_weight_map(correlation_map)
    return weighted_mse_loss(y, predictions, weight_map)