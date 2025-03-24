class CNNPredictOp(Op):
    """
    Operación personalizada de PyTensor para la predicción de la CNN.
    """
    itypes = [pt.dtensor3, pt.dscalar, pt.dscalar]  # Tipos de entrada
    otypes = [pt.dtensor3]  # Tipo de salida

    def __init__(self, model_state, correlation_map):
        self.model_state = model_state
        self.correlation_map = correlation_map

    def perform(self, node, inputs, outputs):
        x_cov, n1, n2 = inputs
        # Convertir a escalares float
        n1 = float(n1)
        n2 = float(n2)
        # Ejecutar la predicción de la CNN
        result = cnn_predict(x_cov, self.model_state, n1, n2, self.correlation_map)
        outputs[0][0] = np.array(result, dtype=np.float64)  # Convertir salida a NumPy float64

# ✅ Modificar `cnn_predict` para evitar `eval()`
@jax.jit
def cnn_predict(x_cov, model_state, n1, n2,correlation_map):
    """
    Realiza una predicción con la CNN de Flax/JAX.

    Args:
        x_cov (np.ndarray or jnp.ndarray): Imagen de entrada con shape (height, width, channels).
        model_state (TrainState): Estado del modelo Flax con `apply_fn` y `params`.
        n1 (float): Valor asignado si el pixel es <= 0.05.
        n2 (float): Valor asignado si el pixel es > 0.05.

    Returns:
        jnp.ndarray: Imagen de profundidad predicha.
    """
    x_cov = jnp.array(x_cov)  # Convertir a JAX array
    image = x_cov[:, :, 0]

    # ✅ No usar `.eval()`, solo pasar los valores directamente
    image_new = jnp.where(image > 0.05, n2, n1)

    # Crear copia con el nuevo primer canal
    x_cov_new = x_cov.at[:, :, 0].set(image_new)

    # Expandir dimensiones para batch
    input_image = jnp.expand_dims(x_cov_new, axis=0)

    # Obtener el batch size y la shape de salida esperada
    batch_size = image.shape[0]
    # Expandir y redimensionar el mapa de correlación para predicción
    correlation_map_batch = get_correlation_map_batch(correlation_map, batch_size)

    # Realizar predicción
    prediction = model_state.apply_fn(model_state.params, input_image,correlation_map_batch)

    return jnp.squeeze(prediction)