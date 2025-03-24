# Redes neuronales convolucionales (CNN)
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pickle
from flax.training import train_state
# ============================
# CNN con atenuación por correlación
# ============================
class CNNModel(nn.Module):
    use_correlation: bool = True  # Flag para controlar el uso de la máscara

    @nn.compact
    def __call__(self, x, correlation_map=None):
        # Primera capa convolucional
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Segunda capa convolucional
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Tercera capa convolucional sin dilatación
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Cuarta capa convolucional sin dilatación
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Flatten para fully connected layers
        x = x.reshape((x.shape[0], -1))  # Aplanar excepto batch size

        # Fully connected layers antes de la salida
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Restaurar dimensión espacial
        x = nn.Dense(387 * 450)(x)  # Ajustamos a la salida esperada
        x = x.reshape((x.shape[0], 387, 450, 1))  # Salida con dimensión espacial

        # Aplicar mapa de correlación si está habilitado
        if self.use_correlation and correlation_map is not None:
            correlation_map_resized = jax.image.resize(correlation_map, shape=x.shape, method='bilinear')
            x = x * correlation_map_resized

        return x