# ============================
# Import required libraries
# ============================
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from flax.training import train_state
from pytensor.graph.op import Op
import pytensor.tensor as pt

import arviz as az
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op

# ============================
# Compute correlation between roughness and depth
# ============================
def correlation_weigths(X_train, y_train):
    """
    Calculates the pixel-wise correlation between roughness (n) and depth 
    across all images and returns a correlation heatmap.
    """
    num_images, height, width, _ = X_train.shape
    correlation_matrix = np.full((height, width), np.nan)

    for row in range(height):
        for col in range(width):
            roughness_values = X_train[:, row, col, 0]
            depth_values = y_train[:, row, col, 0]
            if np.std(roughness_values) > 0 and np.std(depth_values) > 0:
                correlation_matrix[row, col] = np.corrcoef(roughness_values, depth_values)[0, 1]

    return correlation_matrix

# ============================
# Create weight map based on correlation
# ============================
def create_weight_map(correlation_map, min_weight=0.1, max_weight=1.0):
    """
    Creates a weight map where low-correlation values are assigned less weight.
    """
    weight_map = min_weight + (max_weight - min_weight) * jnp.abs(correlation_map)
    return weight_map

# ============================
# Expand correlation map to match batch size
# ============================
def get_correlation_map_batch(correlation_map_2d, batch_size):
    """
    Expands a 2D correlation map to match the batch size for training.
    """
    corr = jnp.expand_dims(correlation_map_2d, axis=(0, -1))  # (1, H, W, 1)
    return jnp.repeat(corr, batch_size, axis=0)               # (batch, H, W, 1)

# ============================
# Define convolutional neural network model with correlation attenuation
# ============================
class CNNModel(nn.Module):
    use_correlation: bool = True  # Flag to control use of correlation mask

    @nn.compact
    def __call__(self, x, correlation_map=None):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        x = nn.Dense(387 * 450)(x)  # Output dimension
        x = x.reshape((x.shape[0], 387, 450, 1))

        if self.use_correlation and correlation_map is not None:
            correlation_map_resized = jax.image.resize(correlation_map, shape=x.shape, method='bilinear')
            x = x * correlation_map_resized

        return x

# ============================
# Define training state
# ============================
class TrainState(train_state.TrainState):
    pass

# ============================
# Define weighted loss function based on correlation
# ============================
def weighted_mse_loss(y_true, y_pred, weight_map):
    """
    Computes the mean squared error loss weighted by a correlation-based weight map.
    """
    return jnp.mean(jnp.square(y_true - y_pred) * weight_map)

def loss_fn(params, apply_fn, x, y, correlation_map):
    """
    Applies the model to the inputs and computes the weighted MSE loss.
    """
    predictions = apply_fn(params, x, correlation_map)
    weight_map = create_weight_map(correlation_map)
    return weighted_mse_loss(y, predictions, weight_map)

# ============================
# Define training and validation steps
# ============================
@jax.jit
def train_step(state, batch_x, batch_y, correlation_map):
    """
    Performs one training step (forward + backward pass and parameter update).
    """
    loss, grads = jax.value_and_grad(loss_fn)(state.params, state.apply_fn, batch_x, batch_y, correlation_map)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def validation_step(state, batch_x, batch_y, correlation_map):
    """
    Performs one validation step (forward pass only).
    """
    loss = loss_fn(state.params, state.apply_fn, batch_x, batch_y, correlation_map)
    return loss

# ===============================
# JAX-based CNN Prediction Function
# ===============================
@jax.jit
def cnn_predict(x_cov, model_state, n1, n2, correlation_map):
    """
    Performs a CNN-based prediction using a Flax/JAX model.

    Args:
        x_cov (np.ndarray or jnp.ndarray): Input image of shape (height, width, channels).
        model_state (TrainState): Flax model state with `apply_fn` and `params`.
        n1 (float): Value assigned if pixel ≤ 0.05.
        n2 (float): Value assigned if pixel > 0.05.
        correlation_map (jnp.ndarray): Correlation map used during inference.

    Returns:
        jnp.ndarray: Predicted depth image (squeezed).
    """
    x_cov = jnp.array(x_cov)
    image = x_cov[:, :, 0]

    # Replace values in the first channel based on threshold
    image_new = jnp.where(image > 0.05, n2, n1)

    # Update the first channel with thresholded values
    x_cov_new = x_cov.at[:, :, 0].set(image_new)

    # Add batch dimension
    input_image = jnp.expand_dims(x_cov_new, axis=0)

    # Prepare correlation map batch (assumed to be handled externally)
    batch_size = input_image.shape[0]
    correlation_map_batch = get_correlation_map_batch(correlation_map, batch_size)

    # Perform prediction
    prediction = model_state.apply_fn(model_state.params, input_image, correlation_map_batch)

    return jnp.squeeze(prediction)

# ===============================
# Custom PyTensor Operation for CNN Prediction
# ===============================
class CNNPredictOp(Op):
    """
    PyTensor custom Op for CNN-based depth prediction using JAX.

    Inputs:
        - x_cov (pt.dtensor3): 3D tensor representing the input image.
        - n1 (pt.dscalar): Scalar used for pixel values ≤ 0.05.
        - n2 (pt.dscalar): Scalar used for pixel values > 0.05.

    Output:
        - pt.dtensor3: Predicted depth image tensor.
    """
    itypes = [pt.dtensor3, pt.dscalar, pt.dscalar]
    otypes = [pt.dtensor3]

    def __init__(self, model_state, correlation_map):
        self.model_state = model_state
        self.correlation_map = correlation_map

    def perform(self, node, inputs, outputs):
        x_cov, n1, n2 = inputs
        # Convert scalar inputs to float
        n1 = float(n1)
        n2 = float(n2)

        # Run CNN prediction
        result = cnn_predict(x_cov, self.model_state, n1, n2, self.correlation_map)

        # Output result as NumPy float64 array
        outputs[0][0] = np.array(result, dtype=np.float64)

def build_model(x_cov, state, correlation_map, true_image, coords_y, coords_x, pixel_x, pixel_y, observed_values, use_discrepancy=True):
    with pm.Model() as model:
        # Priors
        n1 = pm.Uniform("n1", lower=0.01, upper=0.05)
        n2 = pm.Uniform("n2", lower=0.05, upper=0.085)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Input tensor
        x_cov_t = pt.as_tensor_variable(x_cov)

        # CNN emulator (η)
        cnn_op = CNNPredictOp(model_state=state, correlation_map=correlation_map)
        depth_mean = cnn_op(x_cov_t, n1, n2)
        depth_full = pt.reshape(depth_mean, true_image.shape)
        #S_xtheta = depth_full[coords_y, coords_x]  # Emulator output η(x, θ)
        S_xtheta = depth_full[pixel_x, pixel_y]  # Emulator output η(x, θ)

        # Likelihood setup
        n_obs = observed_values.shape[0]

        if use_discrepancy:
            # GP params
            sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.0)
            beta_x = pm.Normal("beta_x", mu=0, sigma=1)
            beta_y = pm.Normal("beta_y", mu=0, sigma=1)
            omega_x = pm.Gamma("omega_x", alpha=2, beta=1)
            omega_y = pm.Gamma("omega_y", alpha=2, beta=1)

            # Coordenadas observadas
            x1 = pt.as_tensor_variable(coords_x).reshape((-1, 1))
            x2 = pt.as_tensor_variable(coords_x).reshape((1, -1))
            y1 = pt.as_tensor_variable(coords_y).reshape((-1, 1))
            y2 = pt.as_tensor_variable(coords_y).reshape((1, -1))

            sq_dist = omega_x * pt.sqr(x1 - x2) + omega_y * pt.sqr(y1 - y2)
            K = sigma_delta**2 * pt.exp(-sq_dist)

            # GP mean
            mean_gp = beta_x * coords_x + beta_y * coords_y

            mu_total = S_xtheta + mean_gp
            Sigma = K + pt.eye(n_obs) * sigma**2
        else:
            mu_total = S_xtheta
            Sigma = pt.eye(n_obs) * sigma**2

        # Likelihood
        pm.MvNormal(
            "observed",
            mu=mu_total,
            cov=Sigma,
            observed=observed_values
        )

    return model