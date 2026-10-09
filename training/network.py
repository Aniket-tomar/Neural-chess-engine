import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

def residual_block(x, filters: int, l2_reg: float = 1e-4):
    shortcut = x
    y = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(y)
    y = layers.BatchNormalization()(y)
    out = layers.add([shortcut, y])
    return layers.ReLU()(out)

def build_configurable_alphazero_model(
    input_shape=(18, 8, 8),
    num_res_blocks: int = 6,
    num_filters: int = 128,
    action_space_dim: int = 4672,
    l2_reg: float = 1e-4
) -> Model:
    """Builds a parametric AlphaZero network with named inputs/outputs for ONNX export."""
    # Data arrives as [batch, 18, 8, 8] (channels-first representation from the C++ bitboard)
    inputs = layers.Input(shape=input_shape, name="input_state")
    
    # Permute to [batch, 8, 8, 18] for GPU-friendly Conv2D operations
    x = layers.Permute((2, 3, 1))(inputs)

    # Initial Convolutional Block
    x = layers.Conv2D(
        num_filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual Tower
    for _ in range(num_res_blocks):
        x = residual_block(x, filters=num_filters, l2_reg=l2_reg)

    # Policy Head
    p = layers.Conv2D(32, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Flatten()(p)
    policy_out = layers.Dense(
        action_space_dim, 
        activation="softmax", 
        name="policy_head",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(p)

    # Value Head
    v = layers.Conv2D(1, 1, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(v)
    value_out = layers.Dense(
        1, 
        activation="tanh", 
        name="value_head",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(v)

    return Model(inputs=inputs, outputs=[policy_out, value_out], name="AlphaZeroChess")