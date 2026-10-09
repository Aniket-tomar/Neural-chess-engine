import os
import argparse
import numpy as np
import tensorflow as tf
from replay_buffer import AlphaZeroReplayBuffer
from network import create_alphazero_model

def alphazero_loss(y_true_policy, y_pred_policy, y_true_value, y_pred_value):
    # Categorical Cross-Entropy for Policy
    # Add epsilon to prevent log(0)
    eps = 1e-8
    p_pred = tf.clip_by_value(y_pred_policy, eps, 1.0 - eps)
    policy_loss = -tf.reduce_sum(y_true_policy * tf.math.log(p_pred), axis=-1)
    policy_loss = tf.reduce_mean(policy_loss)

    # Mean Squared Error for Value
    value_loss = tf.reduce_mean(tf.square(y_true_value - y_pred_value))

    # Total loss
    return policy_loss + value_loss, policy_loss, value_loss

def train(
    dataset_glob="dataset/*.npz",
    model_path="alphazero_random_weights.keras",
    output_model="alphazero_v002.keras",
    batch_size=64,
    steps_per_epoch=200,
    epochs=5,
    lr=0.001,
    buffer_capacity=50000
):
    print("--- Initializing AlphaZero Training Pipeline ---")
    
    # 1. Populate Replay Buffer
    buffer = AlphaZeroReplayBuffer(capacity=buffer_capacity, version="v0.0.2")
    buffer.load_npz_files(dataset_glob)
    buffer.print_stats()

    if buffer.size < batch_size:
        raise ValueError(f"Buffer has {buffer.size} samples, need at least {batch_size} to train.")

    # 2. Load or Build Model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        print(f"Creating fresh AlphaZero model...")
        model = create_alphazero_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 3. Training Step
    @tf.function
    def train_step(states, target_policies, target_values):
        with tf.GradientTape() as tape:
            pred_policies, pred_values = model(states, training=True)
            total_loss, pol_loss, val_loss = alphazero_loss(
                target_policies, pred_policies,
                target_values, pred_values
            )
            # Add L2 regularization losses if defined in layers
            reg_loss = tf.reduce_sum(model.losses) if model.losses else 0.0
            loss = total_loss + reg_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, pol_loss, val_loss

    # 4. Training Loop
    print(f"\nStarting {epochs} epochs ({steps_per_epoch} steps/epoch, batch_size={batch_size})...")
    for epoch in range(1, epochs + 1):
        epoch_total, epoch_pol, epoch_val = [], [], []

        for step in range(steps_per_epoch):
            s_batch, p_batch, v_batch = buffer.sample(batch_size=batch_size)
            
            t_loss, p_loss, v_loss = train_step(
                tf.convert_to_tensor(s_batch),
                tf.convert_to_tensor(p_batch),
                tf.convert_to_tensor(v_batch)
            )

            epoch_total.append(t_loss.numpy())
            epoch_pol.append(p_loss.numpy())
            epoch_val.append(v_loss.numpy())

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Total Loss: {np.mean(epoch_total):.4f} | "
            f"Policy Loss: {np.mean(epoch_pol):.4f} | "
            f"Value Loss: {np.mean(epoch_val):.4f}"
        )

    # 5. Save Updated Model
    model.save(output_model)
    print(f"\nTraining complete. Model weights saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Model Trainer")
    parser.add_argument("--data", type=str, default="dataset/*.npz", help="Path glob to .npz batch files")
    parser.add_argument("--model", type=str, default="alphazero_random_weights.keras", help="Base model weights")
    parser.add_argument("--out", type=str, default="alphazero_v002.keras", help="Destination file for trained model")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--steps", type=int, default=150, help="Steps per epoch")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train(
        dataset_glob=args.data,
        model_path=args.model,
        output_model=args.out,
        batch_size=args.batch,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        lr=args.lr
    )