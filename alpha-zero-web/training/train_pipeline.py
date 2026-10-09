import os
import re
import json
import time
import argparse
import numpy as np
import tensorflow as tf
import tf2onnx

from replay_buffer import AlphaZeroReplayBuffer
from network import build_configurable_alphazero_model


def alphazero_loss(y_true_pi, y_pred_pi, y_true_z, y_pred_z, lambda_val: float):
    # Categorical Cross-Entropy: -Σ π(a) log P(a)
    eps = 1e-8
    p_pred = tf.clip_by_value(y_pred_pi, eps, 1.0 - eps)
    loss_policy = -tf.reduce_sum(y_true_pi * tf.math.log(p_pred), axis=-1)
    loss_policy = tf.reduce_mean(loss_policy)

    # Value Loss (MSE): (z - V(s))²
    loss_value = tf.reduce_mean(tf.square(y_true_z - y_pred_z))

    # Total loss: L = L_policy + λ * L_value
    total_loss = loss_policy + (lambda_val * loss_value)
    return total_loss, loss_policy, loss_value


def export_to_onnx(model: tf.keras.Model, export_path: str):
    """Exports trained Keras model to ONNX format matching frontend feeds."""
    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    spec = (tf.TensorSpec((None, 18, 8, 8), tf.float32, name="input_state"),)
    print(f"\n[ONNX] Converting model to {export_path} (opset 15)...")
    
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=15,
        output_path=export_path
    )
    print(f"[ONNX] Export successfully written to: {export_path}")


def get_latest_checkpoint(checkpoint_dir: str):
    """Finds the highest-version checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
    checkpoints = []
    for f in files:
        m = re.match(r"^ckpt_v(\d+)\.keras$", f)
        if m:
            checkpoints.append((int(m.group(1)), os.path.join(checkpoint_dir, f)))
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1], checkpoints[-1][0]


def run_pipeline(args):
    # Setup Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.log_dir, run_id)
    summary_writer = tf.summary.create_file_writer(log_dir)

    print("=" * 60)
    print("           ALPHAZERO TRAINING PIPELINE")
    print("=" * 60)
    print(f" Logs & TensorBoard: {log_dir}")
    print(f" Checkpoints:        {args.checkpoint_dir}")
    print(f" ResNet Config:      {args.blocks} blocks, {args.filters} filters")
    print(f" Training Config:    batch={args.batch_size}, lr={args.lr}, λ={args.val_lambda}")

    # 1. Load Data into Buffer
    buffer = AlphaZeroReplayBuffer(capacity=args.buffer_capacity)
    buffer.load_npz_files(args.data)
    buffer.print_stats()

    if buffer.size < args.batch_size:
        raise ValueError(f"Insufficient samples ({buffer.size}) for mini-batch size {args.batch_size}.")

    # 2. Train / Validation Split
    indices = np.arange(buffer.size)
    np.random.shuffle(indices)
    val_count = int(buffer.size * args.val_split)
    train_count = buffer.size - val_count

    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_states, train_pi, train_z = buffer.states[train_idx], buffer.policies[train_idx], buffer.values[train_idx]
    val_states, val_pi, val_z = buffer.states[val_idx], buffer.policies[val_idx], buffer.values[val_idx]

    print(f" Partitioned Samples: {train_count} Train | {val_count} Validation")

    # 3. Model Resolution & Resume Logic
    start_epoch = 1
    model_version = 1
    model = None

    if args.resume:
        latest_ckpt, ckpt_ver = get_latest_checkpoint(args.checkpoint_dir)
        if latest_ckpt:
            print(f"[Resume] Found checkpoint {latest_ckpt} (Version {ckpt_ver}). Loading weights...")
            model = tf.keras.models.load_model(latest_ckpt, compile=False)
            model_version = ckpt_ver + 1
            meta_path = os.path.join(args.checkpoint_dir, f"ckpt_v{ckpt_ver:03d}_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    start_epoch = meta.get("epoch", 0) + 1
            print(f"[Resume] Resuming training from Epoch {start_epoch} as Version v{model_version}...")

    if model is None:
        if args.base_model and os.path.exists(args.base_model):
            print(f"[Init] Loading baseline model: {args.base_model}")
            model = tf.keras.models.load_model(args.base_model, compile=False)
        else:
            print(f"[Init] Building fresh model ({args.blocks} blocks, {args.filters} filters)...")
            model = build_configurable_alphazero_model(
                num_res_blocks=args.blocks,
                num_filters=args.filters,
                action_space_dim=4672
            )

    # 4. Optimizer & Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 5. Compiled Execution Steps
    @tf.function
    def train_step(states, target_pi, target_z):
        with tf.GradientTape() as tape:
            pred_pi, pred_z = model(states, training=True)
            tot_loss, pol_loss, val_loss = alphazero_loss(
                target_pi, pred_pi, target_z, pred_z, args.val_lambda
            )
            reg_loss = tf.reduce_sum(model.losses) if model.losses else 0.0
            total_loss_with_reg = tot_loss + reg_loss

        grads = tape.gradient(total_loss_with_reg, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss_with_reg, pol_loss, val_loss

    @tf.function
    def val_step(states, target_pi, target_z):
        pred_pi, pred_z = model(states, training=False)
        tot_loss, pol_loss, val_loss = alphazero_loss(
            target_pi, pred_pi, target_z, pred_z, args.val_lambda
        )
        return tot_loss, pol_loss, val_loss

    # 6. Training Loop
    steps_per_epoch = max(1, train_count // args.batch_size)
    global_step = (start_epoch - 1) * steps_per_epoch

    print(f"\nStarting optimization: {args.epochs} Epochs ({steps_per_epoch} steps/epoch)...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        # Shuffle training set for each epoch
        perm = np.random.permutation(train_count)
        epoch_t_loss, epoch_p_loss, epoch_v_loss = [], [], []

        for step in range(steps_per_epoch):
            batch_slice = perm[step * args.batch_size : (step + 1) * args.batch_size]
            b_s = tf.convert_to_tensor(train_states[batch_slice])
            b_p = tf.convert_to_tensor(train_pi[batch_slice])
            b_z = tf.convert_to_tensor(train_z[batch_slice])

            tot, pol, val = train_step(b_s, b_p, b_z)
            epoch_t_loss.append(tot.numpy())
            epoch_p_loss.append(pol.numpy())
            epoch_v_loss.append(val.numpy())
            global_step += 1

        # Validation Pass
        val_t_loss, val_p_loss, val_v_loss = [], [], []
        val_steps = max(1, val_count // args.batch_size) if val_count > 0 else 0
        for v_step in range(val_steps):
            vb_s = tf.convert_to_tensor(val_states[v_step * args.batch_size : (v_step + 1) * args.batch_size])
            vb_p = tf.convert_to_tensor(val_pi[v_step * args.batch_size : (v_step + 1) * args.batch_size])
            vb_z = tf.convert_to_tensor(val_z[v_step * args.batch_size : (v_step + 1) * args.batch_size])

            v_tot, v_pol, v_val = val_step(vb_s, vb_p, vb_z)
            val_t_loss.append(v_tot.numpy())
            val_p_loss.append(v_pol.numpy())
            val_v_loss.append(v_val.numpy())

        # Metrics aggregation
        tr_tot = float(np.mean(epoch_t_loss))
        tr_pol = float(np.mean(epoch_p_loss))
        tr_val = float(np.mean(epoch_v_loss))
        v_tot_avg = float(np.mean(val_t_loss)) if val_t_loss else 0.0
        v_pol_avg = float(np.mean(val_p_loss)) if val_p_loss else 0.0
        v_val_avg = float(np.mean(val_v_loss)) if val_v_loss else 0.0

        current_lr = float(optimizer.learning_rate(global_step).numpy()) if callable(optimizer.learning_rate) else float(optimizer.learning_rate.numpy())

        # TensorBoard Logging
        with summary_writer.as_default():
            tf.summary.scalar("train/total_loss", tr_tot, step=epoch)
            tf.summary.scalar("train/policy_loss", tr_pol, step=epoch)
            tf.summary.scalar("train/value_loss", tr_val, step=epoch)
            tf.summary.scalar("val/total_loss", v_tot_avg, step=epoch)
            tf.summary.scalar("val/policy_loss", v_pol_avg, step=epoch)
            tf.summary.scalar("val/value_loss", v_val_avg, step=epoch)
            tf.summary.scalar("train/lr", current_lr, step=epoch)
        summary_writer.flush()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d} ({elapsed:.1f}s) | "
            f"Train [Tot: {tr_tot:.4f}, Pol: {tr_pol:.4f}, Val: {tr_val:.4f}] | "
            f"Val [Tot: {v_tot_avg:.4f}, Pol: {v_pol_avg:.4f}, Val: {v_val_avg:.4f}] | "
            f"LR: {current_lr:.6f}"
        )

        # 7. Checkpointing
        if epoch % args.save_every == 0 or epoch == (start_epoch + args.epochs - 1):
            ckpt_name = f"ckpt_v{model_version:03d}.keras"
            save_path = os.path.join(args.checkpoint_dir, ckpt_name)
            model.save(save_path)

            meta = {
                "version": f"v{model_version:03d}",
                "epoch": epoch,
                "global_step": global_step,
                "filters": args.filters,
                "blocks": args.blocks,
                "train_loss": tr_tot,
                "val_loss": v_tot_avg,
                "timestamp": time.time()
            }
            meta_path = os.path.join(args.checkpoint_dir, f"ckpt_v{model_version:03d}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[Checkpoint] Version {ckpt_name} saved at {save_path}")

    # 8. Export Final Artifact to ONNX
    export_to_onnx(model, args.onnx_out)
    print("\nTraining and conversion pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Neural Network Training Pipeline")
    
    # Dataset & Paths
    parser.add_argument("--data", type=str, default="dataset/*.npz", help="Glob pattern for self-play data batches")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to store versioned checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--onnx_out", type=str, default="model_v002.onnx", help="Path for exported ONNX model")
    parser.add_argument("--base_model", type=str, default=None, help="Path to initial weights (.keras)")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in checkpoint_dir")
    
    # Architecture Hyperparameters
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--filters", type=int, default=64, help="Number of convolutional filters")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--decay_steps", type=int, default=1000, help="LR exponential decay steps")
    parser.add_argument("--decay_rate", type=float, default=0.96, help="LR decay rate")
    parser.add_argument("--val_lambda", type=float, default=1.0, help="Weight multiplier λ for Value Loss")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation dataset fraction (0.0 to 1.0)")
    parser.add_argument("--save_every", type=int, default=5, help="Checkpoint save interval (epochs)")
    parser.add_argument("--buffer_capacity", type=int, default=50000, help="Replay buffer maximum capacity")

    args = parser.parse_args()
    run_pipeline(args)