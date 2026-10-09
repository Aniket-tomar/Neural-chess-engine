import numpy as np
import tensorflow as tf
from network import create_alphazero_model

def compile_model(model):
    """Compiles the model with AlphaZero's dual loss functions."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy_head': 'categorical_crossentropy', 
            'value_head': 'mean_squared_error'
        },
        loss_weights={
            'policy_head': 1.0, 
            'value_head': 1.0
        },
        metrics={
            'policy_head': 'accuracy',
            'value_head': 'mae'
        }
    )
    return model

def generate_synthetic_data(num_samples=1000, action_space_size=4672):
    """
    Generates fully random tensors to simulate a replay buffer of chess positions.
    """
    print(f"Generating {num_samples} random positions for training...")
    
    # 1. Random Input States (Shape: [N, 18, 8, 8])
    # Using 0s and 1s to simulate piece/flag planes
    X = np.random.randint(0, 2, size=(num_samples, 18, 8, 8)).astype(np.float32)
    
    # 2. Random Policy Targets (Shape: [N, 4672])
    # Simulating a one-hot target (e.g., the MCTS decided move X is 100% the best)
    Y_policy = np.zeros((num_samples, action_space_size), dtype=np.float32)
    random_moves = np.random.randint(0, action_space_size, size=num_samples)
    Y_policy[np.arange(num_samples), random_moves] = 1.0
    
    # 3. Random Value Targets (Shape: [N, 1])
    # Simulating game outcomes between -1.0 (Black wins) and +1.0 (White wins)
    Y_value = np.random.uniform(-1.0, 1.0, size=(num_samples, 1)).astype(np.float32)
    
    return X, {'policy_head': Y_policy, 'value_head': Y_value}

def main():
    print("Initializing AlphaZero Training Pipeline...")
    
    # Instantiate and compile the un-trained model
    model = create_alphazero_model()
    model = compile_model(model)
    
    # Generate a dummy dataset
    X_train, Y_train = generate_synthetic_data(num_samples=2048)
    
    print("\nStarting Training Loop...")
    
    # Execute the forward pass, calculate loss, and update weights
    history = model.fit(
        x=X_train,
        y=Y_train,
        batch_size=64,
        epochs=5,
        validation_split=0.2, # Hold back 20% of random data to monitor 'overfitting'
        verbose=1
    )
    
    print("\nTraining complete. Saving weights...")
    model.save("alphazero_random_weights.keras")
    print("Model saved to alphazero_random_weights.keras")

if __name__ == "__main__":
    main()