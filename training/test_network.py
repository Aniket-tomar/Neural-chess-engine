import numpy as np
import tensorflow as tf
from network import create_alphazero_model

def test_network():
    print("Testing AlphaZero Neural Network Architecture...")
    
    # 1. Initialization and Parameter Count
    model = create_alphazero_model(
        input_shape=(18, 8, 8), 
        action_space_size=4672, 
        filters=64, 
        res_blocks=4
    )
    
    model.summary()
    param_count = model.count_params()
    print(f"\nTotal Parameters: {param_count:,}")
    assert param_count < 1_000_000, "Model is too large for browser execution!"
    
    # 2. Forward Pass & Output Shapes
    print("\nRunning Forward Pass with random input...")
    # Generate a dummy batch of 1
    dummy_input = np.random.rand(1, 18, 8, 8).astype(np.float32)
    
    # training=False is required to lock BatchNorm layers for deterministic inference
    policy, value = model(dummy_input, training=False)
    
    print(f"Policy Shape: {policy.shape}")
    print(f"Value Shape:  {value.shape}")
    
    assert policy.shape == (1, 4672), f"Expected Policy (1, 4672), got {policy.shape}"
    assert value.shape == (1, 1), f"Expected Value (1, 1), got {value.shape}"
    
    # 3. Value Range Constraint
    val = value.numpy()[0][0]
    print(f"Value Output: {val:.4f}")
    assert -1.0 <= val <= 1.0, f"Value {val} strictly outside [-1, +1] tanh bounds."
    
    # 4. Policy Softmax Constraint
    policy_sum = np.sum(policy.numpy()[0])
    print(f"Policy Sum:   {policy_sum:.4f}")
    assert np.isclose(policy_sum, 1.0, atol=1e-5), "Policy does not sum to 1.0."

    # 5. Deterministic Inference Check
    print("Checking Deterministic Inference...")
    policy2, value2 = model(dummy_input, training=False)
    assert np.allclose(policy.numpy(), policy2.numpy()), "Policy output varies between identical passes!"
    assert np.allclose(value.numpy(), value2.numpy()), "Value output varies between identical passes!"
    
    print("\nAll architecture constraints and tests PASSED.")

if __name__ == "__main__":
    test_network()