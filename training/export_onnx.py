import os
import tf2onnx
import tensorflow as tf

def convert_keras_to_onnx(keras_path: str, onnx_path: str):
    print(f"Loading {keras_path}...")
    model = tf.keras.models.load_model(keras_path, compile=False)

    spec = (tf.TensorSpec((None, 18, 8, 8), tf.float32, name="input_state"),)
    
    print(f"Converting to ONNX format (opset 15)...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=15, 
        output_path=onnx_path
    )
    print(f"ONNX model saved successfully to: {onnx_path}")

if __name__ == "__main__":
    # Ensure tf2onnx is installed: pip install tf2onnx
    convert_keras_to_onnx("alphazero_v002.keras", "model_v002.onnx")