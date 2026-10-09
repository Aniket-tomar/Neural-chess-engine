# AlphaZero-Web Chess Engine

A browser-based, AlphaZero-inspired chess engine. Chess analysis and AI move generation happen entirely on the client side without relying on a remote backend server.

## Project Structure
* `frontend/`: Next.js / React user interface.
* `engine/`: C++20 chess engine (MCTS, legal move generation).
* `training/`: Python offline training pipeline (TensorFlow/Keras).
* `models/`: Exported ONNX models for browser inference.
* `tests/`: Integration and unit tests.
* `docs/`: Architecture and design documentation.