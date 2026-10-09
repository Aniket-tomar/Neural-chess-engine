# Architecture & Compilation Strategy

## Core Architecture
The application runs entirely in the browser using the following stack:
1. **Frontend UI:** Next.js and React handle the chessboard rendering and user interaction.
2. **Web Worker:** Heavy computation runs in a background thread to keep the UI responsive.
3. **WebAssembly (Wasm) Engine:** The C++20 chess engine handles state, legal move generation, and Monte Carlo Tree Search (MCTS).
4. **ONNX Runtime Web:** Executes the Policy + Value neural network evaluation locally.

## C++ Compilation Strategy

### Phase A: Native Compilation (Current)
During early development, the C++ engine is compiled natively (e.g., as a `.exe` on Windows). 
* **Why:** Enables rapid testing, easy debugging, and memory checking without browser overhead.
* **How:** We use a simple `g++` command directly. No CMake or complex build systems are required for this phase to ensure maximum compatibility and simplicity on Windows.

### Phase B: Emscripten Compilation (Future)
Once the core engine logic (move generation, MCTS) is verified natively, the build target will switch to WebAssembly using Emscripten.
* **Why:** To run the C++ code at near-native speed inside the browser's Web Worker.
* **How:** `g++` commands will be replaced by `emcc` (Emscripten C++ Compiler). Emscripten will compile the C++ source into a `.wasm` binary and generate a JavaScript wrapper to handle communication between the Web Worker and the Wasm module.