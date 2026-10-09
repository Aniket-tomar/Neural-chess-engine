# ♟️ AlphaZero-Web — Neural Chess Engine

A **fully client-side**, AlphaZero-inspired chess engine that runs entirely in the browser. No remote server is involved — move generation, Monte Carlo Tree Search, and neural network inference all execute locally using WebAssembly and ONNX Runtime.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
  - [Runtime Dataflow](#runtime-dataflow)
  - [Neural Network](#neural-network)
  - [Board Encoding](#board-encoding)
  - [Action Space](#action-space)
  - [MCTS Search](#mcts-search)
- [Development Journey](#development-journey)
  - [Phase 1 — C++ Chess Engine](#phase-1--c-chess-engine)
  - [Phase 2 — Neural Network Design](#phase-2--neural-network-design)
  - [Phase 3 — Stockfish Data Bootstrapping](#phase-3--stockfish-data-bootstrapping)
  - [Phase 4 — Self-Play Loop](#phase-4--self-play-loop)
  - [Phase 5 — WebAssembly Compilation](#phase-5--webassembly-compilation)
  - [Phase 6 — Browser Integration](#phase-6--browser-integration)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [How to Run](#how-to-run)
  - [1. Frontend (Next.js UI)](#1-frontend-nextjs-ui)
  - [2. C++ Engine — Native Build (Development & Testing)](#2-c-engine--native-build-development--testing)
  - [3. C++ Engine — WebAssembly Build (Browser Deployment)](#3-c-engine--webassembly-build-browser-deployment)
  - [4. Training Pipeline](#4-training-pipeline)
- [Training Workflow](#training-workflow)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Key Challenges Solved](#key-challenges-solved)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER                                 │
│                                                                 │
│  ┌──────────────┐      ┌──────────────────────────────────────┐ │
│  │   Next.js /  │      │           Web Worker Thread          │ │
│  │   React UI   │◄────►│                                      │ │
│  │              │ msg  │  ┌────────────┐   ┌───────────────┐  │ │
│  │  Chessboard  │      │  │  C++ WASM  │   │  ONNX Runtime │  │ │
│  │  Eval Bar    │      │  │  Engine    │◄─►│  Web (WASM)   │  │ │
│  │  Controls    │      │  │            │   │               │  │ │
│  └──────────────┘      │  │ Chess Rules│   │ Policy+Value  │  │ │
│                        │  │ MCTS       │   │ Neural Net    │  │ │
│                        │  │ Move Gen   │   │ (~2 MB ONNX)  │  │ │
│                        │  └────────────┘   └───────────────┘  │ │
│                        └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE (Python)                              │
│                                                                 │
│  Self-Play Data  ──►  Training Pipeline  ──►  ONNX Export       │
│  (python-chess       (TensorFlow/Keras)      (tf2onnx)          │
│   + MCTS)                                                       │
└─────────────────────────────────────────────────────────────────┘
```

The system is split into two worlds:

| Layer | Role | Language |
|---|---|---|
| **Frontend UI** | Chessboard rendering, user interaction, evaluation display | TypeScript (Next.js / React) |
| **Web Worker** | Offloads all heavy computation to a background thread | TypeScript orchestration |
| **WASM Engine** | Legal move generation, board state, MCTS tree search | C++20 → WebAssembly (Emscripten) |
| **ONNX Runtime Web** | Neural network inference (policy + value heads) | WASM backend via `onnxruntime-web` |
| **Training Pipeline** | Self-play data generation, network training, ONNX export | Python (TensorFlow / Keras) |

---

## How It Works

### Runtime Dataflow

When the user clicks **"Analyze Position"**, the following loop executes inside the Web Worker:

```
1. UI sends ANALYZE command ─────────────► Web Worker
2. Worker calls engine.startAnalysis()     (C++ WASM)
3. MCTS Loop (up to N simulations):
   a. engine.mctsStep()                    → selects leaf node
   b. engine.mctsGetTensor()               → extracts [18,8,8] board tensor
   c. nn.predict(tensor)                   → ONNX Runtime runs inference
   d. engine.mctsProvideEval(policy,value) → resumes backpropagation
4. engine.mctsGetResult()                  → returns best move + eval
5. Worker posts ANALYSIS_COMPLETE ────────► UI renders result
```

The MCTS is implemented as a **cooperative state machine** — it pauses whenever it needs a neural evaluation and resumes once the ONNX result is provided. This lets the async `await` in JavaScript yield to the event loop between simulations, keeping the worker responsive.

### Neural Network

A dual-headed residual network (ResNet) inspired by the original AlphaZero paper:

| Component | Details |
|---|---|
| **Input** | `[18, 8, 8]` tensor (channels-first) |
| **Initial Conv Block** | Conv2D 3×3 → BatchNorm → ReLU |
| **Residual Tower** | 6 residual blocks (Conv → BN → ReLU → Conv → BN → Skip → ReLU) |
| **Policy Head** | Conv2D 1×1 → BN → ReLU → Flatten → Dense(4672) → Softmax |
| **Value Head** | Conv2D 1×1 → BN → ReLU → Flatten → Dense(128) → ReLU → Dense(1) → Tanh |
| **Filters** | 128 per convolutional layer |
| **Output** | Policy: 4672-dim probability vector, Value: scalar ∈ [-1, +1] |

The model is trained in TensorFlow/Keras, exported to ONNX via `tf2onnx`, and loaded in-browser by `onnxruntime-web`.

### Board Encoding

The board state is encoded as **18 binary planes** of size 8×8 (1152 floats total):

| Planes | Feature |
|---|---|
| 0–5 | White pieces (Pawn, Knight, Bishop, Rook, Queen, King) |
| 6–11 | Black pieces (Pawn, Knight, Bishop, Rook, Queen, King) |
| 12 | Side to move (all 1s = White, all 0s = Black) |
| 13–16 | Castling rights (WK, WQ, BK, BQ) — broadcast across 8×8 |
| 17 | En passant target square |

### Action Space

The policy head outputs **4,672 probabilities** representing every geometrically possible move:

- **64 source squares × 73 direction planes = 4,672**
- Planes 0–55: Queen-like moves (8 directions × 7 distances)
- Planes 56–63: Knight jumps (8 L-shaped moves)
- Planes 64–72: Underpromotions (Knight / Bishop / Rook × 3 capture directions)

During MCTS, only legal moves are extracted from the policy vector; illegal indices are ignored.

### MCTS Search

The Monte Carlo Tree Search uses the **PUCT** (Predictor + Upper Confidence bound for Trees) formula:

```
score(a) = Q(a) + c_puct × P(a) × √(N_parent) / (1 + N(a))
```

Where `P(a)` is the neural network's prior probability and `Q(a)` is the mean action value from previous rollouts.

---

## Development Journey

Building this project from scratch involved six distinct phases, each dependent on the last. This section documents the full process of going from zero to a working browser-based neural chess engine.

### Phase 1 — C++ Chess Engine

**Goal:** Build a complete, correct chess engine from scratch in C++20.

The foundation of the entire project is a fast, correct chess engine. Everything else — MCTS, neural evaluation, the web UI — depends on the engine producing *perfectly legal* moves.

**What was built:**

1. **Board Representation** — An array-based `uint8_t board[64]` representation storing piece type and color in each square. While bitboards are faster for high-performance engines, the array approach was chosen for simplicity and direct compatibility with the neural network's spatial encoding.

2. **Move Generation** — A `pseudoLegalMoves()` generator produces all candidate moves (pawn pushes, captures, double pushes, en passant, castling, knight/bishop/rook/queen/king moves). A legality filter then removes any move that leaves the king in check.

3. **Make / Undo Move** — Every `makeMove()` pushes a `State` snapshot (captured piece, castling rights, en passant square, half-move clock) onto a history stack, enabling `undoMove()` to perfectly restore the position. This is critical for MCTS, which explores and backtracks thousands of times per search.

4. **FEN Parsing** — Full FEN serialization (`loadFEN` / `getFEN`) for interop with the frontend and standard chess tools.

5. **Terminal Detection** — `isCheckmate()`, `isStalemate()`, `isDraw()` (50-move rule, insufficient material), and `isTerminal()` for MCTS leaf evaluation.

6. **Perft Testing** — The `perft()` function counts all leaf nodes at a given depth and was used to validate move generation against known perft results from the chess programming community. This is the gold standard for proving correctness.

7. **Tensor Export** — `toTensor()` converts the board state into an `[18, 8, 8]` float array, implementing the exact same encoding used by the Python training pipeline. This ensures the C++ engine and the neural network speak the same language.

**Validation approach:** Native compilation with `g++ -std=c++20` on Windows for rapid edit-compile-test cycles. Each component has a dedicated test file (`test_action.cpp`, `test_mcts.cpp`, `test_puct.cpp`, `test_tensor.cpp`).

### Phase 2 — Neural Network Design

**Goal:** Design and implement a dual-headed ResNet that is small enough for real-time browser inference (~2 MB ONNX) but expressive enough to learn chess strategy.

**What was built:**

1. **Network Architecture** (`network.py`) — A configurable ResNet with a `Permute` layer at the input (channels-first → channels-last conversion) to maximize compatibility with TensorFlow's CPU/ONNX backends. The architecture has two tunable knobs: `num_res_blocks` (depth) and `num_filters` (width).

2. **Action Space Encoding** (`action_space.py`) — A Python implementation of the 4,672-move action space mapping, mirroring the C++ `ActionSpace` class. Both use the same `action_index = from_square × 73 + direction_plane` formula. Cross-validated with `test_action.py` and `test_action.cpp` to guarantee identical index mappings.

3. **Board Encoder** (`encode.py`) — `fen_to_tensor()` converts any FEN string into the same `[18, 8, 8]` tensor format produced by the C++ engine's `toTensor()`, cross-validated with `test_tensor.py` and `test_tensor.cpp`.

4. **Loss Function** — The AlphaZero loss combines:
   - **Policy loss**: Categorical cross-entropy `−Σ π(a) log P(a)` between the MCTS visit distribution and the network's predicted policy.
   - **Value loss**: Mean squared error `(z − V(s))²` between the game outcome and the network's predicted value.
   - **L2 regularization**: Applied to all convolutional and dense kernels to prevent overfitting.
   - **Total**: `L = L_policy + λ × L_value + c × ||θ||²`

5. **ONNX Export** (`export_onnx.py`) — Converts trained `.keras` checkpoints to `.onnx` format (opset 15) with a fixed input signature `[batch, 18, 8, 8]` matching the C++ tensor layout.

### Phase 3 — Stockfish Data Bootstrapping

**Goal:** Overcome the cold-start problem. A randomly initialized neural network plays terrible chess, making pure self-play prohibitively slow in early stages.

**The problem:** AlphaZero at DeepMind used thousands of TPUs running for days. On a single consumer GPU, pure self-play from random weights would take weeks to produce meaningful training signal — the random network has no concept of piece value, king safety, or basic tactics, so early self-play games are mostly random moves ending in draws or blunders.

**The solution:** Use Stockfish as a "teacher" to generate high-quality supervised training data that bootstraps the network's knowledge before switching to self-play.

**How `stockfish_generator.py` works:**

1. **Multi-PV Analysis** — For each position, Stockfish analyzes with `multipv=5` (top 5 moves) at a configurable search depth (default: 10). This produces a ranked list of strong moves, not just the single best move.

2. **Policy Target Construction** — The Stockfish move scores (centipawn evaluations) are converted into a soft probability distribution using a sharpened softmax:
   ```
   P(move) = exp(score × 5) / Σ exp(scores × 5)
   ```
   This creates a policy target that concentrates probability on the best moves while giving partial credit to reasonable alternatives — much richer signal than a one-hot "best move only" label.

3. **Value Conversion** — Stockfish's centipawn scores are squashed into `[-1, +1]` using `tanh(cp / 400)`. Mate scores map directly to ±1.0.

4. **Game Outcome Assignment** — Each position in a game is labeled with the final game result `z ∈ {-1, 0, +1}`, combining Stockfish's per-move analysis with hindsight knowledge of how the game actually ended.

5. **Batch Saving** — Positions are saved as `.npz` files every 20 games, producing the same `(states, policies, values)` tensor format that the training pipeline expects.

**Why this matters:** After training on ~5,000 Stockfish-generated positions, the network learns basic piece values, elementary tactics, and reasonable opening play. This takes hours instead of weeks, providing a much stronger starting point for self-play.

### Phase 4 — Self-Play Loop

**Goal:** Once bootstrapped, the network improves itself through the classic AlphaZero self-play cycle.

**The self-play pipeline has two implementations:**

#### Standard Self-Play (`self_play.py`)

The straightforward approach: play one game at a time, sequentially.

1. Load the current best model weights
2. Play a full game using MCTS (default: 100 simulations/move)
3. Apply **temperature sampling** for move selection:
   - Early game (moves 1–30): `τ = 1.0` (exploratory — proportional to visit count)
   - Late game (moves 31+): `τ = 0.1` (nearly greedy — picks the most-visited move)
4. Record every position's `(state, π_target, z)` tuple
5. After the game ends, assign the terminal outcome `z` to all positions (hindsight labeling)
6. Save batches of positions as `.npz` files

#### Batched Self-Play (`batched_self_play.py`)

A GPU-optimized version that runs **64 games simultaneously** to saturate the GPU:

1. Maintain a pool of `GameWorker` instances, each tracking its own board, MCTS tree, and game history
2. In each MCTS simulation round, collect leaf nodes from *all* active games
3. **Batch the neural network forward pass**: stack all leaf tensors into a single `[batch_size, 18, 8, 8]` input and run one `model(batch)` call
4. Distribute the results back to each game's MCTS tree for backpropagation
5. After all simulations are done for a round, each game selects its move independently
6. Finished games are replaced with fresh ones until the quota is met

This batched approach achieves **10–50× throughput** compared to sequential self-play on GPU, since the neural network's fixed overhead (kernel launch, memory transfer) is amortized across many positions.

#### The Continuous Improvement Loop

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    │
│   │  Self-Play   │    │   Replay     │    │  Train Network    │    │
│   │  (MCTS +    │───►│   Buffer     │───►│  (SGD on replay   │    │
│   │   Current   │    │  (50K FIFO)  │    │   buffer batches) │    │
│   │   Network)  │    │              │    │                   │    │
│   └─────────────┘    └──────────────┘    └────────┬──────────┘    │
│         ▲                                         │               │
│         │           ┌──────────────┐              │               │
│         │           │  Export to   │              │               │
│         └───────────│  ONNX       │◄─────────────┘               │
│                     │  (tf2onnx)  │                               │
│   New weights       └──────────────┘   Improved checkpoint       │
│   fed back to                                                     │
│   self-play                                                       │
└────────────────────────────────────────────────────────────────────┘
```

Each iteration produces a stronger network:
1. **Generate** 100–500 self-play games → ~10K–50K training positions
2. **Aggregate** into the circular `AlphaZeroReplayBuffer` (FIFO, 50K capacity) — old data is gradually evicted as new, higher-quality data arrives
3. **Train** for 10–20 epochs on shuffled mini-batches (batch size 64, Adam optimizer with exponential LR decay)
4. **Checkpoint** the model with versioned naming (`ckpt_v001.keras`, `ckpt_v002.keras`, ...) and metadata (epoch, losses, hyperparameters)
5. **Export** the latest checkpoint to `.onnx`
6. **Repeat** — the new model generates stronger self-play data, creating the positive feedback loop

The `train_pipeline.py` handles steps 3–5 in a single run, with `--resume` support to continue from the latest checkpoint.

### Phase 5 — WebAssembly Compilation

**Goal:** Port the validated C++ engine to run in-browser at near-native speed.

Once all C++ unit tests pass natively, the build target switches from `g++` to Emscripten's `em++`:

```bash
em++ -std=c++20 chess.cpp action_space.cpp mcts.cpp api.cpp -o chess_engine.js \
    --bind -s MODULARIZE=1 -s EXPORT_NAME="createChessModule" \
    -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -O3
```

**Key decisions:**
- **Embind** (`--bind`) for C++ ↔ JavaScript interop — exposes `ChessEngine` as a JavaScript class with methods like `loadFEN()`, `getLegalMoves()`, `mctsStep()`
- **MODULARIZE=1** — wraps the WASM module in a factory function `createChessModule()` for clean async loading in Web Workers
- **ALLOW_MEMORY_GROWTH** — the MCTS tree allocates nodes dynamically; fixed memory would crash on deep searches
- **-O3** — maximum optimization for production performance

The output is two files: `chess_engine.js` (loader + glue code) and `chess_engine.wasm` (the compiled binary).

### Phase 6 — Browser Integration

**Goal:** Wire everything together into a responsive single-page application.

1. **Web Worker** (`worker.ts`) — Loads the WASM module and ONNX model in a background thread. Implements a message-passing protocol (`UIEvent` / `WorkerEvent`) for communication with the React UI. The worker orchestrates the MCTS ↔ ONNX inference loop.

2. **Neural Network Wrapper** (`nn.ts`) — A thin TypeScript class around `onnxruntime-web` that loads the `.onnx` model, wraps `Float32Array` inputs into ONNX tensors, and extracts policy/value outputs. Uses the WASM execution backend for maximum compatibility.

3. **React Hook** (`useChessWorker.ts`) — Encapsulates all Web Worker communication behind a clean React hook: `{ isReady, fen, legalMoves, makeMove, undoMove, startAnalysis, bestMove, evaluation, ... }`. The UI never touches WASM or ONNX directly.

4. **UI Components** — Chessboard rendering, evaluation bar, analysis panel, and neural engine controls. The board converts FEN strings into a visual grid; legal moves are highlighted on click; analysis results display the best move and evaluation score.

5. **Memory Management** — Every Embind vector returned from C++ (legal moves, tensors) is explicitly `.delete()`'d in JavaScript to prevent WebAssembly memory leaks during long analysis sessions.

---

## Project Structure

```
alpha-zero-web/
├── frontend/                  # Browser UI (Next.js + React + Tailwind)
│   ├── src/
│   │   ├── app/page.tsx       # Main game page
│   │   ├── components/chess/  # ChessBoard, EvaluationBar, SidePanels
│   │   ├── hooks/             # useChessWorker (React ↔ Worker bridge)
│   │   └── lib/engine/
│   │       ├── worker.ts      # Web Worker (WASM + ONNX orchestration)
│   │       ├── nn.ts          # ONNX Runtime inference wrapper
│   │       └── protocol.ts    # Type-safe message protocol
│   └── public/
│       ├── engine/            # chess_engine.js + chess_engine.wasm
│       └── models/            # ONNX model files
│
├── engine/                    # C++20 Chess Engine
│   ├── chess.hpp / chess.cpp  # Board representation, move gen, FEN, perft
│   ├── mcts.hpp / mcts.cpp   # Monte Carlo Tree Search (state machine)
│   ├── action_space.*        # AlphaZero move ↔ index mapping
│   ├── wasm/
│   │   ├── api.cpp            # Embind API (C++ ↔ JavaScript bindings)
│   │   └── build.bat          # Emscripten compilation script
│   └── test_*.cpp             # Native unit tests
│
├── training/                  # Python Training Pipeline
│   ├── network.py             # ResNet model definition (TensorFlow/Keras)
│   ├── train_pipeline.py      # Full training loop with checkpointing
│   ├── self_play.py           # MCTS self-play data generator
│   ├── batched_self_play.py   # Batched parallel self-play (GPU-optimized)
│   ├── replay_buffer.py       # Circular FIFO experience replay buffer
│   ├── encode.py              # FEN → [18,8,8] tensor encoding
│   ├── action_space.py        # Python mirror of C++ action space
│   ├── export_onnx.py         # Keras → ONNX conversion
│   ├── stockfish_generator.py # Supervised data from Stockfish (bootstrapping)
│   ├── train_random.py        # Baseline training from random weights
│   ├── checkpoints/           # Versioned .keras checkpoints + metadata
│   ├── dataset/               # Self-play .npz data files
│   ├── logs/                  # TensorBoard event files
│   └── requirements.txt       # Python dependencies
│
├── models/                    # Production ONNX models for browser
└── docs/                      # Architecture & design documentation
    ├── architecture.md        # Compilation strategy & system design
    ├── network_architecture.md # Neural network topology reference
    └── neural_encoding.md     # Input/output tensor specification
```

---

## Technologies Used

### Frontend
| Technology | Purpose |
|---|---|
| [Next.js 16](https://nextjs.org/) | React framework, routing, bundling |
| [React 19](https://react.dev/) | Component-based UI |
| [TypeScript 5](https://www.typescriptlang.org/) | Type-safe JavaScript |
| [Tailwind CSS 4](https://tailwindcss.com/) | Utility-first styling |
| [ONNX Runtime Web](https://onnxruntime.ai/) | In-browser neural network inference (WASM backend) |
| Web Workers | Background thread for engine + NN compute |

### Engine
| Technology | Purpose |
|---|---|
| C++20 | Core chess engine (board, move gen, MCTS) |
| [Emscripten](https://emscripten.org/) | C++ → WebAssembly compiler |
| Embind | C++ ↔ JavaScript API bindings |

### Training
| Technology | Purpose |
|---|---|
| [Python 3.10+](https://python.org/) | Training scripts & self-play |
| [TensorFlow ≥ 2.15](https://www.tensorflow.org/) | Neural network training & GPU acceleration |
| [tf2onnx](https://github.com/onnx/tensorflow-onnx) | Keras → ONNX model export |
| [NumPy](https://numpy.org/) | Tensor manipulation & data pipeline |
| [python-chess](https://python-chess.readthedocs.io/) | Legal move generation for self-play |
| [Stockfish](https://stockfishchess.org/) | Supervised data generation (bootstrapping) |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | Loss curves & training visualization |

---

## Prerequisites

| Tool | Version | Install Guide |
|---|---|---|
| **Node.js** | ≥ 18 | [nodejs.org](https://nodejs.org/) |
| **npm** | ≥ 9 | Bundled with Node.js |
| **Python** | ≥ 3.10 | [python.org](https://python.org/) |
| **Emscripten SDK** | Latest | [emscripten.org/docs/getting_started](https://emscripten.org/docs/getting_started/) |
| **g++ (MinGW)** | C++20 support | [mingw-w64.org](https://www.mingw-w64.org/) (for native testing) |
| **Stockfish** | ≥ 16 | [stockfishchess.org](https://stockfishchess.org/) (optional, for data bootstrapping) |

---

## How to Run

### 1. Frontend (Next.js UI)

```bash
cd alpha-zero-web/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The app will be available at **http://localhost:3000**.

> **Note:** The frontend expects compiled WASM files at `public/engine/` and an ONNX model at `public/models/`. See the WASM build step below.

### 2. C++ Engine — Native Build (Development & Testing)

Use native compilation for rapid iteration and debugging before compiling to WASM:

```bash
cd alpha-zero-web/engine

# Compile the engine (example: perft test)
g++ -std=c++20 -O2 chess.cpp action_space.cpp perft.cpp -o perft.exe
./perft.exe

# Run MCTS unit test
g++ -std=c++20 -O2 chess.cpp action_space.cpp mcts.cpp test_mcts.cpp -o test_mcts.exe
./test_mcts.exe

# Run action space tests
g++ -std=c++20 -O2 chess.cpp action_space.cpp test_action.cpp -o test_action.exe
./test_action.exe

# Run PUCT scoring tests
g++ -std=c++20 -O2 chess.cpp action_space.cpp mcts.cpp test_puct.cpp -o test_puct.exe
./test_puct.exe

# Run tensor encoding cross-validation
g++ -std=c++20 -O2 chess.cpp test_tensor.cpp -o test_tensor.exe
./test_tensor.exe
```

### 3. C++ Engine — WebAssembly Build (Browser Deployment)

Requires the [Emscripten SDK](https://emscripten.org/) to be installed and activated:

```bash
cd alpha-zero-web/engine/wasm

# Compile C++ to WebAssembly
# (or simply run the build script)
build.bat
```

This produces `chess_engine.js` and `chess_engine.wasm`. Copy them into the frontend:

```bash
copy chess_engine.js ..\..\frontend\public\engine\
copy chess_engine.wasm ..\..\frontend\public\engine\
```

### 4. Training Pipeline

```bash
cd alpha-zero-web/training

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
pip install python-chess      # Required for self-play
```

#### Step A: Bootstrap with Stockfish (Recommended First)

```bash
# Generate supervised data from Stockfish analysis
python stockfish_generator.py \
    --engine /path/to/stockfish \
    --games 200 \
    --depth 10 \
    --dir dataset
```

#### Step B: Train on Stockfish Data

```bash
python train_pipeline.py \
    --data "dataset/*.npz" \
    --epochs 15 \
    --batch_size 64 \
    --blocks 6 \
    --filters 128 \
    --onnx_out model_v001.onnx
```

#### Step C: Self-Play with Trained Network

```bash
# Standard self-play (CPU, sequential)
python self_play.py \
    --weights checkpoints/ckpt_v001.keras \
    --games 200 \
    --sims 100 \
    --dir dataset

# OR: Batched self-play (GPU, 64 parallel games)
python batched_self_play.py \
    --weights checkpoints/ckpt_v001.keras \
    --games 256 \
    --batch_size 64 \
    --sims 100 \
    --dir dataset
```

#### Step D: Retrain on Combined Data

```bash
python train_pipeline.py \
    --data "dataset/*.npz" \
    --resume \
    --epochs 10 \
    --onnx_out model_v002.onnx
```

#### Step E: Repeat Steps C–D

Each cycle produces a stronger network. Monitor progress:

```bash
tensorboard --logdir logs
```

#### Step F: Deploy to Browser

```bash
copy model_v002.onnx ..\frontend\public\models\model_v001.onnx
```

---

## Training Workflow

The full training pipeline combines supervised bootstrapping with self-play reinforcement:

```
                    ┌─────────────────────────────────┐
                    │         BOOTSTRAPPING            │
                    │                                  │
                    │  Stockfish  ──►  Supervised Data │
                    │  (MultiPV)      (states, π, z)   │
                    │                                  │
                    └──────────┬──────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                   SELF-PLAY IMPROVEMENT LOOP                       │
│                                                                    │
│   ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    │
│   │  Self-Play   │    │   Replay     │    │  Train Network    │    │
│   │  (MCTS +    │───►│   Buffer     │───►│  (SGD on replay   │    │
│   │   Current   │    │  (50K FIFO)  │    │   buffer batches) │    │
│   │   Network)  │    │              │    │                   │    │
│   └─────────────┘    └──────────────┘    └────────┬──────────┘    │
│         ▲                                         │               │
│         │           ┌──────────────┐              │               │
│         │           │  Export to   │              │               │
│         └───────────│  ONNX       │◄─────────────┘               │
│                     │  (tf2onnx)  │                               │
│   New weights       └──────────────┘   Improved checkpoint       │
│   fed back to                                                     │
│   self-play                                                       │
└────────────────────────────────────────────────────────────────────┘
```

**Data format:** Every training sample is a tuple `(state, π, z)`:

| Field | Shape | Description |
|---|---|---|
| `state` | `[18, 8, 8]` | Board position tensor |
| `π` | `[4672]` | MCTS visit distribution (policy target) |
| `z` | `[1]` | Game outcome from this position's perspective: +1 (white win), 0 (draw), −1 (black win) |

**Loss function:** The network is trained to minimize:
- **Policy loss**: Cross-entropy `−Σ π(a) log P(a)` — teaches the network which moves MCTS found most promising
- **Value loss**: MSE `(z − V(s))²` — teaches the network to predict game outcomes
- **L2 regularization**: `c × ||θ||²` — prevents overfitting on the finite replay buffer

**Learning rate schedule:** Exponential decay (Adam optimizer, initial LR = 0.001, decay every 1000 steps by factor 0.96).

**Checkpointing:** Models are saved as versioned `.keras` files with JSON metadata tracking epoch, global step, architecture config, and loss values.

---

## Design Decisions & Trade-offs

| Decision | Rationale |
|---|---|
| **Array board (not bitboards)** | Simpler to implement, debug, and map directly to the [18,8,8] tensor. Bitboards are faster but add complexity that doesn't pay off when the bottleneck is neural network inference, not move generation. |
| **State-machine MCTS (not threaded)** | JavaScript Web Workers are single-threaded. A state machine that yields to `await` between steps keeps the event loop alive and allows the `STOP_ANALYSIS` cancel button to work. |
| **Emscripten Embind (not raw WASM exports)** | Embind provides a natural C++ class interface in JavaScript (`engine.loadFEN("...")`) instead of dealing with raw memory pointers and manual serialization. |
| **Channels-first → Permute layer** | The C++ engine naturally produces [18,8,8] tensors (channels-first, like PyTorch). Instead of transposing in C++, a `Permute` layer in the network handles conversion, keeping both sides simpler. |
| **Stockfish bootstrapping** | Pure self-play from random weights wastes compute on meaningless games. Stockfish data provides a strong initialization in hours, not weeks. The self-play loop then refines beyond Stockfish's style. |
| **FIFO replay buffer (not prioritized)** | Follows the original AlphaZero paper. Old data is evicted as the network improves, naturally biasing training toward recent, higher-quality games. |
| **Softmax policy (not logits)** | The network outputs calibrated probabilities, making it easy to directly mask and renormalize for legal moves without temperature adjustments. |
| **ONNX (not TensorFlow.js)** | ONNX Runtime Web provides reliable WASM inference that works consistently across browsers. TensorFlow.js has more overhead and browser-specific issues. |

---

## Key Challenges Solved

### 1. C++ ↔ Python Tensor Parity
The board encoding must be *exactly identical* between the C++ `toTensor()` and Python `fen_to_tensor()`. A single off-by-one in plane ordering or spatial mapping would cause the neural network to misinterpret positions entirely. This was validated with cross-language test suites (`test_tensor.cpp` vs `test_tensor.py`) that compare tensors for every square, every piece type, and every special flag.

### 2. Action Space Consistency
The 4,672 action indices must map identically in C++ (`ActionSpace::moveToIndex`) and Python (`ActionSpace.move_to_index`). Direction ordering (N, NE, E, SE, S, SW, W, NW), knight jump sequences, and underpromotion encoding were cross-validated with `test_action.cpp` and `test_action.py`.

### 3. WASM Memory Leaks
Every Embind `std::vector` returned to JavaScript allocates WebAssembly heap memory that the JavaScript garbage collector cannot see. Without explicit `.delete()` calls, the WASM heap grows unboundedly during long analysis sessions. Every vector in `worker.ts` is explicitly deleted after use.

### 4. Cooperative MCTS in Single-Threaded JavaScript
Traditional MCTS runs a tight C++ loop. In the browser, the neural network runs asynchronously via ONNX Runtime. The solution is a state-machine MCTS (`SELECTING → AWAITING_EVAL → back to SELECTING`) that pauses at each leaf and resumes when the evaluation arrives, yielding to `await` at each step.

### 5. Cold-Start Problem
A randomly initialized network produces a uniform policy over all 4,672 actions — essentially random play. Self-play games between random players produce near-zero learning signal (mostly draws by the 50-move rule). Stockfish bootstrapping solves this by providing thousands of expert-level positions with rich policy targets, giving the network a strong foundation before self-play begins.

---

## Future Roadmap

- [ ] **Opening book integration** — precomputed first 10–15 moves to skip MCTS overhead in well-known positions
- [ ] **Pondering** — continue MCTS search during the opponent's turn
- [ ] **Shared WASM memory** — use `SharedArrayBuffer` for zero-copy tensor transfer between WASM and ONNX
- [ ] **WebGPU inference** — replace ONNX WASM backend with WebGPU for 5–10× faster neural network evaluation
- [ ] **History planes** — extend input encoding with the last 8 board positions (like full AlphaZero) for better repetition and tempo understanding
- [ ] **Distributed self-play** — run `batched_self_play.py` across multiple machines (Colab, Kaggle) with a central training server
- [ ] **ELO tracking** — automated match tournaments between model checkpoints to measure improvement quantitatively
- [ ] **Time management** — allocate more MCTS simulations to critical positions and fewer to forced moves

---

## License

This project is for educational and research purposes.