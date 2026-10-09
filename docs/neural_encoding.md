# Neural Network Input Encoding

The board state is encoded as a 3D float32 tensor of shape `[18, 8, 8]` (Planes × Height × Width). 
When flattened for Dense layers or ONNX inference, it is a 1D array of exactly `1,152` elements.

## Spatial Mapping (Height x Width)
The 8×8 spatial grid represents the chessboard from a fixed absolute perspective (White at the bottom). The mapping follows standard computer vision and FEN reading order (top-to-bottom, left-to-right):
* **Row 0:** Rank 8 (Squares A8 to H8)
* **Row 7:** Rank 1 (Squares A1 to H1)
* **Col 0:** File A
* **Col 7:** File H

## Plane Specification (18 Planes)
Each plane is an 8×8 grid where `1.0` represents the presence of a feature and `0.0` represents its absence. Global state flags (like turn and castling) are broadcasted across their entire 8×8 plane so convolutional filters can detect them at any spatial location.

| Plane | Feature | Encoding Strategy |
|---|---|---|
| 0 | White Pawns | `1.0` at piece locations |
| 1 | White Knights | `1.0` at piece locations |
| 2 | White Bishops | `1.0` at piece locations |
| 3 | White Rooks | `1.0` at piece locations |
| 4 | White Queens | `1.0` at piece locations |
| 5 | White King | `1.0` at piece locations |
| 6 | Black Pawns | `1.0` at piece locations |
| 7 | Black Knights | `1.0` at piece locations |
| 8 | Black Bishops | `1.0` at piece locations |
| 9 | Black Rooks | `1.0` at piece locations |
| 10 | Black Queens | `1.0` at piece locations |
| 11 | Black King | `1.0` at piece locations |
| 12 | Side to Move | All `1.0` if White's turn, All `0.0` if Black's turn |
| 13 | White Kingside Castling | All `1.0` if available, All `0.0` otherwise |
| 14 | White Queenside Castling | All `1.0` if available, All `0.0` otherwise |
| 15 | Black Kingside Castling | All `1.0` if available, All `0.0` otherwise |
| 16 | Black Queenside Castling | All `1.0` if available, All `0.0` otherwise |
| 17 | En Passant Capture Square | `1.0` exactly on the target square, `0.0` elsewhere |

## Action Space Encoding (Policy Head)

The policy head outputs a flat vector of `4,672` probabilities representing every mathematically possible move on the board, regardless of legality in the current position. 

### Space Dimensions
* **Source Squares:** 64
* **Move Directions:** 73
* **Total Action Space:** 64 × 73 = 4,672

### Index Calculation
`action_index = (from_square * 73) + plane`

### The 73 Direction Planes
1. **Queen Moves (Planes 0 to 55):**
   * Encodes standard moves for Queens, Kings, Pawns, Rooks, and Bishops.
   * Queen promotions are also encoded here as normal forward/diagonal pawn moves of distance 1.
   * 8 compass directions × 7 distances.
   * Directions ordered clockwise starting from North: `N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7`.
   * Plane formula: `direction_index * 7 + (distance - 1)`
2. **Knight Jumps (Planes 56 to 63):**
   * Ordered by clock-face: `(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)`.
3. **Underpromotions (Planes 64 to 72):**
   * Knight (64-66), Bishop (67-69), Rook (70-72).
   * Inside each group of 3, directions are ordered by dx: Left Capture (-1), Straight (0), Right Capture (+1).

### Masking Illegal Actions
During MCTS, the engine generates the strictly legal moves, computes their `action_index`, and extracts only those specific probabilities from the `4,672` tensor. Illegal moves are masked out by ignoring their indices.

