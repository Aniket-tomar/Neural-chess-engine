import numpy as np
from encode import fen_to_tensor

test_fen = "rnbqkbnr/p1pppppp/8/8/1pP5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1"
tensor = fen_to_tensor(test_fen)
flat_tensor = tensor.flatten()

print(f"Python Tensor Hash Test\nFEN: {test_fen}")
non_zero_indices = np.nonzero(flat_tensor)[0]

output = [f"{idx}:{flat_tensor[idx]}" for idx in non_zero_indices]
print(" ".join(output))
print(f"Total active floats: {len(non_zero_indices)}")