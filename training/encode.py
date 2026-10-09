import numpy as np

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Converts a FEN string to an 18x8x8 numpy array matching the C++ encoding.
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    parts = fen.split(' ')
    board_part, turn_part, castling_part, ep_part = parts[0], parts[1], parts[2], parts[3]
    
    # Planes 0-11: Pieces
    row, col = 0, 0
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for char in board_part:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            plane = piece_to_plane[char]
            tensor[plane, row, col] = 1.0
            col += 1
            
    # Plane 12: Side to Move
    if turn_part == 'w':
        tensor[12, :, :] = 1.0
        
    # Planes 13-16: Castling
    if 'K' in castling_part: tensor[13, :, :] = 1.0
    if 'Q' in castling_part: tensor[14, :, :] = 1.0
    if 'k' in castling_part: tensor[15, :, :] = 1.0
    if 'q' in castling_part: tensor[16, :, :] = 1.0
    
    # Plane 17: En Passant
    if ep_part != '-':
        file_idx = ord(ep_part[0]) - ord('a')
        rank_idx = int(ep_part[1]) - 1
        ep_row = 7 - rank_idx
        ep_col = file_idx
        tensor[17, ep_row, ep_col] = 1.0
        
    return tensor