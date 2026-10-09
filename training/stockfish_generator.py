import os
import math
import time
import argparse
import numpy as np
import chess
import chess.engine

from encode import fen_to_tensor
from action_space import ActionSpace

# Reuse your existing move flag logic
def get_move_flag(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    target = board.piece_at(move.to_square)
    if piece and piece.piece_type == chess.PAWN:
        if abs(move.to_square - move.from_square) == 16: return 1
        if move.promotion:
            if move.promotion == chess.QUEEN: return 11
            if move.promotion == chess.KNIGHT: return 8
            if move.promotion == chess.BISHOP: return 9
            if move.promotion == chess.ROOK: return 10
        if board.is_en_passant(move): return 5
    elif piece and piece.piece_type == chess.KING:
        if move.to_square - move.from_square == 2: return 2
        if move.to_square - move.from_square == -2: return 3
    return 4 if target is not None else 0

def cp_to_value(score_obj, turn) -> float:
    """Converts Stockfish centipawn/mate score to a [-1, 1] scalar."""
    if score_obj.is_mate():
        moves_to_mate = score_obj.mate()
        val = 1.0 if moves_to_mate > 0 else -1.0
    else:
        cp = score_obj.score()
        # Scale: 400 centipawns maps roughly to 0.76, 800 cp to 0.96
        val = math.tanh(cp / 400.0)
    
    # Stockfish scores are relative to the side to move; convert to absolute White perspective
    return val if turn == chess.WHITE else -val

def run_stockfish_generation(engine_path: str, num_games: int, output_dir: str, depth: int = 10, multi_pv: int = 5):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 2, "Hash": 128})
    
    all_states, all_policies, all_values = [], [], []
    
    print(f"Generating {num_games} games using Stockfish (Depth={depth}, MultiPV={multi_pv})...")
    
    for game_idx in range(1, num_games + 1):
        board = chess.Board()
        game_states, game_policies = [], []
        
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= 150:
            # Request Top N moves from Stockfish
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multi_pv)
            
            pi_target = np.zeros(4672, dtype=np.float32)
            move_scores = []
            valid_moves = []
            
            # Extract scores for the top moves
            for pv in info:
                if "pv" in pv and len(pv["pv"]) > 0:
                    mv = pv["pv"][0]
                    score_val = cp_to_value(pv["score"], board.turn)
                    # Convert absolute board value back to relative value for softmax weighting
                    rel_score = score_val if board.turn == chess.WHITE else -score_val
                    move_scores.append(rel_score)
                    valid_moves.append(mv)
            
            # Create a probability distribution (Policy) favoring the best moves
            if not move_scores:
                break
                
            # Softmax with temperature
            scores = np.array(move_scores)
            exp_scores = np.exp((scores - np.max(scores)) * 5.0) # Multiply by 5 to sharpen the distribution
            probs = exp_scores / np.sum(exp_scores)
            
            for mv, prob in zip(valid_moves, probs):
                flag = get_move_flag(board, mv)
                idx = ActionSpace.move_to_index(mv.from_square, mv.to_square, flag)
                pi_target[idx] = prob
                
            game_states.append(fen_to_tensor(board.fen()))
            game_policies.append(pi_target)
            
            # Play the best move
            board.push(valid_moves[0])
            
        # Game finished, assign final terminal value
        res = board.result(claim_draw=True)
        if res == "1-0": final_z = 1.0
        elif res == "0-1": final_z = -1.0
        else: final_z = 0.0
        
        # Hindsight value assignment: mix Stockfish's per-move eval with the final game result
        for s, p in zip(game_states, game_policies):
            all_states.append(s)
            all_policies.append(p)
            all_values.append(np.array([final_z], dtype=np.float32))
            
        print(f"Game {game_idx}/{num_games} completed ({board.fullmove_number} moves, Result: {res})")
        
        # Save batches periodically
        if game_idx % 20 == 0:
            out_file = os.path.join(output_dir, f"stockfish_distill_{game_idx}_{int(time.time())}.npz")
            np.savez_compressed(
                out_file,
                states=np.array(all_states, dtype=np.float32),
                policies=np.array(all_policies, dtype=np.float32),
                values=np.array(all_values, dtype=np.float32)
            )
            all_states, all_policies, all_values = [], [], []

    engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="/usr/games/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=100, help="Games to generate")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish search depth")
    parser.add_argument("--dir", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()
    
    run_stockfish_generation(args.engine, args.games, args.dir, args.depth)