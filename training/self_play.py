import os
import re
import time
import math
import argparse
import numpy as np
import chess
import tensorflow as tf

from network import create_alphazero_model
from encode import fen_to_tensor
from action_space import ActionSpace

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

class MCTSNode:
    def __init__(self, prior: float, turn: bool, parent=None):
        self.prior = prior
        self.turn = turn 
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0 
        self.children = {} 

class SelfPlayMCTS:
    def __init__(self, model, c_puct=1.414):
        self.model = model
        self.c_puct = c_puct

    def search(self, root_board: chess.Board, simulations: int) -> dict:
        root = MCTSNode(1.0, root_board.turn)
        
        for _ in range(simulations):
            node = root
            board = root_board.copy(stack=False)
            
            while node.children:
                best_score = -float('inf')
                best_move, best_child = None, None
                sqrt_parent_visits = math.sqrt(node.visits)
                
                for move_str, child in node.children.items():
                    q = (child.value_sum / child.visits) if child.visits > 0 else 0.0
                    q_persp = q if node.turn == chess.WHITE else -q
                    u = self.c_puct * child.prior * sqrt_parent_visits / (1.0 + child.visits)
                    
                    if q_persp + u > best_score:
                        best_score = q_persp + u
                        best_move, best_child = move_str, child
                
                board.push_uci(best_move)
                node = best_child
                
            terminal = board.is_game_over(claim_draw=True)
            if terminal:
                res = board.result(claim_draw=True)
                val = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
            else:
                tensor = fen_to_tensor(board.fen())
                policy, value = self.model(np.expand_dims(tensor, 0), training=False)
                val = value.numpy()[0][0]
                policy_np = policy.numpy()[0]
                
                legal_moves = list(board.legal_moves)
                sum_prior = 0.0
                priors = []
                
                for move in legal_moves:
                    flag = get_move_flag(board, move)
                    idx = ActionSpace.move_to_index(move.from_square, move.to_square, flag)
                    p = policy_np[idx]
                    priors.append(p)
                    sum_prior += p
                    
                for i, move in enumerate(legal_moves):
                    norm_p = priors[i] / sum_prior if sum_prior > 1e-6 else 1.0 / len(legal_moves)
                    node.children[move.uci()] = MCTSNode(norm_p, board.turn, parent=node)

            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += val
                curr = curr.parent
                
        return {move: child.visits for move, child in root.children.items()}

def apply_temperature(pi: dict, temp: float) -> str:
    moves = list(pi.keys())
    visits = list(pi.values())
    
    if temp < 1e-3:
        return moves[np.argmax(visits)]
        
    visits = np.array(visits) ** (1.0 / temp)
    probs = visits / np.sum(visits)
    return np.random.choice(moves, p=probs)

def play_game(model, simulations=100, max_moves=200, temp_threshold=30):
    board = chess.Board()
    mcts = SelfPlayMCTS(model)
    game_history = [] 
    
    step = 0
    while not board.is_game_over(claim_draw=True) and step < max_moves:
        temp = 1.0 if step < temp_threshold else 0.1
        
        pi_dict = mcts.search(board, simulations)
        
        state_tensor = fen_to_tensor(board.fen())
        pi_target = np.zeros(4672, dtype=np.float32)
        total_visits = sum(pi_dict.values())
        
        for move_uci, visits in pi_dict.items():
            move = chess.Move.from_uci(move_uci)
            flag = get_move_flag(board, move)
            idx = ActionSpace.move_to_index(move.from_square, move.to_square, flag)
            pi_target[idx] = visits / total_visits
            
        game_history.append((state_tensor, pi_target))
        
        best_move_uci = apply_temperature(pi_dict, temp)
        board.push_uci(best_move_uci)
        step += 1
            
    res = board.result(claim_draw=True)
    if res == "1-0": z = 1.0
    elif res == "0-1": z = -1.0
    else: z = 0.0
    
    print(f"Game over after {step} moves. Result: {res} (Z={z})")
    
    states, policies, values = [], [], []
    for state, pi in game_history:
        states.append(state)
        policies.append(pi)
        values.append(np.array([z], dtype=np.float32))
        
    return states, policies, values

def get_next_batch_index(output_dir: str, prefix: str) -> int:
    """Scans output_dir for existing files like games_batch_120.npz and returns the highest number."""
    if not os.path.exists(output_dir):
        return 0
    max_idx = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.npz$")
    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx

def generate_dataset(weights_path, num_games, save_every, output_dir, simulations, prefix, seed):
    # Dynamic or explicit seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32 - 1)
    np.random.seed(seed)
    print(f"Active random seed: {seed}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect resume index
    starting_game_offset = get_next_batch_index(output_dir, prefix)
    print(f"Detected existing games up to: {starting_game_offset}. New batches will continue from there.")
    
    print(f"Loading weights from {weights_path}...")
    if os.path.exists(weights_path):
        model = tf.keras.models.load_model(weights_path)
    else:
        print("Weights not found, initializing fresh model...")
        model = create_alphazero_model()
        
    all_states, all_policies, all_values = [], [], []
    
    for i in range(1, num_games + 1):
        global_game_num = starting_game_offset + i
        print(f"--- Game {i}/{num_games} (Total Run Count: {global_game_num}) ---")
        s, p, v = play_game(model, simulations=simulations)
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)
        
        if i % save_every == 0 or i == num_games:
            filename = os.path.join(output_dir, f"{prefix}_{global_game_num}.npz")
            np.savez_compressed(
                filename, 
                states=np.array(all_states, dtype=np.float32),
                policies=np.array(all_policies, dtype=np.float32),
                values=np.array(all_values, dtype=np.float32)
            )
            print(f"*** Checkpoint Saved: {len(all_states)} positions written to {filename} ***")
            all_states, all_policies, all_values = [], [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Data Generator")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate in this run")
    parser.add_argument("--save_every", type=int, default=10, help="Save frequency (micro-batching)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--dir", type=str, default="dataset", help="Output directory for .npz files")
    parser.add_argument("--weights", type=str, default="alphazero_random_weights.keras", help="Path to model weights")
    parser.add_argument("--prefix", type=str, default="games_batch", help="Filename prefix (e.g. colab_batch, kaggle_batch)")
    parser.add_argument("--seed", type=int, default=None, help="Explicit random seed (leaves dynamic if omitted)")
    
    args = parser.parse_args()
    
    generate_dataset(
        weights_path=args.weights,
        num_games=args.games,
        save_every=args.save_every,
        output_dir=args.dir,
        simulations=args.sims,
        prefix=args.prefix,
        seed=args.seed
    )