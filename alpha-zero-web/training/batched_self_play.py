import os
import re
import time
import math
import argparse
import numpy as np
import chess
import tensorflow as tf

from network import build_configurable_alphazero_model
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

class BatchedNode:
    def __init__(self, prior: float, turn: bool, parent=None):
        self.prior = prior
        self.turn = turn
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}

class GameWorker:
    """Manages board state, MCTS root, and move selection for one parallel game."""
    def __init__(self, game_id: int, sims: int = 100, max_moves: int = 200, temp_threshold: int = 30):
        self.game_id = game_id
        self.sims = sims
        self.max_moves = max_moves
        self.temp_threshold = temp_threshold
        self.board = chess.Board()
        self.root = BatchedNode(1.0, self.board.turn)
        self.history = []
        self.steps = 0
        self.done = False
        self.result = 0.0

    def select_leaf(self):
        """Descends the MCTS tree to find a leaf requiring neural evaluation."""
        node = self.root
        b = self.board.copy(stack=False)
        c_puct = 1.414

        while node.children:
            best_score = -float('inf')
            best_move, best_child = None, None
            sqrt_parent = math.sqrt(node.visits)

            for mv, child in node.children.items():
                q = (child.value_sum / child.visits) if child.visits > 0 else 0.0
                q_p = q if node.turn == chess.WHITE else -q
                u = c_puct * child.prior * sqrt_parent / (1.0 + child.visits)
                if q_p + u > best_score:
                    best_score = q_p + u
                    best_move, best_child = mv, child

            b.push_uci(best_move)
            node = best_child

        return node, b

    def step_game(self):
        """Extracts MCTS target policy, plays the sampled move, and resets the root."""
        pi_target = np.zeros(4672, dtype=np.float32)
        total_visits = sum(c.visits for c in self.root.children.values())
        if total_visits == 0:
            self.done = True
            return

        moves, visits = [], []
        for mv_str, child in self.root.children.items():
            mv = chess.Move.from_uci(mv_str)
            flag = get_move_flag(self.board, mv)
            idx = ActionSpace.move_to_index(mv.from_square, mv.to_square, flag)
            pi_target[idx] = child.visits / total_visits
            moves.append(mv_str)
            visits.append(child.visits)

        state_tensor = fen_to_tensor(self.board.fen())
        self.history.append((state_tensor, pi_target))

        # Temperature sampling
        temp = 1.0 if self.steps < self.temp_threshold else 0.1
        if temp < 1e-3:
            chosen_uci = moves[np.argmax(visits)]
        else:
            probs = np.array(visits) ** (1.0 / temp)
            probs /= np.sum(probs)
            chosen_uci = np.random.choice(moves, p=probs)

        self.board.push_uci(chosen_uci)
        self.steps += 1

        # Check termination
        if self.board.is_game_over(claim_draw=True) or self.steps >= self.max_moves:
            self.done = True
            res = self.board.result(claim_draw=True)
            if res == "1-0": self.result = 1.0
            elif res == "0-1": self.result = -1.0
            else: self.result = 0.0
        else:
            self.root = BatchedNode(1.0, self.board.turn)


def run_batched_self_play(model, num_games=100, batch_size=64, sims=100, output_dir="dataset", prefix="batched"):
    os.makedirs(output_dir, exist_ok=True)
    all_states, all_policies, all_values = [], [], []

    active_workers = [GameWorker(i, sims=sims) for i in range(batch_size)]
    completed_games = 0

    print(f"[Batched MCTS] Running {batch_size} concurrent games on GPU...")

    while active_workers and completed_games < num_games:
        # 1. Run MCTS iterations in lockstep across all workers
        for sim_idx in range(sims):
            leaves = []
            states_to_eval = []
            eval_indices = []

            for i, worker in enumerate(active_workers):
                leaf_node, leaf_board = worker.select_leaf()
                
                # If terminal reached within tree, backprop directly
                if leaf_board.is_game_over(claim_draw=True):
                    res = leaf_board.result(claim_draw=True)
                    val = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
                    curr = leaf_node
                    while curr:
                        curr.visits += 1
                        curr.value_sum += val
                        curr = curr.parent
                else:
                    leaves.append((leaf_node, leaf_board))
                    states_to_eval.append(fen_to_tensor(leaf_board.fen()))
                    eval_indices.append(i)

            if not states_to_eval:
                continue

            # 2. SATURATE THE GPU: Run 64-128 states through the neural net in one pass
            input_batch = np.array(states_to_eval, dtype=np.float32)
            policies, values = model(input_batch, training=False)
            policies = policies.numpy()
            values = values.numpy()

            # 3. Expand leaves and backpropagate values
            for idx, (leaf_node, leaf_board) in enumerate(leaves):
                val = values[idx][0]
                pol = policies[idx]

                legal_moves = list(leaf_board.legal_moves)
                sum_p = 0.0
                priors = []
                for mv in legal_moves:
                    flag = get_move_flag(leaf_board, mv)
                    m_idx = ActionSpace.move_to_index(mv.from_square, mv.to_square, flag)
                    p = pol[m_idx]
                    priors.append(p)
                    sum_p += p

                for j, mv in enumerate(legal_moves):
                    norm_p = priors[j] / sum_p if sum_p > 1e-6 else 1.0 / len(legal_moves)
                    leaf_node.children[mv.uci()] = BatchedNode(norm_p, leaf_board.turn, parent=leaf_node)

                curr = leaf_node
                while curr:
                    curr.visits += 1
                    curr.value_sum += val
                    curr = curr.parent

        # 4. Each worker executes its move
        surviving_workers = []
        for worker in active_workers:
            worker.step_game()
            if worker.done:
                completed_games += 1
                for s, p in worker.history:
                    all_states.append(s)
                    all_policies.append(p)
                    all_values.append(np.array([worker.result], dtype=np.float32))
                print(f"[Done] Game {completed_games}/{num_games} finished ({worker.steps} moves, Z={worker.result})")
                
                # Spawn replacement game if quota remains
                if completed_games + len(surviving_workers) < num_games:
                    surviving_workers.append(GameWorker(completed_games + len(surviving_workers), sims=sims))
            else:
                surviving_workers.append(worker)

        active_workers = surviving_workers

    # Save final aggregated data
    out_file = os.path.join(output_dir, f"{prefix}_{int(time.time())}.npz")
    np.savez_compressed(
        out_file,
        states=np.array(all_states, dtype=np.float32),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32)
    )
    print(f"\nSaved {len(all_states)} positions from {num_games} parallel games to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .keras checkpoint")
    parser.add_argument("--games", type=int, default=128, help="Total games to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Parallel games / GPU forward batch size")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--dir", type=str, default="/kaggle/working/gen2", help="Save directory")
    args = parser.parse_args()

    # Load model with tf.function compilation for maximum GPU throughput
    loaded_model = tf.keras.models.load_model(args.weights, compile=False)
    
    # Warm-up GPU
    dummy_input = tf.zeros((args.batch_size, 18, 8, 8), dtype=tf.float32)
    _ = loaded_model(dummy_input, training=False)

    run_batched_self_play(
        model=loaded_model,
        num_games=args.games,
        batch_size=args.batch_size,
        sims=args.sims,
        output_dir=args.dir
    )