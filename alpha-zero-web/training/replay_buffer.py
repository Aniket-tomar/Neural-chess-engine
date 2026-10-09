import os
import json
import glob
import numpy as np
from typing import Dict, Any, Tuple, Optional

class AlphaZeroReplayBuffer:
    """
    Circular FIFO Replay Buffer for AlphaZero chess data.
    Maintains exact tensor compatibility:
      - States:  [N, 18, 8, 8]  float32
      - Policy:  [N, 4672]      float32
      - Values:  [N, 1]         float32
    """
    def __init__(self, capacity: int = 50000, version: str = "v1.0.0"):
        self.capacity = int(capacity)
        self.version = version

        # Preallocate memory buffers for fast FIFO updates
        self.states = np.zeros((self.capacity, 18, 8, 8), dtype=np.float32)
        self.policies = np.zeros((self.capacity, 4672), dtype=np.float32)
        self.values = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

        # Dataset-level metadata tracking
        self.total_samples_ingested = 0
        self.total_games_ingested = 0
        self.win_loss_draw = {"1.0": 0, "-1.0": 0, "0.0": 0}

    def add_game(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """
        Adds samples from a single completed game or a contiguous batch.
        Guarantees strict FIFO replacement once capacity is saturated.
        """
        n = len(states)
        if n == 0:
            return

        assert states.shape[1:] == (18, 8, 8), f"Unexpected state shape: {states.shape}"
        assert policies.shape[1] == 4672, f"Unexpected policy shape: {policies.shape}"
        assert values.shape[1] == 1, f"Unexpected value shape: {values.shape}"

        # Track outcome statistics
        final_z = float(values[-1][0])
        z_key = f"{final_z:.1f}"
        if z_key in self.win_loss_draw:
            self.win_loss_draw[z_key] += 1
        self.total_games_ingested += 1
        self.total_samples_ingested += n

        # Write samples into circular buffer
        if self.ptr + n <= self.capacity:
            self.states[self.ptr : self.ptr + n] = states
            self.policies[self.ptr : self.ptr + n] = policies
            self.values[self.ptr : self.ptr + n] = values
            self.ptr = (self.ptr + n) % self.capacity
        else:
            first_chunk = self.capacity - self.ptr
            second_chunk = n - first_chunk

            self.states[self.ptr :] = states[:first_chunk]
            self.policies[self.ptr :] = policies[:first_chunk]
            self.values[self.ptr :] = values[:first_chunk]

            self.states[:second_chunk] = states[first_chunk:]
            self.policies[:second_chunk] = policies[first_chunk:]
            self.values[:second_chunk] = values[first_chunk:]

            self.ptr = second_chunk

        self.size = min(self.size + n, self.capacity)

    def load_npz_files(self, paths_or_glob: str):
        """
        Loads one or multiple .npz batch files produced by self_play.py.
        Supports wildcards (e.g. 'dataset/*.npz').
        """
        if isinstance(paths_or_glob, str):
            files = sorted(glob.glob(paths_or_glob))
        else:
            files = list(paths_or_glob)

        if not files:
            print(f"[Buffer] No files matched: {paths_or_glob}")
            return

        print(f"[Buffer] Loading {len(files)} .npz files into buffer...")
        for filepath in files:
            with np.load(filepath) as data:
                s = data["states"]
                p = data["policies"]
                v = data["values"]

                if v.ndim == 1:
                    v = np.expand_dims(v, -1)

                self.add_game(s, p, v)

        print(f"[Buffer] Successfully populated. Current buffer size: {self.size}/{self.capacity}")

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly samples a batch without replacement."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer containing {self.size} samples.")
        
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Computes current dataset statistics and balance metrics."""
        active_vals = self.values[:self.size].flatten() if self.size > 0 else np.array([])
        
        white_wins = int(np.sum(active_vals > 0.5))
        black_wins = int(np.sum(active_vals < -0.5))
        draws = int(np.sum(np.abs(active_vals) <= 0.5))

        avg_game_len = (
            (self.total_samples_ingested / self.total_games_ingested)
            if self.total_games_ingested > 0
            else 0.0
        )

        return {
            "version_metadata": {
                "schema_version": self.version,
                "state_shape": [18, 8, 8],
                "action_space_dim": 4672,
            },
            "capacity": self.capacity,
            "current_size": self.size,
            "total_samples_ingested": self.total_samples_ingested,
            "total_games_ingested": self.total_games_ingested,
            "average_game_length": round(avg_game_len, 2),
            "buffer_outcome_distribution": {
                "white_win_positions": white_wins,
                "black_win_positions": black_wins,
                "draw_positions": draws,
                "white_win_pct": round((white_wins / self.size * 100), 2) if self.size else 0.0,
                "black_win_pct": round((black_wins / self.size * 100), 2) if self.size else 0.0,
                "draw_pct": round((draws / self.size * 100), 2) if self.size else 0.0,
            },
            "cumulative_game_results": {
                "white_wins (1-0)": self.win_loss_draw.get("1.0", 0),
                "black_wins (0-1)": self.win_loss_draw.get("-1.0", 0),
                "draws/unfinished (*)": self.win_loss_draw.get("0.0", 0),
            }
        }

    def print_stats(self):
        """Displays readable statistics in the terminal."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("          ALPHAZERO REPLAY BUFFER STATS          ")
        print("=" * 50)
        print(f" Buffer Utilization:    {stats['current_size']} / {stats['capacity']} positions")
        print(f" Total Games Processed: {stats['total_games_ingested']}")
        print(f" Average Game Length:   {stats['average_game_length']} moves")
        print("-" * 50)
        print(" Active Position Balance:")
        dist = stats["buffer_outcome_distribution"]
        print(f"   White Wins (+1.0):   {dist['white_win_positions']} ({dist['white_win_pct']}%)")
        print(f"   Black Wins (-1.0):   {dist['black_win_positions']} ({dist['black_win_pct']}%)")
        print(f"   Draws/Stalemates:    {dist['draw_positions']} ({dist['draw_pct']}%)")
        print("-" * 50)
        print(" Game End Results:")
        for k, v in stats["cumulative_game_results"].items():
            print(f"   {k}: {v}")
        print("=" * 50 + "\n")

    def save(self, filepath: str):
        """Saves current buffer state, samples, and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        stats = self.get_stats()
        
        np.savez_compressed(
            filepath,
            states=self.states[:self.size],
            policies=self.policies[:self.size],
            values=self.values[:self.size],
            metadata=json.dumps(stats)
        )
        print(f"[Buffer] Saved {self.size} positions to {filepath}")

    def load(self, filepath: str):
        """Restores a serialized buffer and updates tracking metrics."""
        with np.load(filepath, allow_pickle=True) as data:
            s = data["states"]
            p = data["policies"]
            v = data["values"]
            
            n = len(s)
            self.states[:n] = s
            self.policies[:n] = p
            self.values[:n] = v
            self.size = n
            self.ptr = n % self.capacity

            if "metadata" in data:
                meta = json.loads(str(data["metadata"]))
                self.total_samples_ingested = meta.get("total_samples_ingested", n)
                self.total_games_ingested = meta.get("total_games_ingested", 0)
                self.version = meta.get("version_metadata", {}).get("schema_version", self.version)

        print(f"[Buffer] Restored {self.size} samples from {filepath}")


if __name__ == "__main__":
    # Demonstration & self-test
    buf = AlphaZeroReplayBuffer(capacity=10000, version="v0.0.1")
    
    # 1. Ingest all Kaggle and Colab files generated so far
    buf.load_npz_files("dataset/*.npz")
    
    # 2. View distribution stats
    buf.print_stats()

    # 3. Test random batch extraction
    if buf.size >= 64:
        s_batch, p_batch, v_batch = buf.sample(batch_size=64)
        print(f"Sampled mini-batch successfully:")
        print(f"  States:   {s_batch.shape}")
        print(f"  Policies: {p_batch.shape}")
        print(f"  Values:   {v_batch.shape}")