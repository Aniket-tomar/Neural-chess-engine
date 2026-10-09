#include "action_space.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <cctype>

const int q_dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
const int q_dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int k_dx[8] = {1, 2, 2, 1, -1, -2, -2, -1};
const int k_dy[8] = {2, 1, -1, -2, -2, -1, 1, 2};

int sign(int val) { return (0 < val) - (val < 0); }

int ActionSpace::moveToIndex(Move m) {
    int from = m.from(), to = m.to(), flag = m.flags();
    int dx = (to % 8) - (from % 8);
    int dy = (to / 8) - (from / 8);
    int plane = -1;

    // Underpromotions (Knight=8, Bishop=9, Rook=10)
    if (flag >= 8 && flag <= 10) {
        int type_idx = flag - 8;
        int dir_idx = (dx == -1) ? 0 : ((dx == 0) ? 1 : 2);
        plane = 64 + type_idx * 3 + dir_idx;
    } else {
        // Check Knight Moves
        bool is_knight = false;
        for (int i = 0; i < 8; ++i) {
            if (k_dx[i] == dx && k_dy[i] == dy) {
                plane = 56 + i;
                is_knight = true;
                break;
            }
        }
        
        // Queen/Sliding/Pawn/King Moves
        if (!is_knight) {
            int sx = sign(dx), sy = sign(dy);
            int dir_idx = -1;
            for (int i = 0; i < 8; ++i) {
                if (q_dx[i] == sx && q_dy[i] == sy) { dir_idx = i; break; }
            }
            int dist = std::max(std::abs(dx), std::abs(dy));
            plane = dir_idx * 7 + (dist - 1);
        }
    }
    return from * 73 + plane;
}

Move ActionSpace::indexToMove(int index, const Position& pos) {
    int from = index / 73;
    int plane = index % 73;
    int dx = 0, dy = 0, flag = 0;
    
    if (plane < 56) {
        int dir = plane / 7;
        int dist = (plane % 7) + 1;
        dx = q_dx[dir] * dist;
        dy = q_dy[dir] * dist;
    } else if (plane < 64) {
        int k_idx = plane - 56;
        dx = k_dx[k_idx];
        dy = k_dy[k_idx];
    } else {
        int up_idx = plane - 64;
        int type_idx = up_idx / 3; // 0=N, 1=B, 2=R
        int dir_idx = up_idx % 3;  // 0=-1, 1=0, 2=+1
        dx = dir_idx - 1;
        
        // We know it's an underpromotion, meaning it's a pawn reaching the end
        dy = (from / 8 == 6) ? 1 : -1; 
        flag = 8 + type_idx;
    }

    int to = from + dy * 8 + dx;
    
    // Recover flags dynamically using the FEN string (which is const-safe)
    if (flag == 0) {
        std::string fen = pos.getFEN();
        
        std::string ep_str = "-";
        size_t ep_start = fen.find(" ", fen.find(" ", fen.find(" ") + 1) + 1) + 1;
        if (ep_start != std::string::npos) {
            ep_str = fen.substr(ep_start, fen.find(" ", ep_start) - ep_start);
        }
        
        int ep_sq = -1;
        if (ep_str != "-") {
            ep_sq = (ep_str[1] - '1') * 8 + (ep_str[0] - 'a');
        }

        // Look at FEN to reconstruct board briefly for piece lookups
        int board[64] = {0};
        int sq = 56, i = 0;
        while (fen[i] != ' ') {
            char c = fen[i++];
            if (c == '/') sq -= 16;
            else if (isdigit(c)) sq += c - '0';
            else { board[sq++] = c; }
        }

        char p_char = board[from];
        char to_char = board[to];
        char lc = tolower(p_char);
        
        if (lc == 'p') {
            if (std::abs(dy) == 2) flag = 1; // Double push
            else if (to == ep_sq) flag = 5; // EP capture
            else if (to / 8 == 0 || to / 8 == 7) flag = 11; // Q promo
            else if (to_char != 0) flag = 4; // Normal capture
        } else if (lc == 'k') {
            if (dx == 2) flag = 2; // K-castle
            else if (dx == -2) flag = 3; // Q-castle
            else if (to_char != 0) flag = 4; // Normal capture
        } else {
            if (to_char != 0) flag = 4; // Normal capture
        }
    }
    
    return Move(from, to, flag);
}