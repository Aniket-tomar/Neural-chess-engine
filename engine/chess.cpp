#include "chess.hpp"
#include <cmath>
#include <cctype>
#include <sstream>

const int CASTLING_MASKS[64] = {
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11
};

Position::Position() {
    loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::loadFEN(const std::string& fen) {
    for (int i = 0; i < 64; i++) board[i] = 0;
    int sq = 56, i = 0;
    
    while (fen[i] != ' ') {
        char c = fen[i++];
        if (c == '/') sq -= 16;
        else if (isdigit(c)) sq += c - '0';
        else {
            int color = isupper(c) ? WHITE : BLACK;
            int type = NONE;
            char lc = tolower(c);
            if (lc == 'p') type = PAWN;
            else if (lc == 'n') type = KNIGHT;
            else if (lc == 'b') type = BISHOP;
            else if (lc == 'r') type = ROOK;
            else if (lc == 'q') type = QUEEN;
            else if (lc == 'k') type = KING;
            board[sq++] = type | (color << 3);
        }
    }
    
    turn = (fen[++i] == 'w') ? WHITE : BLACK;
    i += 2;
    castling_rights = 0;
    while (fen[i] != ' ') {
        if (fen[i] == 'K') castling_rights |= 1;
        if (fen[i] == 'Q') castling_rights |= 2;
        if (fen[i] == 'k') castling_rights |= 4;
        if (fen[i] == 'q') castling_rights |= 8;
        i++;
    }
    i++;
    if (fen[i] == '-') { ep_square = -1; i++; }
    else {
        int file = fen[i++] - 'a';
        int rank = fen[i++] - '1';
        ep_square = rank * 8 + file;
    }
    i++;
    
    size_t space_pos = fen.find(' ', i);
    if (space_pos != std::string::npos) {
        half_move_clock = std::stoi(fen.substr(i, space_pos - i));
        full_move_number = std::stoi(fen.substr(space_pos + 1));
    } else {
        half_move_clock = 0;
        full_move_number = 1;
    }
    history.clear();
}

std::string Position::getFEN() const {
    std::string fen = "";
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            if (board[sq] == 0) empty++;
            else {
                if (empty > 0) { fen += std::to_string(empty); empty = 0; }
                int pc = board[sq];
                char c = '?';
                int type = pc & 7;
                if (type == PAWN) c = 'p';
                else if (type == KNIGHT) c = 'n';
                else if (type == BISHOP) c = 'b';
                else if (type == ROOK) c = 'r';
                else if (type == QUEEN) c = 'q';
                else if (type == KING) c = 'k';
                if ((pc >> 3) == WHITE) c = std::toupper(c);
                fen += c;
            }
        }
        if (empty > 0) fen += std::to_string(empty);
        if (rank > 0) fen += '/';
    }
    fen += (turn == WHITE) ? " w " : " b ";
    
    std::string castling = "";
    if (castling_rights & 1) castling += "K";
    if (castling_rights & 2) castling += "Q";
    if (castling_rights & 4) castling += "k";
    if (castling_rights & 8) castling += "q";
    fen += castling.empty() ? "-" : castling;
    
    fen += " ";
    if (ep_square == -1) fen += "-";
    else {
        fen += (char)('a' + (ep_square % 8));
        fen += (char)('1' + (ep_square / 8));
    }
    fen += " " + std::to_string(half_move_clock) + " " + std::to_string(full_move_number);
    return fen;
}

void Position::addPromotions(std::vector<Move>& moves, int from, int to) const {
    moves.push_back(Move(from, to, 11)); // Q
    moves.push_back(Move(from, to, 10)); // R
    moves.push_back(Move(from, to, 9));  // B
    moves.push_back(Move(from, to, 8));  // N
}

std::vector<Move> Position::pseudoLegalMoves() const {
    std::vector<Move> moves;
    moves.reserve(256);

    for (int sq = 0; sq < 64; ++sq) {
        if (board[sq] == 0 || (board[sq] >> 3) != turn) continue;
        
        int type = board[sq] & 7;
        
        if (type == PAWN) {
            int dir = (turn == WHITE) ? 8 : -8;
            int promo_rank = (turn == WHITE) ? 7 : 0;
            
            // Push
            if (board[sq + dir] == 0) {
                if ((sq + dir) / 8 == promo_rank) addPromotions(moves, sq, sq + dir);
                else moves.push_back(Move(sq, sq + dir, 0));
                
                // Double Push
                if (((turn == WHITE && sq / 8 == 1) || (turn == BLACK && sq / 8 == 6)) && board[sq + 2 * dir] == 0) {
                    moves.push_back(Move(sq, sq + 2 * dir, 1));
                }
            }
            
            // Captures
            for (int offset : {-1, 1}) {
                int to = sq + dir + offset;
                if (to >= 0 && to < 64 && std::abs((sq % 8) - (to % 8)) == 1) {
                    if (board[to] != 0 && (board[to] >> 3) != turn) {
                        if (to / 8 == promo_rank) addPromotions(moves, sq, to);
                        else moves.push_back(Move(sq, to, 4));
                    } else if (to == ep_square) {
                        moves.push_back(Move(sq, to, 5));
                    }
                }
            }
        }
        else if (type == KNIGHT) {
            const int n_off[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
            for (int off : n_off) {
                int to = sq + off;
                if (to >= 0 && to < 64 && std::abs((sq % 8) - (to % 8)) <= 2) {
                    if (board[to] == 0 || (board[to] >> 3) != turn) moves.push_back(Move(sq, to, board[to] ? 4 : 0));
                }
            }
        }
        else if (type == KING) {
            const int k_off[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
            for (int off : k_off) {
                int to = sq + off;
                if (to >= 0 && to < 64 && std::abs((sq % 8) - (to % 8)) <= 1) {
                    if (board[to] == 0 || (board[to] >> 3) != turn) moves.push_back(Move(sq, to, board[to] ? 4 : 0));
                }
            }
            // Castling
            if (turn == WHITE) {
                if ((castling_rights & 1) && board[5] == 0 && board[6] == 0) {
                    if (!isSquareAttacked(4, BLACK) && !isSquareAttacked(5, BLACK) && !isSquareAttacked(6, BLACK))
                        moves.push_back(Move(4, 6, 2));
                }
                if ((castling_rights & 2) && board[1] == 0 && board[2] == 0 && board[3] == 0) {
                    if (!isSquareAttacked(4, BLACK) && !isSquareAttacked(3, BLACK) && !isSquareAttacked(2, BLACK))
                        moves.push_back(Move(4, 2, 3));
                }
            } else {
                if ((castling_rights & 4) && board[61] == 0 && board[62] == 0) {
                    if (!isSquareAttacked(60, WHITE) && !isSquareAttacked(61, WHITE) && !isSquareAttacked(62, WHITE))
                        moves.push_back(Move(60, 62, 2));
                }
                if ((castling_rights & 8) && board[57] == 0 && board[58] == 0 && board[59] == 0) {
                    if (!isSquareAttacked(60, WHITE) && !isSquareAttacked(59, WHITE) && !isSquareAttacked(58, WHITE))
                        moves.push_back(Move(60, 58, 3));
                }
            }
        }
        else { // BISHOP, ROOK, QUEEN
            const int p_dirs[8] = {-9, -7, 7, 9, -8, -1, 1, 8};
            int start = (type == ROOK) ? 4 : 0;
            int end = (type == BISHOP) ? 4 : 8;
            
            for (int i = start; i < end; ++i) {
                int curr = sq;
                while (true) {
                    int next = curr + p_dirs[i];
                    if (next < 0 || next >= 64 || std::abs((curr % 8) - (next % 8)) > 1) break;
                    curr = next;
                    if (board[curr] == 0) moves.push_back(Move(sq, curr, 0));
                    else {
                        if ((board[curr] >> 3) != turn) moves.push_back(Move(sq, curr, 4));
                        break;
                    }
                }
            }
        }
    }
    return moves;
}
std::vector<float> Position::toTensor() const {
    // 18 planes * 64 squares = 1152 elements initialized to 0.0
    std::vector<float> tensor(1152, 0.0f);
    
    // Planes 0-11: Pieces
    for (int sq = 0; sq < 64; ++sq) {
        if (board[sq] != 0) {
            int pc = board[sq];
            int type = pc & 7;
            int color = pc >> 3;
            // Map piece types (PAWN=1 -> 0, KNIGHT=2 -> 1, etc.)
            int plane = (color == WHITE ? 0 : 6) + (type - 1);
            
            int rank = sq / 8;
            int file = sq % 8;
            int row = 7 - rank;
            int col = file;
            
            tensor[plane * 64 + row * 8 + col] = 1.0f;
        }
    }
    
    // Plane 12: Side to Move
    float turn_val = (turn == WHITE) ? 1.0f : 0.0f;
    for (int i = 0; i < 64; ++i) tensor[12 * 64 + i] = turn_val;
    
    // Planes 13-16: Castling Rights
    float wk = (castling_rights & 1) ? 1.0f : 0.0f;
    float wq = (castling_rights & 2) ? 1.0f : 0.0f;
    float bk = (castling_rights & 4) ? 1.0f : 0.0f;
    float bq = (castling_rights & 8) ? 1.0f : 0.0f;
    for (int i = 0; i < 64; ++i) {
        tensor[13 * 64 + i] = wk;
        tensor[14 * 64 + i] = wq;
        tensor[15 * 64 + i] = bk;
        tensor[16 * 64 + i] = bq;
    }
    
    // Plane 17: En Passant
    if (ep_square != -1) {
        int rank = ep_square / 8;
        int file = ep_square % 8;
        int row = 7 - rank;
        int col = file;
        tensor[17 * 64 + row * 8 + col] = 1.0f;
    }
    
    return tensor;
}
std::vector<Move> Position::legalMoves() {
    std::vector<Move> pseudo = pseudoLegalMoves();
    std::vector<Move> legal;
    for (Move m : pseudo) {
        makeMove(m);
        if (!isSquareAttacked(kingSquare(turn ^ 1), turn)) {
            legal.push_back(m);
        }
        undoMove();
    }
    return legal;
}

int Position::kingSquare(int color) const {
    for (int i = 0; i < 64; ++i) {
        if (board[i] == (KING | (color << 3))) return i;
    }
    return -1;
}

bool Position::isSquareAttacked(int sq, int by_color) const {
    int pawn_dir = (by_color == WHITE) ? -8 : 8;
    for (int offset : {-1, 1}) {
        int p_sq = sq + pawn_dir + offset;
        if (p_sq >= 0 && p_sq < 64 && std::abs((sq % 8) - (p_sq % 8)) == 1) {
            if (board[p_sq] == (PAWN | (by_color << 3))) return true;
        }
    }
    
    const int n_off[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (int off : n_off) {
        int n_sq = sq + off;
        if (n_sq >= 0 && n_sq < 64 && std::abs((sq % 8) - (n_sq % 8)) <= 2) {
            if (board[n_sq] == (KNIGHT | (by_color << 3))) return true;
        }
    }
    
    const int k_off[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int off : k_off) {
        int k_sq = sq + off;
        if (k_sq >= 0 && k_sq < 64 && std::abs((sq % 8) - (k_sq % 8)) <= 1) {
            if (board[k_sq] == (KING | (by_color << 3))) return true;
        }
    }
    
    const int dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int i = 0; i < 8; ++i) {
        int curr = sq;
        while (true) {
            int next = curr + dirs[i];
            if (next < 0 || next >= 64 || std::abs((curr % 8) - (next % 8)) > 1) break;
            curr = next;
            if (board[curr] != 0) {
                int pc = board[curr];
                if ((pc >> 3) == by_color) {
                    int pt = pc & 7;
                    if (pt == QUEEN) return true;
                    if (pt == ROOK && (dirs[i] == -8 || dirs[i] == 8 || dirs[i] == -1 || dirs[i] == 1)) return true;
                    if (pt == BISHOP && (dirs[i] == -9 || dirs[i] == -7 || dirs[i] == 7 || dirs[i] == 9)) return true;
                }
                break;
            }
        }
    }
    return false;
}

void Position::makeMove(Move m) {
    State st = {m.data, board[m.to()], static_cast<uint8_t>(ep_square == -1 ? 255 : ep_square), static_cast<uint8_t>(castling_rights), static_cast<uint16_t>(half_move_clock)};
    history.push_back(st);

    int from = m.from(), to = m.to(), flag = m.flags();

    half_move_clock++;
    if ((board[from] & 7) == PAWN || board[to] != 0) half_move_clock = 0;

    ep_square = -1;
    if (flag == 1) ep_square = (turn == WHITE) ? from + 8 : from - 8;

    board[to] = board[from];
    board[from] = 0;

    if (flag == 5) {
        int cap = (turn == WHITE) ? to - 8 : to + 8;
        history.back().captured_piece = board[cap];
        board[cap] = 0;
    } 
    else if (flag == 2) {
        int r_from = (turn == WHITE) ? 7 : 63, r_to = (turn == WHITE) ? 5 : 61;
        board[r_to] = board[r_from]; board[r_from] = 0;
    } 
    else if (flag == 3) {
        int r_from = (turn == WHITE) ? 0 : 56, r_to = (turn == WHITE) ? 3 : 59;
        board[r_to] = board[r_from]; board[r_from] = 0;
    } 
    else if (flag >= 8) {
        int pt = PAWN;
        if (flag == 8) pt = KNIGHT; else if (flag == 9) pt = BISHOP;
        else if (flag == 10) pt = ROOK; else if (flag == 11) pt = QUEEN;
        board[to] = pt | (turn << 3);
    }

    castling_rights &= CASTLING_MASKS[from];
    castling_rights &= CASTLING_MASKS[to];
    
    turn ^= 1;
    if (turn == WHITE) full_move_number++;
}

void Position::undoMove() {
    State st = history.back();
    history.pop_back();

    turn ^= 1;
    if (turn == BLACK) full_move_number--; // Revert the increment that happened when making black's move

    Move m(st.move);
    int from = m.from(), to = m.to(), flag = m.flags();

    ep_square = (st.ep_square == 255) ? -1 : st.ep_square;
    castling_rights = st.castling_rights;
    half_move_clock = st.half_move_clock;

    board[from] = board[to];
    board[to] = st.captured_piece; 

    if (flag == 5) { // EP Capture Undo
        board[to] = 0; // The EP target square was empty
        int cap = (turn == WHITE) ? to - 8 : to + 8;
        board[cap] = st.captured_piece;
    } 
    else if (flag == 2) { // K Castle
        int r_from = (turn == WHITE) ? 7 : 63, r_to = (turn == WHITE) ? 5 : 61;
        board[r_from] = board[r_to]; board[r_to] = 0;
    } 
    else if (flag == 3) { // Q Castle
        int r_from = (turn == WHITE) ? 0 : 56, r_to = (turn == WHITE) ? 3 : 59;
        board[r_from] = board[r_to]; board[r_to] = 0;
    } 
    else if (flag >= 8) { // Undo Promo
        board[from] = PAWN | (turn << 3);
    }
}

bool Position::isCheck() const { return isSquareAttacked(kingSquare(turn), turn ^ 1); }
bool Position::isCheckmate() { return isCheck() && legalMoves().empty(); }
bool Position::isStalemate() { return !isCheck() && legalMoves().empty(); }
bool Position::isDraw() const { return half_move_clock >= 100; }
bool Position::isTerminal() { return isCheckmate() || isStalemate() || isDraw(); }
std::string Position::result() {
    if (isCheckmate()) return turn == WHITE ? "0-1" : "1-0";
    if (isStalemate() || isDraw()) return "1/2-1/2";
    return "*";
}

uint64_t Position::perft(int depth) {
    if (depth == 0) return 1;
    std::vector<Move> moves = legalMoves();
    if (depth == 1) return moves.size();

    uint64_t nodes = 0;
    for (Move m : moves) {
        makeMove(m);
        nodes += perft(depth - 1);
        undoMove();
    }
    return nodes;
}