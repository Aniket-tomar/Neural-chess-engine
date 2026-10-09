#ifndef BOARD_HPP
#define BOARD_HPP

#include <cstdint>
#include <string>

// Enums mapped to integers for shared C++/Python encoding
enum Color { WHITE = 0, BLACK = 1, COLOR_NONE = 2 };
enum PieceType { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, PIECE_NONE = 6 };

struct Move {
    uint16_t data;
    
    Move(uint16_t d) : data(d) {}
    Move(int from, int to, int flags = 0) {
        data = (from & 0x3F) | ((to & 0x3F) << 6) | ((flags & 0xF) << 12);
    }
    
    int from() const { return data & 0x3F; }
    int to() const { return (data >> 6) & 0x3F; }
    int flags() const { return (data >> 12) & 0xF; }
};

class Board {
public:
    uint64_t piece_bbs[2][6]; // [Color][PieceType]
    uint64_t occupancy[2];    // [Color]
    uint64_t empty_bb;
    
    Color side_to_move;
    uint8_t castling_rights; // 4 bits: WK, WQ, BK, BQ
    uint8_t en_passant_sq;   // 0-63, or 64 if none
    int half_move_clock;
    
    Board(); // Initializes standard starting position
    
    // Core methods to be implemented
    void load_fen(const std::string& fen);
    std::string get_fen() const;
    void make_move(Move m);
};

#endif