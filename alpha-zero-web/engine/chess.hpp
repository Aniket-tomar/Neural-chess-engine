#pragma once

#include <vector>
#include <string>
#include <cstdint>

enum Color { WHITE = 0, BLACK = 1 };
enum PieceType { NONE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };

struct Move {
    uint16_t data;

    Move(uint16_t d = 0) : data(d) {}
    Move(int from, int to, int flags = 0) {
        data = (from & 0x3F) | ((to & 0x3F) << 6) | ((flags & 0xF) << 12);
    }

    int from() const { return data & 0x3F; }
    int to() const { return (data >> 6) & 0x3F; }
    int flags() const { return (data >> 12) & 0xF; }
};

struct State {
    uint16_t move;
    uint8_t captured_piece;
    uint8_t ep_square;
    uint8_t castling_rights;
    uint16_t half_move_clock;
};

class Position {
public:
    Position();
    void loadFEN(const std::string& fen);
    std::string getFEN() const;
    std::vector<float> toTensor() const;

    std::vector<Move> legalMoves();
    void makeMove(Move move);
    void undoMove();

    bool isCheck() const;
    bool isCheckmate();
    bool isStalemate();
    bool isDraw() const;
    bool isTerminal();
    std::string result();  
    int getTurn() const { return turn; } 

    uint64_t perft(int depth);

private:
    uint8_t board[64];
    int turn;
    int ep_square;
    int castling_rights; // Bitmask: 1=WK, 2=WQ, 4=BK, 8=BQ
    int half_move_clock;
    int full_move_number;
    std::vector<State> history;

    std::vector<Move> pseudoLegalMoves() const;
    bool isSquareAttacked(int sq, int by_color) const;
    int kingSquare(int color) const;
    void addPromotions(std::vector<Move>& moves, int from, int to) const;
};