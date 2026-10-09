#include "chess.hpp"
#include <iostream>
#include <string>

void run_perft(const std::string& name, const std::string& fen, int depth, uint64_t expected) {
    Position pos;
    pos.loadFEN(fen);
    uint64_t nodes = pos.perft(depth);
    if (nodes == expected) {
        std::cout << "[PASS] " << name << " Depth " << depth << " -> " << nodes << "\n";
    } else {
        std::cout << "[FAIL] " << name << " Depth " << depth 
                  << " -> Expected: " << expected << ", Got: " << nodes << "\n";
    }
}

int main() {
    std::cout << "Running PERFT validations...\n\n";

    // Standard starting position
    std::string startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    run_perft("Start Position", startpos, 1, 20);
    run_perft("Start Position", startpos, 2, 400);
    run_perft("Start Position", startpos, 3, 8902);
    run_perft("Start Position", startpos, 4, 197281);

    // Kiwipete - Focus on discovered attacks, castling, and mobility
    std::string kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    run_perft("Kiwipete", kiwipete, 1, 48);
    run_perft("Kiwipete", kiwipete, 2, 2039);
    run_perft("Kiwipete", kiwipete, 3, 97862);

    // Position 3 - Focus on checks, pins, and pawn promotions
    std::string pos3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    run_perft("Position 3", pos3, 1, 14);
    run_perft("Position 3", pos3, 2, 191);
    run_perft("Position 3", pos3, 3, 2812);

    // Position 4 - Focus on en passant and king safety
    std::string pos4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    run_perft("Position 4", pos4, 1, 6);
    run_perft("Position 4", pos4, 2, 264);

    return 0;
}