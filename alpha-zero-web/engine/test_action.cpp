#include "action_space.hpp"
#include "chess.hpp"
#include <iostream>
#include <cassert>

int main() {
    Position pos;
    // Edge case position testing all types of moves
    pos.loadFEN("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
    
    std::vector<Move> moves = pos.legalMoves();
    int passed = 0;
    
    std::cout << "C++ Action Space Round-Trip Test\n";
    for (Move m : moves) {
        int idx = ActionSpace::moveToIndex(m);
        Move m2 = ActionSpace::indexToMove(idx, pos);
        
        if (m.from() == m2.from() && m.to() == m2.to() && m.flags() == m2.flags()) passed++;
        else {
            std::cout << "FAIL: from " << m.from() << " to " << m.to() << " flag " << m.flags() << "\n";
            std::cout << "Decoded: from " << m2.from() << " to " << m2.to() << " flag " << m2.flags() << "\n";
        }
    }
    
    std::cout << passed << "/" << moves.size() << " moves successfully round-tripped.\n";
    return 0;
}