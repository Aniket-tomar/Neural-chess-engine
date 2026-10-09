#include <iostream>
#include <cassert>
#include "board.hpp"

int main() {
    assert(sizeof(Move) == 2); // Must remain strictly 16-bit
    assert(sizeof(uint64_t) == 8); 
    
    Move m(8, 16); // A2 to A3
    assert(m.from() == 8);
    assert(m.to() == 16);
    
    std::cout << "Phase 1: Basic structures size and bitwise logic OK." << std::endl;
    return 0;
}