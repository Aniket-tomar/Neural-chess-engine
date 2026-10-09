#include <iostream>
#include "chess.hpp"

int main() {
    Position pos;
    // Edge case: Black to move, en passant on c3, full castling
    std::string test_fen = "rnbqkbnr/p1pppppp/8/8/1pP5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1";
    pos.loadFEN(test_fen);
    
    std::vector<float> tensor = pos.toTensor();
    
    std::cout << "C++ Tensor Hash Test\nFEN: " << test_fen << "\n";
    int count = 0;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] > 0.0f) {
            std::cout << i << ":" << tensor[i] << " ";
            count++;
        }
    }
    std::cout << "\nTotal active floats: " << count << "\n";
    return 0;
}