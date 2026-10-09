#include "mcts.hpp"
#include <iostream>
#include <cassert>

void run_test(const std::string& name, const std::string& fen, int expected_from, int expected_to) {
    Position pos;
    pos.loadFEN(fen);
    
    // 1000 simulations guarantees exploring the winning terminal branch
    SearchResult res = MCTS::search(pos, 1000); 
    
    std::cout << "[TEST] " << name << "\n";
    std::cout << "  Simulations: " << res.stats.simulations << "\n";
    std::cout << "  Nodes: " << res.stats.nodes << "\n";
    std::cout << "  Best Q: " << res.stats.best_q << "\n";
    
    bool passed = (res.bestMove.from() == expected_from && res.bestMove.to() == expected_to);
    if (passed) {
        std::cout << "  [PASS] Found expected move: " << expected_from << "->" << expected_to << "\n\n";
    } else {
        std::cout << "  [FAIL] Expected " << expected_from << "->" << expected_to 
                  << ", Got " << res.bestMove.from() << "->" << res.bestMove.to() << "\n\n";
    }
}

int main() {
    std::cout << "Running MCTS Unit Tests...\n\n";

    // 1. White to move: Back-rank mate Ra8# (Square 0 -> 56)
    // Black pawns on f7, g7, h7 trap the King on g8
    run_test("White Mate in 1 (Perspective + Terminal)", "6k1/5ppp/8/8/8/8/8/R4K2 w - - 0 1", 0, 56);

    // 2. Black to move: Back-rank mate Ra1# (Square 56 -> 0)
    // White pawns on f2, g2, h2 trap the King on g1
    run_test("Black Mate in 1 (Perspective + Terminal)", "r4k2/8/8/8/8/8/5PPP/6K1 b - - 0 1", 56, 0);

    // 3. Legal Moves & Expansion Node Validation
    Position pos3;
    pos3.loadFEN("8/8/8/8/8/8/4k3/4K3 w - - 0 1"); 
    SearchResult res3 = MCTS::search(pos3, 50);
    
    int total_child_visits = 0;
    for (auto& rv : res3.rootVisits) total_child_visits += rv.second;
    
    std::cout << "[TEST] Visit Counts & Expansion\n";
    if (total_child_visits == 50) {
        std::cout << "  [PASS] Child visits (" << total_child_visits << ") identically match simulations (50)\n\n";
    } else {
        std::cout << "  [FAIL] Child visits: " << total_child_visits << " != 50\n\n";
    }

    return 0;
}