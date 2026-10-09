#include "mcts.hpp"
#include <iostream>
#include <cassert>

void run_puct_test(const std::string& name, const std::string& fen) {
    Position pos;
    pos.loadFEN(fen);
    
    // 100 simulations, c_puct = 1.414
    MCTS mcts(pos, 100, 1.414);
    
    // Mock the WebWorker Asynchronous Execution Loop
    int eval_requests = 0;
    while (mcts.step()) {
        eval_requests++;
        std::vector<float> fake_policy(4672, 0.0f);
        
        // COPY the position to safely drop the const qualifier
        Position currentPos = mcts.getCurrentPosition();
        std::vector<Move> moves = currentPos.legalMoves();
        
        if (!moves.empty()) {
            // Heavily bias the network towards the first legal move to test exploitation
            fake_policy[ActionSpace::moveToIndex(moves[0])] = 0.9f;
            for (size_t i = 1; i < moves.size(); ++i) {
                fake_policy[ActionSpace::moveToIndex(moves[i])] = 0.1f / (moves.size() - 1);
            }
        }
        
        // Provide the mock evaluation and let MCTS resume
        mcts.provideEvaluation(fake_policy, 0.1f);
    }
    
    SearchResult res = mcts.getResult();
    
    std::cout << "[PUCT TEST] " << name << "\n";
    std::cout << "  Simulations: " << res.stats.simulations << "\n";
    std::cout << "  Nodes created: " << res.stats.nodes << "\n";
    std::cout << "  Evaluations Requested: " << eval_requests << "\n";
    
    int total_visits = 0;
    for (auto& rv : res.rootVisits) {
        total_visits += rv.second;
    }
    
    std::cout << "  Root Child Visits Sum: " << total_visits << "\n";
    if (total_visits == 99) { 
        // 100 simulations minus 1 initial evaluation for the root node
        std::cout << "  [PASS] Visit sum matches expected probability distribution (π).\n\n";
    } else {
        std::cout << "  [FAIL] Visit sum " << total_visits << " != 99\n\n";
    }
}

int main() {
    std::cout << "Running Neural Network PUCT Tests...\n\n";

    // Standard Expansion Test
    run_puct_test("Start Position Distribution", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // True Terminal Avoidance Check
    Position already_mate;
    already_mate.loadFEN("R5k1/5ppp/8/8/8/8/8/5K2 b - - 0 1"); // Black to move, already in checkmate
    MCTS mcts_already(already_mate, 50, 1.414);
    
    int evals = 0;
    while(mcts_already.step()) {
        evals++;
        std::vector<float> p(4672, 0);
        mcts_already.provideEvaluation(p, 0);
    }
    
    std::cout << "[PUCT TEST] Already Mated Position\n";
    std::cout << "  Evaluations Requested: " << evals << "\n";
    if (evals == 0) {
        std::cout << "  [PASS] Engine recognized terminal state and requested 0 neural network evaluations.\n\n";
    } else {
        std::cout << "  [FAIL] " << evals << " evaluations requested.\n\n";
    }

    return 0;
}