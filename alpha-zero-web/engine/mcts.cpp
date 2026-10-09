#include "mcts.hpp"
#include <cmath>

MCTS::MCTS(Position pos, int max_sims, float puct_c)
    : rootPos(pos), maxSimulations(max_sims), c_puct(puct_c), simulations(0), totalNodes(1),
      currentPos(pos), state(SELECTING) {
    root = std::make_unique<Node>(Move(0), nullptr, pos.getTurn());
    currentNode = root.get();
}

void MCTS::select() {
    currentNode = root.get();
    currentPos = rootPos;
    
    // Traverse down to a leaf node
    while (!currentNode->children.empty()) {
        double bestScore = -1e9;
        Node* bestChild = nullptr;
        
        double sqrt_parent_visits = std::sqrt(currentNode->visits);
        
        for (auto& child : currentNode->children) {
            double q = child->getQ();
            // PUCT Formula: U = c * P * sqrt(N_parent) / (1 + N_child)
            double u = c_puct * child->prior * sqrt_parent_visits / (1.0 + child->visits);
            double score = q + u;
            
            if (score > bestScore) {
                bestScore = score;
                bestChild = child.get();
            }
        }
        currentNode = bestChild;
        currentPos.makeMove(currentNode->move);
    }
}

void MCTS::backup(float value) {
    Node* node = currentNode;
    double v = value;
    while (node != nullptr) {
        node->visits++;
        node->valueSum += v;
        v = -v; // Invert the value for the opposing player at the parent node
        node = node->parent;
    }
}

bool MCTS::step() {
    while (simulations < maxSimulations) {
        if (state == SELECTING) {
            select();
            
            if (currentPos.isTerminal()) {
                double value = 0.0;
                if (currentPos.isCheckmate()) {
                    // The player whose turn it is has no moves and is in check. They have lost.
                    value = -1.0; 
                }
                backup(value);
                simulations++;
            } else {
                // Yield execution: Ask JavaScript/ONNX for policy and value
                state = AWAITING_EVAL;
                return true; 
            }
        }
    }
    state = SEARCH_COMPLETE;
    return false;
}

void MCTS::provideEvaluation(const std::vector<float>& policy, float value) {
    std::vector<Move> legalMoves = currentPos.legalMoves();
    
    double sumPrior = 0.0;
    std::vector<double> priors;
    priors.reserve(legalMoves.size());
    
    // Extract network probabilities using Action Space encoding
    for (Move m : legalMoves) {
        int idx = ActionSpace::moveToIndex(m);
        double p = policy[idx];
        priors.push_back(p);
        sumPrior += p;
    }
    
    // Mask illegal moves implicitly by only creating children for legal moves
    // Normalize probabilities so they sum to 1.0 locally
    for (size_t i = 0; i < legalMoves.size(); ++i) {
        double normalized_p = (sumPrior > 1e-6) ? (priors[i] / sumPrior) : (1.0 / legalMoves.size());
        currentNode->children.push_back(std::make_unique<Node>(
            legalMoves[i], currentNode, currentPos.getTurn() ^ 1, normalized_p
        ));
        totalNodes++;
    }
    
    backup(value);
    simulations++;
    state = SELECTING;
}

SearchResult MCTS::getResult() {
    Move bestMove(0);
    int maxVisits = -1;
    double bestQ = 0.0;
    std::vector<std::pair<Move, int>> rootVisits;
    
    for (auto& child : root->children) {
        rootVisits.push_back({child->move, child->visits});
        if (child->visits > maxVisits) {
            maxVisits = child->visits;
            bestMove = child->move;
            bestQ = child->getQ();
        }
    }
    
    SearchStats stats{simulations, totalNodes, bestQ};
    return {bestMove, rootVisits, stats};
}