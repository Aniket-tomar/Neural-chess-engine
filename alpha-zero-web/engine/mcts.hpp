#pragma once
#include "chess.hpp"
#include "action_space.hpp"
#include <vector>
#include <memory>
#include <utility>

struct SearchStats {
    int simulations;
    int nodes;
    double best_q;
};

struct SearchResult {
    Move bestMove;
    std::vector<std::pair<Move, int>> rootVisits; // Used to derive policy target π
    SearchStats stats;
};

class Node {
public:
    Move move;
    Node* parent;
    std::vector<std::unique_ptr<Node>> children;
    int visits;
    double valueSum;
    double prior; // Neural network P(s, a)
    int turn; 

    Node(Move m, Node* p, int t, double pr = 0.0) 
        : move(m), parent(p), visits(0), valueSum(0.0), prior(pr), turn(t) {}
    
    double getQ() const { return visits == 0 ? 0.0 : valueSum / visits; }
};

class MCTS {
public:
    // Configurable iterations and exploration constant (c_puct)
    MCTS(Position pos, int max_sims, float puct_c);
    
    // State machine controls
    bool step(); // Returns TRUE if it needs a Neural Network evaluation
    void provideEvaluation(const std::vector<float>& policy, float value);
    
    // Getters
    SearchResult getResult();
    std::vector<float> getEvalTensor() const { return currentPos.toTensor(); }
    const Position& getCurrentPosition() const { return currentPos; }
    
private:
    std::unique_ptr<Node> root;
    Position rootPos;
    int maxSimulations;
    float c_puct;
    int simulations;
    int totalNodes;
    
    Node* currentNode;
    Position currentPos;
    
    enum State { SELECTING, AWAITING_EVAL, SEARCH_COMPLETE };
    State state;

    void select();
    void backup(float value);
};