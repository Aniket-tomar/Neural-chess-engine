#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "../mcts.hpp"
#include "../chess.hpp"

using namespace emscripten;

// A simple structure to pass move data safely to JavaScript
struct MoveJS {
    int from;
    int to;
    int flags;
};

class ChessWasmAPI {
    Position pos;
    std::unique_ptr<MCTS> mcts;
public:
    ChessWasmAPI() {}

    std::vector<float> getTensor() const { 
    return pos.toTensor(); 
}

    void loadFEN(const std::string& fen) { pos.loadFEN(fen); }
    std::string getFEN() const { return pos.getFEN(); }

    // Converts C++ std::vector<Move> into an array of JS-friendly objects
    std::vector<MoveJS> getLegalMoves() {
        std::vector<MoveJS> result;
        for (Move m : pos.legalMoves()) {
            result.push_back({m.from(), m.to(), m.flags()});
        }
        return result;
    }

    void makeMove(int from, int to, int flags) {
        pos.makeMove(Move(from, to, flags));
    }

    void undoMove() { pos.undoMove(); }

    bool isCheck() const { return pos.isCheck(); }
    bool isCheckmate() { return pos.isCheckmate(); }
    bool isStalemate() { return pos.isStalemate(); }
    bool isDraw() const { return pos.isDraw(); }
    std::string result() { return pos.result(); }

    void startAnalysis(int max_sims, float puct_c) {
        mcts = std::make_unique<MCTS>(pos, max_sims, puct_c);
    }

    bool mctsStep() {
        if (mcts) return mcts->step();
        return false;
    }

    std::vector<float> mctsGetTensor() {
        if (mcts) return mcts->getEvalTensor();
        return std::vector<float>();
    }

    void mctsProvideEval(const std::vector<float>& policy, float value) {
        if (mcts) mcts->provideEvaluation(policy, value);
    }

    // Return the final SearchResult as a JavaScript object
    val mctsGetResult() {
        if (!mcts) return val::null();
        SearchResult res = mcts->getResult();
        
        val obj = val::object();
        
        val bestMove = val::object();
        bestMove.set("from", res.bestMove.from());
        bestMove.set("to", res.bestMove.to());
        bestMove.set("flags", res.bestMove.flags());
        
        obj.set("bestMove", bestMove);
        obj.set("simulations", res.stats.simulations);
        obj.set("nodes", res.stats.nodes);
        obj.set("bestQ", res.stats.best_q);
        
        return obj;
    }
};

EMSCRIPTEN_BINDINGS(chess_module) {
    value_object<MoveJS>("MoveJS")
        .field("from", &MoveJS::from)
        .field("to", &MoveJS::to)
        .field("flags", &MoveJS::flags);

    // Binds the C++ vector so it can be traversed in JS
    register_vector<MoveJS>("VectorMoveJS");
    register_vector<float>("VectorFloat");

    class_<ChessWasmAPI>("ChessEngine")
        .constructor<>()
        .function("loadFEN", &ChessWasmAPI::loadFEN)
        .function("getFEN", &ChessWasmAPI::getFEN)
        .function("getLegalMoves", &ChessWasmAPI::getLegalMoves)
        .function("makeMove", &ChessWasmAPI::makeMove)
        .function("undoMove", &ChessWasmAPI::undoMove)
        .function("isCheck", &ChessWasmAPI::isCheck)
        .function("isCheckmate", &ChessWasmAPI::isCheckmate)
        .function("isStalemate", &ChessWasmAPI::isStalemate)
        .function("isDraw", &ChessWasmAPI::isDraw)
        .function("result", &ChessWasmAPI::result)
        .function("getTensor", &ChessWasmAPI::getTensor)
        .function("startAnalysis", &ChessWasmAPI::startAnalysis)
        .function("mctsStep", &ChessWasmAPI::mctsStep)
        .function("mctsGetTensor", &ChessWasmAPI::mctsGetTensor)
        .function("mctsProvideEval", &ChessWasmAPI::mctsProvideEval)
        .function("mctsGetResult", &ChessWasmAPI::mctsGetResult);
}