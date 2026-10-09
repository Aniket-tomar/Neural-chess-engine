#pragma once
#include "chess.hpp"

class ActionSpace {
public:
    static int moveToIndex(Move m);
    static Move indexToMove(int index, const Position& pos);
};