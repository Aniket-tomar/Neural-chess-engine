@echo off
echo Compiling C++ to WebAssembly via Emscripten...
em++ -std=c++20 ../chess.cpp ../action_space.cpp ../mcts.cpp api.cpp -o chess_engine.js ^
    --bind ^
    -s MODULARIZE=1 ^
    -s EXPORT_NAME="createChessModule" ^
    -s WASM=1 ^
    -s ALLOW_MEMORY_GROWTH=1 ^
    -O3

echo Build complete! Generated chess_engine.js and chess_engine.wasm