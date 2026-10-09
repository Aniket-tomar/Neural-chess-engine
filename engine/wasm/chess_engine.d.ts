export interface MoveJS {
    from: number;
    to: number;
    flags: number;
}

export interface VectorMoveJS {
    size(): number;
    get(index: number): MoveJS;
    delete(): void;
}

export interface ChessEngine {
    loadFEN(fen: string): void;
    getFEN(): string;
    getLegalMoves(): VectorMoveJS;
    makeMove(from: number, to: number, flags: number): void;
    undoMove(): void;
    isCheck(): boolean;
    isCheckmate(): boolean;
    isStalemate(): boolean;
    isDraw(): boolean;
    result(): string;
    delete(): void;
}

export interface ChessModule {
    ChessEngine: new () => ChessEngine;
}

export default function createChessModule(): Promise<ChessModule>;