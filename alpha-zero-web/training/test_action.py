from action_space import ActionSpace

test_cases = [
    {
        # Position with castling, pawn promotions, and knight jumps
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
        "moves": [
            {'from': 60, 'to': 62, 'flag': 2},  # Black Kingside Castle
            {'from': 60, 'to': 58, 'flag': 3},  # Black Queenside Castle
            {'from': 9,  'to': 0,  'flag': 11}, # Pawn b2xa1=Q (Capture + Promo)
            {'from': 9,  'to': 0,  'flag': 8},  # Pawn b2xa1=N (Capture + Underpromo)
            {'from': 45, 'to': 35, 'flag': 0},  # Knight f6-d5
        ]
    },
    {
        # Position where White just played c2-c4, enabling En Passant for Black on b4
        "fen": "rnbqkbnr/p1pppppp/8/8/1pP5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
        "moves": [
            {'from': 25, 'to': 18, 'flag': 5},  # En passant capture (b4xc3)
        ]
    },
    {
        # Standard starting position for double pawn pushes
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": [
            {'from': 12, 'to': 28, 'flag': 1},  # Pawn double push e2-e4
        ]
    }
]

print("Python Action Space Round-Trip Test")
passed = 0
total = 0

for tc in test_cases:
    fen = tc["fen"]
    for m in tc["moves"]:
        total += 1
        idx = ActionSpace.move_to_index(m['from'], m['to'], m['flag'])
        decoded = ActionSpace.index_to_move(idx, fen)
        
        if m['from'] == decoded['from'] and m['to'] == decoded['to'] and m['flag'] == decoded['flag']:
            passed += 1
        else:
            print(f"FAIL in FEN {fen}\n  Expected {m}\n  Got {decoded}")

print(f"{passed}/{total} moves successfully round-tripped.")