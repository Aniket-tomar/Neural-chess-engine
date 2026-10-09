class ActionSpace:
    q_dx = [0, 1, 1, 1, 0, -1, -1, -1]
    q_dy = [1, 1, 0, -1, -1, -1, 0, 1]
    k_dx = [1, 2, 2, 1, -1, -2, -2, -1]
    k_dy = [2, 1, -1, -2, -2, -1, 1, 2]

    @staticmethod
    def sign(x):
        return (x > 0) - (x < 0)

    @classmethod
    def move_to_index(cls, from_sq, to_sq, flag):
        dx = (to_sq % 8) - (from_sq % 8)
        dy = (to_sq // 8) - (from_sq // 8)
        
        # Underpromotions (Knight=8, Bishop=9, Rook=10)
        if 8 <= flag <= 10:
            type_idx = flag - 8
            dir_idx = 0 if dx == -1 else (1 if dx == 0 else 2)
            plane = 64 + type_idx * 3 + dir_idx
        else:
            is_knight = False
            for i in range(8):
                if cls.k_dx[i] == dx and cls.k_dy[i] == dy:
                    plane = 56 + i
                    is_knight = True
                    break
            
            if not is_knight:
                sx = cls.sign(dx)
                sy = cls.sign(dy)
                dir_idx = -1
                for i in range(8):
                    if cls.q_dx[i] == sx and cls.q_dy[i] == sy:
                        dir_idx = i
                        break
                dist = max(abs(dx), abs(dy))
                plane = dir_idx * 7 + (dist - 1)
                
        return from_sq * 73 + plane

    @classmethod
    def index_to_move(cls, index, fen):
        from_sq = index // 73
        plane = index % 73
        flag = 0
        
        if plane < 56:
            dir_idx = plane // 7
            dist = (plane % 7) + 1
            dx = cls.q_dx[dir_idx] * dist
            dy = cls.q_dy[dir_idx] * dist
        elif plane < 64:
            k_idx = plane - 56
            dx = cls.k_dx[k_idx]
            dy = cls.k_dy[k_idx]
        else:
            up_idx = plane - 64
            type_idx = up_idx // 3
            dir_idx = up_idx % 3
            dx = dir_idx - 1
            dy = 1 if (from_sq // 8 == 6) else -1
            flag = 8 + type_idx
            
        to_sq = from_sq + dy * 8 + dx
        
        if flag == 0:
            parts = fen.split(' ')
            board_part, turn = parts[0], parts[1]
            ep_str = parts[3]
            ep_sq = -1 if ep_str == "-" else (ord(ep_str[0]) - ord('a')) + (int(ep_str[1]) - 1) * 8
            
            board = [None] * 64
            sq = 56
            for c in board_part:
                if c == '/': sq -= 16
                elif c.isdigit(): sq += int(c)
                else:
                    board[sq] = c
                    sq += 1
                    
            p_char = board[from_sq].lower() if board[from_sq] else ''
            to_char = board[to_sq]
            
            if p_char == 'p':
                if abs(dy) == 2: flag = 1
                elif to_sq == ep_sq: flag = 5
                elif to_sq // 8 == 0 or to_sq // 8 == 7: flag = 11
                elif to_char is not None: flag = 4
            elif p_char == 'k':
                if dx == 2: flag = 2
                elif dx == -2: flag = 3
                elif to_char is not None: flag = 4
            else:
                if to_char is not None: flag = 4
                
        return {'from': from_sq, 'to': to_sq, 'flag': flag}