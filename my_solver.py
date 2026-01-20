import numpy as np
from time import sleep
from utils import get_board, get_state, attack, get_config, submit_ships, ready, get_ships_info
from utils import generate_ships_layout
from utils import ws_connect, wait_call
import asyncio


def conv2d(x, kernel):
    H_in, W_in = x.shape
    H_k, W_k = kernel.shape

    H_out = H_in - H_k + 1
    W_out = W_in - W_k + 1
    shape = (H_out, W_out, H_k, W_k)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[0],
        x.strides[1]
    )
        
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=shape,
        strides=strides
    )
    result = np.sum(windows * kernel, axis=(2, 3))
    return result

def generate_attack_map(board:np.ndarray, ships):
    res = np.zeros_like(board)
    if np.any(board == 3):
        print("Generating attack map based on hits...")
        poss = np.argwhere(board == 3)
        for x,y in poss:
            for size in ships:
                weights = ships[size][0]
                if weights == 0:
                    continue
                kernel = np.ones(size, dtype=int)
                left = right = x
                up = down = y
                for _ in range(size-1):
                    if left-1 < 0 or board[left-1, y] == 1:
                        break
                    if left-2 >=0:
                        if(board[left-2, y] == 3 or board[left-2, y] == 4):
                            break
                        if y-1 >=0 and (board[left-1, y-1] == 3 or board[left-1, y-1] == 4):
                                break
                        if y+1 < board.shape[1] and (board[left-1, y+1] == 3 or board[left-1, y+1] == 4):
                                break
                    left -= 1
                for _ in range(size-1):
                    if right+1 >= board.shape[0] or board[right+1, y] == 1:
                        break
                    if right+2 < board.shape[0]:
                        if(board[right+2, y] == 3 or board[right+2, y] == 4):
                            break
                        if y-1 >=0 and (board[right+1, y-1] == 3 or board[right+1, y-1] == 4):
                                break
                        if y+1 < board.shape[1] and (board[right+1, y+1] == 3 or board[right+1, y+1] == 4):
                                break
                    right += 1
                for _ in range(size-1):
                    if up-1 < 0 or board[x, up-1] == 1:
                        break
                    if up-2 >=0:
                        if(board[x, up-2] == 3 or board[x, up-2] == 4):
                            break
                        if x-1 >=0 and (board[x-1, up-1] == 3 or board[x-1, up-1] == 4):
                                break
                        if x+1 < board.shape[0] and (board[x+1, up-1] == 3 or board[x+1, up-1] == 4):
                                break
                    up -= 1
                for _ in range(size-1):
                    if down+1 >= board.shape[1] or board[x, down+1] == 1:
                        break
                    if down+2 < board.shape[1]:
                        if(board[x, down+2] == 3 or board[x, down+2] == 4):
                            break
                        if x-1 >=0 and (board[x-1, down+1] == 3 or board[x-1, down+1] == 4):
                                break
                        if x+1 < board.shape[0] and (board[x+1, down+1] == 3 or board[x+1, down+1] == 4):
                                break
                    down += 1
                area_h = board[left:right+1, y].copy()
                area_v = board[x, up:down+1].copy()
                len_h = right - left + 1
                len_v = down - up + 1
                pass_v = len_h>=2 and np.any(np.convolve(area_h == 3, np.ones(2, dtype=int), mode='valid') == 2)
                pass_h = len_v>=2 and np.any(np.convolve(area_v == 3, np.ones(2, dtype=int), mode='valid') == 2) and not pass_v
                if len_h >= size and not pass_h:
                    area_h[area_h == 3] = 0
                    conv_h = np.convolve(area_h, kernel, mode='valid')
                    mask_h = (conv_h == 0).astype(int)
                    res_h = np.convolve(mask_h, kernel, mode='full') * weights
                    res[left:right+1, y] += res_h
                if len_v >= size and not pass_v:
                    area_v[area_v == 3] = 0
                    conv_v = np.convolve(area_v, kernel, mode='valid')
                    mask_v = (conv_v == 0).astype(int)
                    res_v = np.convolve(mask_v, kernel, mode='full') * weights
                    res[x, up:down+1] += res_v
        res[poss[:, 0], poss[:, 1]] = -1
    else:
        print("Generating attack map based on empty cells...")
        for x, y in np.argwhere(board == 4):
            board[x, y] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1] and board[nx, ny] == 0:
                        board[nx, ny] = 1
        for size in ships:
            weights = ships[size][0]
            if weights == 0:
                continue
            kernel_h = np.ones((size, 1), dtype=int)
            kernel_v = np.ones((1, size), dtype=int)
            conv_h = conv2d(board, kernel_h)
            conv_v = conv2d(board, kernel_v)
            mask_h = np.pad((conv_h == 0).astype(int), ((size-1, size-1), (0, 0)), mode='constant', constant_values=0)
            mask_v = np.pad((conv_v == 0).astype(int), ((0, 0), (size-1, size-1)), mode='constant', constant_values=0)
            res_h = conv2d(mask_h, kernel_h) * weights
            res_v = conv2d(mask_v, kernel_v) * weights
            res += res_h + res_v
    return res

def generate_ships(conf):
    size = conf["size"]
    ships_req = conf["ships"]
    ships_layout = generate_ships_layout(size, ships_req)
    return ships_layout

def try_attack():
    print("My turn to attack.")
    board = get_board(baseurl, player)
    opponent_ships = get_ships_info(baseurl, player)
    attacks = generate_attack_map(board, opponent_ships)
    x, y = np.unravel_index(np.random.choice(np.where(attacks.ravel() == attacks.max())[0]), attacks.shape)
    x, y = int(x), int(y)
    print(f"Attacking position: ({x}, {y})")
    match hit_res := attack(baseurl, player, x, y):
        case -1:
            print("Error in attack. Retrying...")
        case 1:
            print("Attack missed.")
        case 3:
            print("Attack hit!")
        case 4:
            print("Ship sunk!")
        case 5:
            print("All opponent ships sunk! Victory!")
    return hit_res

async def algorithm():
    if (state:=get_state(baseurl)) == 0:
        conf = get_config(baseurl)
        print(f"Game config:\n{conf}")
        ships = generate_ships(conf)
        submit_ships(baseurl, player, ships)
        print("Ships submitted.")
        ready(baseurl, player)
        print(f"Player{player} ready.")
    elif state == 4 or state == 5:
        print("Game has ended.")
        return
    else:
        print("Game already started, taking over...")

    while True:
        await wait_call(interval)
        if (state:=get_state(baseurl)) == 4-player:
            print("Game over.")
            break
        if state != player+1:
            print("Waiting for opponent's turn...")
            continue
        hit_res = try_attack()
        sleep(interval/10)
        if hit_res == 5:
            break

async def main():
    done, pending = await asyncio.wait(
        [
            asyncio.create_task(ws_connect(baseurl, player)),
            asyncio.create_task(algorithm())
        ],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()

baseurl = "http://127.0.0.1:8000"
player = 0
interval = 5

if __name__ == "__main__":
    player = int(input("Enter player number (0 or 1): "))
    asyncio.run(main())