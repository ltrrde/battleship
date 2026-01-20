import random
from typing import Dict, List
from requests import get, post
import numpy as np
import asyncio
import websockets

server_call = asyncio.Event()
async def wait_call(timeout: float = 30.0):
    try:
        await asyncio.wait_for(server_call.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    server_call.clear()
async def ws_connect(baseurl: str, player: int):
    uri = baseurl.replace("http", "ws") + f"/ws/{player}"
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                server_call.set()
    except Exception as e:
        print(f"WebSocket connection error: {e}")


def get_board(baseurl: str, player: int) -> np.ndarray:
    respone = get(f"{baseurl}/game/board/{player}")
    board = np.array(respone.json()["opponent"], dtype=int)
    return board
def get_state(baseurl: str) -> int:
    response = get(f"{baseurl}/game/status")
    state = response.json()["state"]
    return state
def attack(baseurl: str, player: int, x:int, y:int):
    response = post(f"{baseurl}/game/attack", json={"player": player, "pos": [x, y]})
    result = response.json()
    if "error" in result:
        return -1
    return result["result"]
def get_config(baseurl: str):
    response = get(f"{baseurl}/config/get")
    config = response.json()
    config["ships"] = {int(k): v for k, v in config["ships"].items()}
    return config
def submit_ships(baseurl: str, player: int, ships):
    response = post(f"{baseurl}/game/submit", json={"player": player, "ships": ships})
    result = response.json()
    return result["success"]
def ready(baseurl: str, player: int):
    response = post(f"{baseurl}/game/ready/{player}")
    result = response.json()
    return result["success"]
def get_ships_info(baseurl: str, player: int):
    response = get(f"{baseurl}/game/ships")
    result = response.json()
    result["player0"] = {int(k): v for k, v in result["player0"].items()}
    result["player1"] = {int(k): v for k, v in result["player1"].items()}
    return result[f"player{1-player}"]


def generate_ships_layout(size: int, ships: Dict[int, int]) -> Dict[int, List[List[int]]]:
    """
    生成海战棋布局
    
    参数:
    size: 棋盘大小
    ships: 舰船要求，格式为 {ship_size: ship_count}
    
    返回:
    舰船布局，格式为 {ship_size: [[x,y,d],...]}，xy为舰头位置，d为方向(0:右, 1:下)
    """
    
    def is_valid_position(board: List[List[int]], ship_size: int, x: int, y: int, direction: int) -> bool:
        """检查舰船位置是否合法"""
        
        # 检查是否超出边界
        if direction == 0:  # 向右
            if x + ship_size - 1 >= size:
                return False
        else:  # 向下
            if y + ship_size - 1 >= size:
                return False
        
        # 检查8格相邻（包括对角）
        for i in range(ship_size):
            if direction == 0:  # 向右
                curr_x, curr_y = x + i, y
            else:  # 向下
                curr_x, curr_y = x, y + i
            
            # 检查当前格子及周围8格
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = curr_x + dx, curr_y + dy
                    if 0 <= check_x < size and 0 <= check_y < size:
                        if board[check_y][check_x] == 1:
                            return False
        
        return True
    
    def place_ship(board: List[List[int]], ship_size: int, x: int, y: int, direction: int):
        """在棋盘上放置舰船"""
        for i in range(ship_size):
            if direction == 0:  # 向右
                board[y][x + i] = 1
            else:  # 向下
                board[y + i][x] = 1
    
    # 初始化棋盘，0表示空，1表示有舰船
    board = [[0 for _ in range(size)] for _ in range(size)]
    
    # 初始化结果字典
    result: Dict[int, List[List[int]]] = {}
    
    # 按舰船尺寸从大到小排序（提高放置成功率）
    sorted_ships = sorted(ships.items(), key=lambda x: x[0], reverse=True)
    
    # 尝试放置每种尺寸的舰船
    for ship_size, count in sorted_ships:
        result[ship_size] = []
        placed_count = 0
        attempts = 0
        max_attempts = 1000  # 防止无限循环
        
        while placed_count < count and attempts < max_attempts:
            # 随机选择起始位置和方向
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)
            direction = random.randint(0, 1)
            
            # 检查位置是否有效
            if is_valid_position(board, ship_size, x, y, direction):
                # 放置舰船
                place_ship(board, ship_size, x, y, direction)
                result[ship_size].append([x, y, direction])
                placed_count += 1
            
            attempts += 1
        
        # 如果无法放置所有该尺寸的舰船，抛出异常
        if placed_count < count:
            raise ValueError(f"无法在棋盘上放置所有尺寸为{ship_size}的舰船，只放置了{placed_count}/{count}艘")
    
    return result

if __name__ == "__main__":
    conf = {
        "board_size": 10,
        "ships": {
            4: 1,
            3: 2,
            2: 1,
            1: 1
        }
    }
    ships_layout = generate_ships_layout(conf["board_size"], conf["ships"])
    print(ships_layout)