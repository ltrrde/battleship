import logging
import numpy as np
from typing import Dict, List, Tuple
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import asyncio
import threading


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("battleship")

'''
board:
    0: empty
    1: miss
    2: ship
    3: hit
    4: sink
    5: win
'''
# states
PREPARE = 0
PLAYERA_TURN = 1
PLAYERB_TURN = 2
PLAYERA_WIN = 3
PLAYERB_WIN = 4

class Game:
    def __init__(self, *conf):
        self.size, self.ships = conf
        self.shipsA = {}
        self.shipsB = {}
        self.recordA = []
        self.recordB = []
        self.boardA = np.zeros((self.size, self.size), dtype=int)
        self.boardB = np.zeros((self.size, self.size), dtype=int)
        self.readyA = False
        self.readyB = False
        self.state = PREPARE
    def submit(self, ships:Dict, player:int):
        if self.state != PREPARE:
            logger.warning("Submit rejected for player %d: game not in PREPARE state", player)
            return False
        if not self._check(ships):
            logger.warning("Submit rejected for player %d: invalid ship placement", player)
            return False
        if player == 0:
            self.shipsA = ships
        elif player == 1:
            self.shipsB = ships
        else:
            logger.warning("Submit rejected: unknown player %d", player)
            return False
        logger.info("Player %d submitted ships successfully", player)
        return True
    def _check(self, ships:Dict):
        board = np.zeros((self.size, self.size))
        for size, ship in ships.items():
            if len(ship) != self.ships.get(size, 0):
                return False
            for x, y, d in ship:
                if d == 0:
                    if x + size > self.size:
                        return False
                    for i in range(size):
                        if board[x+i, y] == 1:
                            return False
                        if x+i+1 < self.size and board[x+i+1, y] == 1:
                            return False
                        if y-1 >= 0 and np.any(board[max(0, x+i-1):min(self.size, x+i+2), y-1] == 1):
                            return False
                        if y+1 < self.size and np.any(board[max(0, x+i-1):min(self.size, x+i+2), y+1] == 1):
                            return False
                        board[x + i][y] = 1
                elif d == 1:
                    if y + size > self.size:
                        return False
                    for i in range(size):
                        if board[x, y+i] == 1:
                            return False
                        if y+i+1 < self.size and board[x, y+i+1] == 1:
                            return False
                        if x-1 >= 0 and np.any(board[x-1, max(0, y+i-1):min(self.size, y+i+2)] == 1):
                            return False
                        if x+1 < self.size and np.any(board[x+1, max(0, y+i-1):min(self.size, y+i+2)] == 1):
                            return False
                        board[x][y + i] = 1
                else:
                    return False
        return True
    async def ready(self, player):
        if player == 0:
            self.readyA = True
        elif player == 1:
            self.readyB = True
        else:
            logger.warning("Ready rejected: unknown player %d", player)
            return False
        logger.info("Player %d is ready", player)
        if self.readyA and self.readyB:
            logger.info("Both players ready, starting match")
            await self._start()
        return True
    async def _start(self):
        if self.state != PREPARE:
            logger.warning("Start skipped: state=%d", self.state)
            return False
        if len(self.shipsA) == 0 or len(self.shipsB) == 0:
            logger.warning("Start skipped: ships missing (A:%s, B:%s)", "miss" if len(self.shipsA) == 0 else "fine", "miss" if len(self.shipsB) == 0 else "fine")
            return False
        shipsA = []
        shipsB = []
        for size, ships in self.shipsA.items():
            for x, y, d in ships:
                ship = {}
                if d == 0:
                    for i in range(size):
                        ship[x+i, y] = 1
                        self.boardA[x+i, y] = 2
                else:
                    for i in range(size):
                        ship[x, y+i] = 1
                        self.boardA[x, y+i] = 2
                shipsA.append(ship)
        for size, ships in self.shipsB.items():
            for x, y, d in ships:
                ship = {}
                if d == 0:
                    for i in range(size):
                        ship[x+i, y] = 1
                        self.boardB[x+i, y] = 2
                else:
                    for i in range(size):
                        ship[x, y+i] = 1
                        self.boardB[x, y+i] = 2
                shipsB.append(ship)
        self.shipsA[-1] = shipsA
        self.shipsB[-1] = shipsB
        self.state = PLAYERA_TURN
        logger.info("Game started, player 0 turn")
        await manager.broadcast_all("Game start!")
        await manager.broadcast("Your Turn.", player=0)
        return True
    async def attack(self, pos, player):
        opponent_ships = []
        opponent_board = None
        if player == 0:
            if self.state != PLAYERA_TURN:
                logger.warning("Attack rejected: player 0 attempted move during state %d", self.state)
                return -1
            if pos in self.recordA:
                logger.warning("Attack rejected: player 0 duplicate move at %s", pos)
                return -1
            self.recordA.append(pos)
            opponent_ships = self.shipsB
            opponent_board = self.boardB
        elif player == 1:
            if self.state != PLAYERB_TURN:
                logger.warning("Attack rejected: player 1 attempted move during state %d", self.state)
                return -1
            if pos in self.recordB:
                logger.warning("Attack rejected: player 1 duplicate move at %s", pos)
                return -1
            self.recordB.append(pos)
            opponent_ships = self.shipsA
            opponent_board = self.boardA
        else:
            logger.warning("Attack rejected: unknown player %d", player)
            return -1
        
        for ship in opponent_ships[-1]:
            if pos in ship:
                ship[pos] = 0
                sunk = all(status == 0 for status in ship.values())
                if sunk:
                    for p in ship.keys():
                        opponent_board[p] = 4
                    if all(all(status == 0 for status in s.values()) for s in opponent_ships[-1]):
                        self.state = PLAYERA_WIN if player == 0 else PLAYERB_WIN
                        logger.info("Player %d achieved victory with attack at %s", player, pos)
                        await manager.broadcast("You win!", player = player)
                        await manager.broadcast("You have been defeated!", player = 1 - player)
                        return 5
                    logger.info("Player %d sunk a ship at %s", player, pos)
                    await manager.broadcast("You sunk a ship!", player = player)
                    await manager.broadcast("One of your ships has been sunk!", player = 1 - player)
                    return 4
                else:
                    opponent_board[pos] = 3
                    logger.info("Player %d hit a ship at %s", player, pos)
                    await manager.broadcast("You hit a ship.", player = player)
                    await manager.broadcast("One of your ships has been hit.", player = 1 - player)
                    return 3
        else:
            opponent_board[pos] = 1
            self.state = PLAYERB_TURN if player == 0 else PLAYERA_TURN
            logger.info("Player %d missed at %s; switching to player %d", player, pos, 1 - player)
            await manager.broadcast("You missed.", player = player)
            await manager.broadcast("Your Turn.", player = 1 - player)
            return 1
    def get_board(self, player):
        self_board = self.boardB if player == 1 else self.boardA
        opponent_board = self.boardA if player == 1 else self.boardB
        if self.state != PLAYERA_WIN and self.state != PLAYERB_WIN:
            opponent_board = np.where(opponent_board == 2, 0, opponent_board)
            if player == 2:
                self_board = np.where(self_board == 2, 0, self_board)
        return self_board, opponent_board
    def get_state(self):
        return self.state
    def get_ships_info(self):
        if self.state == PREPARE:
            return {size: [count, 0] for size, count in self.ships.items()}, {size: [count, 0] for size, count in self.ships.items()}
        shipsA = {}
        for size in self.shipsA:
            if size == -1:
                continue
            shipsA[size] = [0, 0]
            for x, y, d in self.shipsA[size]:
                for ships in self.shipsA[-1]:
                    if (x,y) in ships:
                        if any(status == 1 for status in ships.values()):
                            shipsA[size][0] += 1
                        else:
                            shipsA[size][1] += 1
        shipsB = {}
        for size in self.shipsB:
            if size == -1:
                continue
            shipsB[size] = [0, 0]
            for x, y, d in self.shipsB[size]:
                for ships in self.shipsB[-1]:
                    if (x,y) in ships:
                        if any(status == 1 for status in ships.values()):
                            shipsB[size][0] += 1
                        else:
                            shipsB[size][1] += 1
        return shipsA, shipsB

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, list] = {0: [], 1: [], 2: []}
    async def connect(self, websocket, player: int):
        if player != 2 and len(self.active_connections[player]) >= 1:
            raise WebSocketException(code=1000, reason="Only one connection allowed per player")
        await websocket.accept()
        self.active_connections[player].append(websocket)
    def disconnect(self, websocket, player: int):
        self.active_connections[player].remove(websocket)
    async def send_personal_message(self, message: str, websocket):
        await websocket.send_text(message)
    async def broadcast(self, message: str, player: int):
        for connection in self.active_connections[player]:
            await connection.send_text(message)
        for connection in self.active_connections[2]:
            await connection.send_text(f"Update for player {player}: {message}")
    async def broadcast_all(self, message: str):
        for player_connections in self.active_connections.values():
            for connection in player_connections:
                await connection.send_text(message)
    async def heartbeat(self):
        while True:
            await self.broadcast_all("heartbeat")
            await asyncio.sleep(30)
manager = ConnectionManager()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
game_router = APIRouter()
@game_router.get("/status")
async def get_status():
    state = game.get_state()
    return {"state": state}
@game_router.get("/board/{player}")
async def get_board(player: int):
    board_self, board_opponent = game.get_board(player)
    if player == 2:
        return {
            "player0": board_self.tolist(),
            "player1": board_opponent.tolist()
        }
    return {    
        "self": board_self.tolist(),
        "opponent": board_opponent.tolist()
    }
@game_router.get("/ships")
async def get_ships():
    shipsA, shipsB = game.get_ships_info()
    return {
        "player0": shipsA,
        "player1": shipsB
    }
class SubmitData(BaseModel):
    player: int
    ships: Dict[int, List[Tuple[int, int, int]]]
@game_router.post("/submit")
async def submit(data: SubmitData):
    player = data.player
    ships = data.ships
    success = game.submit(ships, player)
    return {"success": success}
@game_router.post("/ready/{player}")
async def ready(player: int):
    success = await game.ready(player)
    return {"success": success}
class AttackData(BaseModel):
    player: int
    pos: Tuple[int, int]
@game_router.post("/attack")
async def attack(data: AttackData):
    player = data.player
    pos = data.pos
    result = await game.attack(pos, player)
    if result == -1:
        return {"error": "Invalid move"}
    else:
        return {"result": result}
app.include_router(game_router, prefix="/game")
conf_router = APIRouter()
@conf_router.get("/get")
async def get_config():
    return {
        "size": game.size,
        "ships": game.ships
    }
class ConfigData(BaseModel):
    size: int
    ships: Dict[int, int]
@conf_router.post("/set")
async def set_config(data: ConfigData):
    size = data.size
    ships = data.ships
    if not isinstance(size, int) or not isinstance(ships, dict):
        logger.warning("Config update rejected: invalid payload")
        return {"success": False, "error": "Invalid configuration"}
    global conf
    conf = (size, ships)
    logger.info("Configuration updated to size=%d ships=%s", size, ships)
    return {"success": True}
@conf_router.get("/reset")
async def reset_game():
    global game
    game = Game(*conf)
    logger.info("Game reset with configuration size=%d ships=%s", conf[0], conf[1])
    return {"success": True}
app.include_router(conf_router, prefix="/config")
ws_router = APIRouter()
@ws_router.websocket("/{player}")
async def websocket_endpoint(websocket: WebSocket, player: int):
    await manager.connect(websocket, player)
    try:
        while True:
            data = await websocket.receive_text()
            for player_id in manager.active_connections:
                if player_id != player:
                    await manager.broadcast(f"Player {player} says: {data}", player=player_id)
    except Exception as e:
        manager.disconnect(websocket, player)
app.include_router(ws_router, prefix="/ws")
@app.get("/")
async def root():
    return FileResponse("./battleship-webclient/dist/index.html")

conf = (10, {5:1, 4:1, 3:2, 2:1})
game = Game(*conf)

if __name__ == "__main__":
    threading.Thread(target=lambda: asyncio.run(manager.heartbeat()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
