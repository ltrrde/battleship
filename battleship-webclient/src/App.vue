<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'

type ShipDraft = {
  id: number
  length: number
  count: number
}

type BoardPayload = {
  self: number[][]
  opponent: number[][]
}

type ShipStatusPayload = {
  player0: Record<string, [number, number]>
  player1: Record<string, [number, number]>
}

const defaultShips: Array<[number, number]> = [
  [5, 1],
  [4, 1],
  [3, 2],
  [2, 1],
]

let shipIdSeed = 0
const nextShipId = () => ++shipIdSeed

const serverBase = ref('')
const playerId = ref<0 | 1 | 2>(0)
const boardSize = ref(10)
const shipDrafts = ref<ShipDraft[]>(
  defaultShips.map(([length, count]) => ({ id: nextShipId(), length, count }))
)
const layoutInput = ref(
  `{
  "5": [[0, 0, 0]],
  "4": [[6, 0, 0]],
  "3": [[0, 2, 0], [7, 2, 0]],
  "2": [[4, 2, 0]]
}`
)
const boardState = ref<BoardPayload>({ self: [], opponent: [] })
const statusState = ref<number | null>(null)
const shipsState = ref<ShipStatusPayload | null>(null)
const logEntries = ref<string[]>([])
const isBusy = ref(false)
const attacking = ref(false)
const wsState = ref<'disconnected' | 'connecting' | 'connected'>('disconnected')
const wsHeartBeat = ref<string>('---')
const lastAttack = ref<[number, number] | null>(null)

let ws: WebSocket | null = null
let refreshTimer: number | null = null

const statusLabel = computed(() => {
  const map: Record<number, string> = {
    0: '等待布阵',
    1: '玩家 A 回合',
    2: '玩家 B 回合',
    3: '玩家 A 获胜',
    4: '玩家 B 获胜',
  }
  return statusState.value == null ? '未知' : map[statusState.value] ?? `未知(${statusState.value})`
})

const normalizedBase = computed(() => {
  const trimmed = serverBase.value.trim()
  if (!trimmed) return ''
  return trimmed.replace(/\/$/, '')
})

const boardLegend: Record<number, { label: string; tone: string }> = {
  0: { label: '雾区', tone: 'muted' },
  1: { label: '落空', tone: 'miss' },
  2: { label: '己方舰', tone: 'ally' },
  3: { label: '命中', tone: 'hit' },
  4: { label: '击沉', tone: 'sunk' },
  5: { label: '终结', tone: 'victory' },
}

const boardLength = computed(() => {
  const derived = boardState.value.self.length || boardState.value.opponent.length
  return derived || boardSize.value
})

const playerLabel = computed(() => (playerId.value === 2 ? "观战" : playerId.value === 0 ? '玩家 A' : '玩家 B'))

const opponentBoard = computed(() => {
  if (boardState.value.opponent.length > 0) {
    return boardState.value.opponent
  }
  return createEmptyBoard(boardLength.value)
})

const selfBoard = computed(() => {
  if (boardState.value.self.length > 0) {
    return boardState.value.self
  }
  return createEmptyBoard(boardLength.value)
})

function createEmptyBoard(size: number) {
  return Array.from({ length: size }, () => Array.from({ length: size }, () => 0))
}

function log(message: string) {
  const stamp = new Date().toLocaleTimeString()
  logEntries.value = [`[${stamp}] ${message}`, ...logEntries.value].slice(0, 120)
}

async function request<T>(path: string, init?: RequestInit) {
  const headers = new Headers(init?.headers ?? {})
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const base = normalizedBase.value
  const url = `${base ? base : 'http://localhost:8000'}${path}`
  const res = await fetch(url, {
    ...init,
    headers,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`HTTP ${res.status}: ${text || '请求失败'}`)
  }
  const contentType = res.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    return null as T
  }
  return (await res.json()) as T
}

function shipDraftsToPayload() {
  return shipDrafts.value.reduce<Record<string, number>>((acc, ship) => {
    if (ship.length > 0 && ship.count > 0) {
      acc[String(ship.length)] = ship.count
    }
    return acc
  }, {})
}

function addShipDraft() {
  shipDrafts.value = [...shipDrafts.value, { id: nextShipId(), length: 3, count: 1 }]
}

function removeShipDraft(id: number) {
  if (shipDrafts.value.length === 1) return
  shipDrafts.value = shipDrafts.value.filter((ship) => ship.id !== id)
}

async function syncConfig() {
  try {
    isBusy.value = true
    const data = await request<{ size: number; ships: Record<string, number> }>('/config/get')
    boardSize.value = data.size
    shipDrafts.value = Object.entries(data.ships).map(([length, count]) => ({
      id: nextShipId(),
      length: Number(length),
      count,
    }))
    log('配置已同步')
  } catch (error) {
    log((error as Error).message)
  } finally {
    isBusy.value = false
  }
}

async function pushConfig() {
  try {
    isBusy.value = true
    await request('/config/set', {
      method: 'POST',
      body: JSON.stringify({ size: boardSize.value, ships: shipDraftsToPayload() }),
    })
    log('待生效配置已更新')
  } catch (error) {
    log((error as Error).message)
  } finally {
    isBusy.value = false
  }
}

async function resetMatch() {
  try {
    isBusy.value = true
    await request('/config/reset')
    boardState.value = { self: [], opponent: [] }
    statusState.value = null
    log('对局已重置')
  } catch (error) {
    log((error as Error).message)
  } finally {
    isBusy.value = false
  }
}

async function submitFleet() {
  try {
    isBusy.value = true
    let ships: Record<string, number[][]>
    try {
      ships = JSON.parse(layoutInput.value)
    } catch (parseError) {
      throw new Error('舰队布局 JSON 解析失败')
    }
    await request('/game/submit', {
      method: 'POST',
      body: JSON.stringify({ player: playerId.value, ships }),
    })
    log('舰队布阵已提交')
  } catch (error) {
    log((error as Error).message)
  } finally {
    isBusy.value = false
  }
}

async function readyUp() {
  try {
    isBusy.value = true
    await request(`/game/ready/${playerId.value}`, { method: 'POST' })
    log('已发送 Ready')
    await fetchStatus()
  } catch (error) {
    log((error as Error).message)
  } finally {
    isBusy.value = false
  }
}

async function fetchStatus() {
  try {
    const data = await request<{ state: number }>('/game/status')
    statusState.value = data.state
  } catch (error) {
    log((error as Error).message)
  }
}

async function fetchBoard() {
  try {
    const board = await request<BoardPayload>(`/game/board/${playerId.value}`)
    board.self.forEach((row, x) => {
      row.forEach((value, y) => {
        if(value === 4) {
          for(let dx = -1; dx <= 1; dx++) {
            for(let dy = -1; dy <= 1; dy++) {
              const nx = x + dx
              const ny = y + dy
              if(nx >= 0 && nx < board.self.length && ny >= 0 && ny < board.self.length) {
                if(board.self[nx]?.[ny] === 0) {
                  board.self[nx][ny] = 1
                }
              }
            }
          }
        }
      })
    })
    board.opponent.forEach((row, x) => {
      row.forEach((value, y) => {
        if(value === 4) {
          for(let dx = -1; dx <= 1; dx++) {
            for(let dy = -1; dy <= 1; dy++) {
              const nx = x + dx
              const ny = y + dy
              if(nx >= 0 && nx < board.opponent.length && ny >= 0 && ny < board.opponent.length) {
                if(board.opponent[nx]?.[ny] === 0) {
                  board.opponent[nx][ny] = 1
                }
              }
            }
          }
        }
      })
    })
    boardState.value = board
  } catch (error) {
    log((error as Error).message)
  }
}

async function fetchShips() {
  try {
    const ships = await request<ShipStatusPayload>('/game/ships')
    shipsState.value = ships
  } catch (error) {
    log((error as Error).message)
  }
}

async function handleAttack(row: number, col: number) {
  if (attacking.value) return
  lastAttack.value = [row, col]
  try {
    attacking.value = true
    const payload = await request<{ result?: number; error?: string }>('/game/attack', {
      method: 'POST',
      body: JSON.stringify({ player: playerId.value, pos: [row, col] }),
    })
    if (payload.error) {
      log(`攻击失败: ${payload.error}`)
    } else if (payload.result != null) {
      const map: Record<number, string> = {
        1: '落空，轮次交给对手',
        3: '命中舰体',
        4: '击沉敌舰',
        5: '取得胜利',
      }
      log(`攻击 (${row}, ${col})：${map[payload.result] ?? payload.result}`)
      if (payload.result === 5) {
        statusState.value = playerId.value === 0 ? 3 : 4
      } else if (payload.result === 1) {
        statusState.value = playerId.value === 0 ? 2 : 1
      }
    }
    await refreshGameState()
  } catch (error) {
    log((error as Error).message)
  } finally {
    attacking.value = false
  }
}

function cellTone(value: number, side: 'self' | 'opponent', row: number, col: number) {
  const legend = boardLegend[value] ?? boardLegend[0] ?? { label: '未知', tone: 'muted' }
  const classes = [`tone-${legend.tone}`]
  if (
    side === 'opponent' &&
    lastAttack.value &&
    lastAttack.value[0] === row &&
    lastAttack.value[1] === col
  ) {
    classes.push('selected')
  }
  return classes.join(' ')
}

async function refreshGameState() {
  await Promise.all([fetchBoard(), fetchShips(), fetchStatus()])
}

function stopPolling() {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
}

function startPolling() {
  if (refreshTimer) return
  refreshTimer = window.setInterval(() => {
    void refreshGameState()
  }, 5000)
}

function ensurePollingMode() {
  if (wsState.value === 'connected') {
    stopPolling()
  } else {
    startPolling()
  }
}

function buildWsUrl() {
  const baseCandidate = normalizedBase.value || "http://localhost:8000"
  if (!baseCandidate) {
    throw new Error('无法解析 WebSocket 目标地址')
  }
  let url: URL
  try {
    url = new URL(baseCandidate)
  } catch {
    if (typeof window === 'undefined') {
      throw new Error('没有可用的全局地址上下文')
    }
    url = new URL(baseCandidate, window.location.origin)
  }
  if (url.protocol === 'https:') {
    url.protocol = 'wss:'
  } else if (url.protocol === 'http:') {
    url.protocol = 'ws:'
  } else if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
    url.protocol = 'ws:'
  }
  url.pathname = `/ws/${playerId.value}`
  url.search = ''
  url.hash = ''
  return url.toString()
}

function connectSocket() {
  disconnectSocket()
  try {
    const wsUrl = buildWsUrl()
    log(`WS: 准备连接 ${wsUrl}`)
    wsState.value = 'connecting'
    wsHeartBeat.value = '---'
    ws = new WebSocket(wsUrl)
    ws.addEventListener('open', () => {
      wsState.value = 'connected'
      log('WebSocket 已连接')
      void refreshGameState()
    })
    ws.addEventListener('message', (event) => {
      if (event.data === 'heartbeat') {
        wsHeartBeat.value = new Date().toLocaleTimeString()
      }
      log(`WS: ${event.data}`)
      void refreshGameState()
    })
    ws.addEventListener('close', () => {
      wsState.value = 'disconnected'
      log('WebSocket 已关闭')
    })
    ws.addEventListener('error', () => {
      wsState.value = 'disconnected'
      log('WebSocket 错误')
    })
  } catch (error) {
    wsState.value = 'disconnected'
    log(`WS 连接失败: ${(error as Error).message}`)
  }
}

function disconnectSocket() {
  if (ws) {
    ws.close()
    ws = null
  }
}

watch(
  () => wsState.value,
  () => {
    ensurePollingMode()
  }
)

watch(playerId, () => {
  if (wsState.value === 'connected') {
    connectSocket()
  }
})

onBeforeUnmount(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  disconnectSocket()
})

ensurePollingMode()
syncConfig()
fetchBoard()
fetchStatus()
fetchShips()
</script>

<template>
  <div class="page-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">海战棋指挥台</p>
        <h1>海战棋WEB端</h1>
        <p class="lede">
          根据服务端 API，直接在浏览器中布置舰队、监控棋盘并发动攻击。配置与攻防都在同一控制台完成。
        </p>
      </div>
      <div class="status-pill">
        <span class="label">当前状态</span>
        <strong>{{ statusLabel }}</strong>
      </div>
    </header>

    <section class="layout-grid">
      <article class="panel control">
        <div class="panel-header">
          <h2>连接参数</h2>
          <span class="tag">{{ playerLabel }}</span>
        </div>
        <label class="field">
          <span>服务器地址</span>
          <input
            v-model="serverBase"
            type="text"
            spellcheck="false"
            placeholder="留空以使用默认服务器"
          />
          <small class="hint">默认服务器：http://localhost:8000</small>
        </label>
        <label class="field inline">
          <span>玩家编号</span>
          <select v-model.number="playerId">
            <option :value="0">0 · 玩家 A</option>
            <option :value="1">1 · 玩家 B</option>
            <option :value="2">2 · 观战</option>
          </select>
        </label>
        <div class="actions">
          <button :disabled="isBusy" @click="syncConfig">同步配置</button>
          <button :disabled="isBusy" @click="resetMatch">重置</button>
        </div>
        <div class="ws-controls">
          <div>
            <p>WebSocket：{{ wsState === 'connected' ? '已连接' : wsState === 'connecting' ? '连接中' : '未连接' }}</p>
            <small>最近心跳：{{ wsHeartBeat }}</small>
          </div>
          <div class="actions compact">
            <button @click="connectSocket">连接</button>
            <button @click="disconnectSocket">断开</button>
          </div>
        </div>
      </article>

      <article class="panel config">
        <div class="panel-header">
          <h2>舰队配置</h2>
          <small>编辑后可保存为待生效配置，再通过 reset 生效</small>
        </div>
        <label class="field inline">
          <span>棋盘尺寸</span>
          <input v-model.number="boardSize" type="number" min="5" max="18" />
        </label>
        <div class="ship-table">
          <div class="ship-row ship-head">
            <span>舰长</span>
            <span>数量</span>
            <span></span>
          </div>
          <div v-for="ship in shipDrafts" :key="ship.id" class="ship-row">
            <input v-model.number="ship.length" type="number" min="2" max="8" />
            <input v-model.number="ship.count" type="number" min="1" max="4" />
            <button @click="removeShipDraft(ship.id)">移除</button>
          </div>
        </div>
        <div class="actions">
          <button @click="addShipDraft">新增舰种</button>
          <button :disabled="isBusy" @click="pushConfig">保存配置</button>
        </div>
      </article>

      <article class="panel layout" v-if="playerId != 2">
        <div class="panel-header">
          <h2>布阵提交</h2>
          <small>使用 API 识别的 JSON 描述舰船 [x, y, d] 信息</small>
        </div>
        <textarea v-model="layoutInput" rows="10"></textarea>
        <div class="actions">
          <button :disabled="isBusy" @click="submitFleet">提交舰队</button>
          <button :disabled="isBusy" @click="readyUp">准备就绪</button>
        </div>
      </article>

      <article class="panel logs">
        <div class="panel-header">
          <h2>事件纪录</h2>
          <small>包含 API 响应、WebSocket 提示与错误</small>
        </div>
        <div class="log-feed">
          <p v-for="entry in logEntries" :key="entry">{{ entry }}</p>
        </div>
      </article>

      <article class="panel board" :style="{ 'order' : playerId === 2 ? 2 : 1}">
        <div class="panel-header">
          <h2>棋盘可视化</h2>
          <small>{{ playerId == 2 ? "仅能查看双方明面信息" : "点击敌方棋盘对应格即可立即发起攻击" }}</small>
        </div>
        <div class="boards">
          <div class="board-card">
            <div class="board-title">{{ playerId == 2 ? "玩家 A 舰队" : "己方舰队" }}</div>
            <div class="grid-shell" :style="{ '--size': boardLength }">
              <div v-for="(row, rowIndex) in selfBoard" :key="`self-${rowIndex}`" class="board-row">
                <button
                  v-for="(value, colIndex) in row"
                  :key="`self-${rowIndex}-${colIndex}`"
                  class="board-cell"
                  :class="cellTone(value, 'self', rowIndex, colIndex)"
                  disabled
                  :aria-label="`${playerId == 2 ? '玩家 A ' : '己方'} (${rowIndex}, ${colIndex}) 状态 ${boardLegend[value]?.label ?? '未知'}`"
                ></button>
              </div>
            </div>
          </div>
          <div class="board-card">
            <div class="board-title">{{ playerId == 2 ? "玩家 B 舰队" : "敌方水域" }}</div>
            <div class="grid-shell" :style="{ '--size': boardLength }">
              <div v-for="(row, rowIndex) in opponentBoard" :key="`op-${rowIndex}`" class="board-row">
                <button
                  v-for="(value, colIndex) in row"
                  :key="`op-${rowIndex}-${colIndex}`"
                  class="board-cell"
                  :class="cellTone(value, 'opponent', rowIndex, colIndex)"
                  :disabled="attacking || !(statusState === playerId + 1) || playerId === 2"
                  @click="value === 0 && handleAttack(rowIndex, colIndex)"
                  :aria-label="`${playerId == 2 ? '玩家 B ' : '敌方'} (${rowIndex}, ${colIndex}) 状态 ${boardLegend[value]?.label ?? '未知'}`"
                ></button>
              </div>
            </div>
          </div>
        </div>
        <div class="legend">
          <span v-for="(meta, key) in boardLegend" :key="key" class="legend-item">
            <i :class="`badge tone-${meta.tone}`"></i>
            {{ key }} · {{ meta.label }}
          </span>
        </div>
        <div class="board-actions">
          <div class="actions">
            <button @click="fetchBoard">刷新棋盘</button>
            <button @click="fetchStatus">刷新状态</button>
            <button @click="fetchShips">舰船毁伤</button>
          </div>
        </div>
      </article>

      <article class="panel" :class="[{'status' : playerId !== 2}]" :style="{ 'order' : playerId === 2 ? 1 : 2}">
        <div class="panel-header">
          <h2>损耗概览</h2>
          <small>展示双方剩余舰船与击沉数量</small>
        </div>
        <div class="fleet-status" v-if="shipsState">
          <div class="fleet-column">
            <h3>{{ playerId == 2 ? "玩家 A" : "己方" }}</h3>
            <ul>
              <li v-for="(pair, key) in shipsState.player0" :key="`p0-${key}`">
                {{ key }} 长：存活 {{ pair[0] }} / 击沉 {{ pair[1] }}
              </li>
            </ul>
          </div>
          <div class="fleet-column">
            <h3>{{ playerId == 2 ? "玩家 B" : "敌方" }}</h3>
            <ul>
              <li v-for="(pair, key) in shipsState.player1" :key="`p1-${key}`">
                {{ key }} 长：存活 {{ pair[0] }} / 击沉 {{ pair[1] }}
              </li>
            </ul>
          </div>
        </div>
        <p v-else class="placeholder">暂无舰船统计，点击 “舰船毁伤”。</p>
      </article>
    </section>
  </div>
</template>

<style scoped>
:global(:root) {
  font-family: 'Space Grotesk', 'Poppins', 'Segoe UI', sans-serif;
  color: #121a2c;
  background-color: #f3f5fb;
}

.page-shell {
  min-height: 100vh;
  padding: 3rem clamp(1.5rem, 3vw, 4rem) 4rem;
  background: radial-gradient(circle at 5% 15%, rgba(255, 183, 77, 0.25), transparent 45%),
    radial-gradient(circle at 90% 0%, rgba(96, 165, 250, 0.4), transparent 40%),
    linear-gradient(135deg, #fcfdff 0%, #f2f4fb 45%, #fdfdfd 100%);
}

.hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 2rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid rgba(18, 26, 44, 0.08);
}

.eyebrow {
  font-size: 0.9rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: rgba(73, 88, 114, 0.8);
}

h1 {
  font-size: clamp(2.4rem, 3vw, 3.4rem);
  margin: 0.3rem 0 0.6rem;
  color: #061437;
}

.lede {
  max-width: 640px;
  color: rgba(18, 26, 44, 0.7);
  line-height: 1.6;
}

.status-pill {
  padding: 1.2rem 1.6rem;
  border-radius: 1rem;
  border: 1px solid rgba(18, 26, 44, 0.08);
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 10px 30px rgba(93, 113, 166, 0.15);
  text-align: right;
}

.status-pill .label {
  display: block;
  font-size: 0.8rem;
  color: rgba(73, 88, 114, 0.9);
}

.status-pill strong {
  font-size: 1.2rem;
  color: #0a1e4b;
}

.layout-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-top: 2.5rem;
}

.panel {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 1.25rem;
  padding: 1.5rem;
  border: 1px solid rgba(130, 146, 189, 0.2);
  box-shadow: 0 20px 45px rgba(119, 134, 168, 0.2);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}

.panel-header small {
  color: rgba(36, 45, 68, 0.65);
  font-size: 0.85rem;
}

.tag {
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
  border: 1px solid rgba(96, 165, 250, 0.5);
  font-size: 0.85rem;
  color: #2563eb;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  margin-bottom: 1rem;
  font-size: 0.9rem;
  color: #1f2942;
}

.hint {
  color: rgba(36, 45, 68, 0.6);
  font-size: 0.78rem;
}

.inline {
  flex-direction: row;
  align-items: center;
}

.inline span {
  width: 4rem;
}

input,
select,
textarea {
  width: 50%;
  padding: 0.7rem 0.9rem;
  border-radius: 0.9rem;
  border: 1px solid rgba(120, 134, 171, 0.35);
  background: #f8f9ff;
  color: #0f1f3c;
}

textarea {
  min-height: 160px;
  font-family: 'JetBrains Mono', 'Space Grotesk', sans-serif;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.ship-row button,
.actions button,
.attack-fields button,
.ws-controls button {
  border: none;
  border-radius: 0.9rem;
  padding: 0.65rem 1.4rem;
  background: linear-gradient(120deg, #38bdf8, #6366f1);
  color: #ffffff;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25);
}

button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  box-shadow: none;
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
}

.ghost {
  background: rgba(99, 102, 241, 0.12) !important;
  color: #4c1d95;
}

.control {
  grid-column: span 1;
}

.config {
  grid-column: span 1;
}

.board {
  grid-column: span 2;
}

.status {
  grid-column: span 2;
}

.logs {
  grid-column: span 1;
}

@media (max-width: 900px) {
  .board {
    grid-column: span 1;
  }
  .hero {
    flex-direction: column;
  }
}

.ship-table {
  border: 1px solid rgba(120, 134, 171, 0.25);
  border-radius: 1rem;
  padding: 0.6rem;
  margin-bottom: 1rem;
  background: rgba(248, 250, 255, 0.7);
}

.ship-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.5rem;
}

.ship-head {
  font-size: 0.8rem;
  text-transform: uppercase;
  color: rgba(47, 58, 87, 0.6);
}

.boards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}

.board-title {
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #0a1e4b;
}

.grid-shell {
  border: 1px solid rgba(120, 134, 171, 0.3);
  border-radius: 1rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.85);
}

.board-row {
  display: grid;
  grid-template-columns: repeat(var(--size), minmax(0, 1fr));
}

.board-cell {
  border: none;
  aspect-ratio: 1 / 1;
  margin: 1px;
  border-radius: 0.35rem;
  background: rgba(95, 110, 155, 0.12);
  cursor: pointer;
}

.board-card button:disabled {
  cursor: default;
}

.tone-muted {
  background: rgba(148, 163, 184, 0.25);
}

.tone-miss {
  background: rgba(14, 165, 233, 0.35);
}

.tone-ally {
  background: rgba(34, 197, 94, 0.4);
}

.tone-hit {
  background: rgba(249, 115, 22, 0.65);
}

.tone-sunk {
  background: rgba(239, 68, 68, 0.75);
}

.tone-victory {
  background: rgba(250, 204, 21, 0.75);
}

.selected {
  outline: 2px solid #f97316;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-top: 1rem;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
  color: rgba(36, 45, 68, 0.75);
}

.badge {
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 0.3rem;
  display: inline-block;
}


.board-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-top: 1rem;
  flex-wrap: wrap;
}

.fleet-status {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.fleet-column ul {
  list-style: none;
  padding: 0;
  margin: 0;
  line-height: 1.6;
  color: rgba(36, 45, 68, 0.75);
}

.placeholder {
  color: rgba(36, 45, 68, 0.55);
}

.log-feed {
  max-height: 300px;
  overflow: auto;
  font-family: 'JetBrains Mono', 'Space Grotesk', sans-serif;
  color: rgba(15, 23, 42, 0.9);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.ws-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1rem;
}

.ws-controls .compact button {
  padding: 0.5rem 0.9rem;
}

.logs p {
  margin: 0;
}
</style>
