"use strict";
// Живой дашборд DAMP: рисунок кистью → путь по уровням обученного цикла (сервер client/dashboard/server.py).
// Кусочек — окно внимания уровня (П.2); его код зажигает точки карты (§6.1), по ним срабатывают детекторы —
// кластеры карты (§6.5); сильнейшие дают деталь окна (П.1) — элемент следующего уровня. Всё считает сервер,
// страница только рисует ответ.

const $ = id => document.getElementById(id);
const DPR = Math.max(1, window.devicePixelRatio || 1);
const LAYER = ["#ff4d4d", "#ffd23f", "#3ddc84", "#2ec4ff", "#8f7bff", "#c0c6d0", "#7fffd4", "#ffffff"]; // слой 0 — самые широкие поля
const CLASS = Array.from({length: 10}, (_, c) => `hsl(${c * 36},85%,58%)`);
const PIN = "#ff8c1a", HOVER = "#ff5ad1", MEMBER = "#4dd2ff";
// Цвет штриха — его угол; у штриха середины линии угол без стрелки (0° и 180° — одна линия): цвет идёт по кругу за 180°.
const hue = a => { const turn = S.model && S.model.axial ? 2 * a : a; return `hsl(${(((turn % 360) + 360) % 360).toFixed(0)},95%,60%)`; };

function dec(text, Type) {
  const bin = atob(text), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// Строки массива rows (номер группы у каждого элемента) по группам: start[g]…start[g+1] в порядке order.
function group(rows, count) {
  const start = new Int32Array(count + 1);
  for (let i = 0; i < rows.length; i++) start[rows[i] + 1]++;
  for (let g = 0; g < count; g++) start[g + 1] += start[g];
  const fill = start.slice(0, count), order = new Int32Array(rows.length);
  for (let i = 0; i < rows.length; i++) order[fill[rows[i]]++] = i;
  return {start, order};
}

const S = {
  model: null, sight: null, image: null, example: null, hover: null, pin: null,
  previous: [], flash: [], views: [], cards: [], dirty: false, final: false, frame: 0,
  // Запросы рисунка: сколько в пути, номер последнего отправленного и последнего показанного, время круга (мс),
  // сколько запросов идёт сразу (готовых рабочих процессов сервера), моменты последних обновлений.
  inflight: 0, sequence: 0, applied: 0, lastSend: 0, round: 150, parallel: 1, workers: null, updates: [],
};

// ================================================================== данные
function prepareModel(raw) {
  const st = raw.sticks;
  const sticks = {
    picture: dec(st.picture, Int32Array), y: dec(st.y, Float32Array), x: dec(st.x, Float32Array), angle: dec(st.angle, Float32Array),
    span: dec(st.span, Float32Array), labels: dec(st.labels, Uint8Array), index: dec(st.index, Int32Array), pictures: st.pictures,
  };
  sticks.start = group(sticks.picture, st.pictures).start; // углопозиции идут по картинкам по возрастанию
  const levels = raw.levels.map(L => {
    const M = {
      number: L.number, height: L.height, width: L.width, points: L.points, count: L.count,
      cells: dec(L.cells, Int16Array), labels: dec(L.labels, Uint8Array), energies: dec(L.energies, Float32Array),
      pointPicture: dec(L.pointPicture, Int32Array), pointCentre: dec(L.pointCentre, Float32Array),
      pointExtent: dec(L.pointExtent, Float32Array), det: dec(L.detectors, Float32Array), layer: dec(L.layer, Uint8Array),
      thresholds: L.layerThresholds, activation: L.activation, half: L.fieldHalf, members: new Map(),
    };
    M.index = new Int32Array(M.width * M.height).fill(-1);
    for (let i = 0; i < M.points; i++) M.index[M.cells[2 * i] * M.width + M.cells[2 * i + 1]] = i;
    return M;
  });
  // Уровень 0 — первичная карта: точка хранит код углопозиции обучающей цифры (угол и место, §4.6).
  let primary = null;
  if (raw.primary) {
    const P = raw.primary;
    primary = {
      height: P.height, width: P.width, points: P.points, count: P.count, period: P.period, activation: P.activation,
      cells: dec(P.cells, Int16Array), angle: dec(P.angle, Float32Array), y: dec(P.y, Float32Array), x: dec(P.x, Float32Array),
      labels: dec(P.labels, Uint8Array), det: dec(P.detectors, Float32Array), layer: dec(P.layer, Uint8Array),
      // Уровень 0 в цикле (LADDER_LEVEL0): элементы уровня 1 — детали уровня 0, а не сами углопозиции.
      inCycle: Boolean(P.inCycle),
    };
    primary.index = new Int32Array(primary.width * primary.height).fill(-1);
    for (let i = 0; i < primary.points; i++) primary.index[primary.cells[2 * i] * primary.width + primary.cells[2 * i + 1]] = i;
    // Цвет точки — угол её штриха, полный круг цвета на период угла (тот же, что у штрихов на рисунках).
    primary.colour = Array.from(primary.angle, a => `hsl(${((((a / primary.period) * 360) % 360 + 360) % 360).toFixed(0)},95%,60%)`);
  }
  return {train: raw.train, originals: raw.originals, decay: raw.decay, brush: raw.brush, cell: raw.grid.cell, sticks, levels, primary,
    axial: Boolean(raw.axial),
    topClusters: raw.topClusters, litPerPiece: raw.litPerPiece, ownClusters: raw.ownClusters, litFloor: raw.litFloor};
}

function prepareSight(raw) {
  const sticks = {
    angle: dec(raw.sticks.angle, Float32Array), y: dec(raw.sticks.y, Float32Array), x: dec(raw.sticks.x, Float32Array),
    strength: dec(raw.sticks.strength, Float32Array), span: dec(raw.sticks.span, Float32Array), count: raw.sticks.count,
  };
  const levels = raw.levels.map((L, n) => {
    const M = S.model.levels[n], K = L.keyholes;
    // Зажжённые точки: у точки — её сильнейшая активация и сколько кусочков её зажгли; у кусочка — его сильнейшие
    // точки (DASHBOARD_LIT_PER_PIECE). Срабатывания детекторов — все.
    const V = {
      number: L.number, K, centre: dec(L.centre, Float32Array), extent: dec(L.extent, Float32Array),
      entryKeyhole: dec(L.entryKeyhole, Uint16Array), entryElement: dec(L.entryElement, Uint16Array),
      pointAct: dec(L.pointActivation, Float32Array), pointCount: dec(L.pointCount, Uint16Array),
      litKeyhole: dec(L.litKeyhole, Uint16Array), litPoint: dec(L.litPoint, Uint32Array), litValue: dec(L.litValue, Float32Array),
      litCount: dec(L.litCount, Uint32Array),
      firedKeyhole: dec(L.firedKeyhole, Uint16Array), firedDetector: dec(L.firedDetector, Uint16Array),
      firedLevel: dec(L.firedLevel, Float32Array), detCount: dec(L.detectorCount, Uint16Array), detLevel: dec(L.detectorLevel, Float32Array),
      litPairs: L.litPairs, firedTotal: L.fired, strength: dec(L.strength, Float32Array), detailLength: L.detailLength,
    };
    V.lit = group(V.litKeyhole, K);
    V.fired = group(V.firedKeyhole, K);
    V.entries = group(V.entryKeyhole, K);
    V.byDetector = group(V.firedDetector, M.count);
    V.actHigh = 0;
    for (let p = 0; p < M.points; p++) V.actHigh = Math.max(V.actHigh, V.pointAct[p]);
    const ranked = [];
    for (let d = 0; d < M.count; d++) if (V.detCount[d]) ranked.push(d);
    ranked.sort((a, b) => V.detCount[b] - V.detCount[a] || V.detLevel[b] - V.detLevel[a]);
    V.ranked = ranked;
    V.top = new Set(ranked.slice(0, S.model.topClusters));
    V.best = new Int32Array(K).fill(-1);
    V.bestLevel = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      for (let j = V.fired.start[k]; j < V.fired.start[k + 1]; j++) {
        const i = V.fired.order[j];
        if (V.firedLevel[i] > V.bestLevel[k]) { V.bestLevel[k] = V.firedLevel[i]; V.best[k] = V.firedDetector[i]; }
      }
    }
    V.clusters = 0;
    for (let d = 0; d < M.count; d++) if (V.detCount[d]) V.clusters++;
    V.litPoints = 0;
    for (let p = 0; p < M.points; p++) if (V.pointAct[p] > 0) V.litPoints++;
    return V;
  });
  // Кусочек уровня n + 1 состоит из деталей уровня n: обратная связь «кусочек n → кусочки n + 1, куда он вошёл».
  for (let n = 1; n < levels.length; n++) levels[n].parents = group(levels[n].entryElement, levels[n - 1].K);
  // Уровень 0: отклик первичной карты — сильнейшая активация каждой точки и сработавшие детекторы.
  let primary = null;
  if (raw.primary && S.model.primary) {
    const Q = raw.primary, M = S.model.primary;
    primary = {
      sticks: Q.sticks, litPairs: Q.litPairs, firedTotal: Q.fired,
      pointAct: dec(Q.pointActivation, Float32Array), pointCount: dec(Q.pointCount, Uint16Array),
      detCount: dec(Q.detectorCount, Uint16Array), detLevel: dec(Q.detectorLevel, Float32Array),
      // Деталь уровня 0 у каждого штриха (П.1): сильнейшая зажжённая им точка карты (−1 — ни одной) и сила детали.
      stickPoint: Q.stickPoint ? dec(Q.stickPoint, Int32Array) : null,
      stickStrength: Q.stickStrength ? dec(Q.stickStrength, Float32Array) : null,
    };
    primary.actHigh = 0; primary.litPoints = 0;
    for (let p = 0; p < M.points; p++) { if (primary.pointAct[p] > 0) primary.litPoints++; primary.actHigh = Math.max(primary.actHigh, primary.pointAct[p]); }
    const ranked = [];
    for (let d = 0; d < M.count; d++) if (primary.detCount[d]) ranked.push(d);
    ranked.sort((a, b) => primary.detCount[b] - primary.detCount[a] || primary.detLevel[b] - primary.detLevel[a]);
    primary.ranked = ranked;
    primary.top = new Set(ranked.slice(0, S.model.topClusters));
  }
  return {sticks, levels, primary, digit: raw.digit, probabilities: raw.probabilities, ms: raw.ms, memory: raw.memory || null, votes: raw.votes || []};
}

// Точки карты в поле детектора d (§6.5: точки не дальше r_d от центра) — для состава кластера.
function members(M, d) {
  if (M.members.has(d)) return M.members.get(d);
  const cy = M.det[5 * d], cx = M.det[5 * d + 1], r = M.det[5 * d + 2], found = [];
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(M.height - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(M.width - 1, Math.ceil(cx + r)); x++) {
      const i = M.index[y * M.width + x];
      if (i >= 0 && Math.hypot(y - cy, x - cx) <= r) found.push(i);
    }
  }
  M.members.set(d, found);
  return found;
}

function composition(M, d) {
  const counts = new Array(10).fill(0), inside = members(M, d);
  for (const i of inside) counts[M.labels[i]]++;
  return counts.map((c, digit) => ({digit, share: inside.length ? c / inside.length : 0})).sort((a, b) => b.share - a.share);
}

// Сильнейшая зажжённая точка кусочка k — окно памяти, на которое кусочек больше всего похож (−1 — ничего не зажёг).
function strongestPoint(V, k) {
  let best = -1, value = 0;
  for (let j = V.lit.start[k]; j < V.lit.start[k + 1]; j++) {
    const i = V.lit.order[j];
    if (V.litValue[i] > value) { value = V.litValue[i]; best = V.litPoint[i]; }
  }
  return best;
}

// Выбранное — кусочек на рисунке, кластер или точка карты — как кусочки своего уровня, с которых идёт путь вниз
// (из чего собраны) и вверх (куда вошли). У точки — кусочки, в чьих DASHBOARD_LIT_PER_PIECE сильнейших она есть.
// Кусочки идут от сильнейшего: у кластера — по уровню его активации E, у точки — по её активации a. Точные кусочки
// точки приходят с сервера (/api/point → S.pin.keys); пока их нет — по сильнейшим точкам кусочков. Ответ помнится,
// пока не сменились рисунок или выбор: его спрашивают несколько раз за кадр.
function pinKey() {
  const P = S.pin;
  return P ? `${P.kind}:${P.level}:${P.kind === "piece" ? P.keyhole : P.index}:${P.keys ? P.keys.length : -1}` : "";
}

function pinnedPieces() {
  const sight = S.sight, P = S.pin;
  if (!sight || !P) return null;
  const V = sight.levels[P.level];
  if (!V) return null;
  const key = pinKey(), known = S.pinnedCache;
  if (known && known.sight === sight && known.key === key) return known.value;
  let keys;
  if (P.kind === "piece") keys = [P.keyhole];
  else if (P.kind === "cluster") {
    const B = V.byDetector, found = [];
    for (let j = B.start[P.index]; j < B.start[P.index + 1]; j++) found.push(B.order[j]);
    found.sort((a, b) => V.firedLevel[b] - V.firedLevel[a]);
    keys = [...new Set(found.map(i => V.firedKeyhole[i]))];
  } else if (P.keys) keys = P.keys.filter(k => k < V.K);
  else {
    const found = [];
    for (let i = 0; i < V.litPoint.length; i++) if (V.litPoint[i] === P.index) found.push(i);
    found.sort((a, b) => V.litValue[b] - V.litValue[a]);
    keys = [...new Set(found.map(i => V.litKeyhole[i]))];
  }
  const value = {level: P.level, keys};
  S.pinnedCache = {sight, key, value};
  return value;
}

// Все кусочки рисунка, зажёгшие выбранную точку карты, — с сервера (у страницы — лишь сильнейшие точки кусочков).
async function loadPointPieces() {
  const P = S.pin;
  if (!P || P.kind !== "point" || !S.image) return;
  try {
    // Рисунок идёт вместе с запросом: сервер помнит путь последнего рисунка, а вкладок может быть несколько.
    const body = JSON.stringify({image: btoa(String.fromCharCode.apply(null, S.image)), level: P.level, index: P.index});
    const answer = await (await fetch("/api/point", {method: "POST", body})).json();
    if (answer.ok && S.pin === P) P.keys = Array.from(dec(answer.keyholes, Uint16Array));
  } catch (error) { /* останется список по сильнейшим точкам кусочков */ }
}

// ================================================================== рисование содержимого
// Деталь кусочка на рисунке (П.1): штрихи окна памяти i карты M, на которое кусочек больше всего похож, — на месте
// кусочка (cx, cy) в настоящем масштабе (cellPx — пикселей на клетку сетки), одним цветом.
function fragment(ctx, M, i, cx, cy, cellPx, alpha, ink, width) {
  const sticks = S.model.sticks, p = M.pointPicture[i];
  if (p < 0) return;
  const py = M.pointCentre[2 * i], px = M.pointCentre[2 * i + 1], extent = M.pointExtent[i] + 1e-5, half = cellPx * 0.45;
  ctx.strokeStyle = ink; ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.lineCap = "round";
  ctx.beginPath();
  for (let j = sticks.start[p]; j < sticks.start[p + 1]; j++) {
    const dy = sticks.y[j] - py, dx = sticks.x[j] - px;
    if (Math.hypot(dy, dx) > extent) continue;
    const along = (sticks.angle[j] + 90) * Math.PI / 180, ux = Math.cos(along) * half, uy = -Math.sin(along) * half;
    const x = cx + dx * cellPx, y = cy + dy * cellPx;
    ctx.moveTo(x - ux, y - uy); ctx.lineTo(x + ux, y + uy);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function stick(ctx, px, py, unit, angle, alpha, ink, span) {
  // Угол углопозиции — направление перепада яркости, поперёк линии цифры: отрезок рисуется вдоль края.
  const along = (angle + 90) * Math.PI / 180, half = unit * 0.45;
  const ux = Math.cos(along) * half, uy = -Math.sin(along) * half;
  ctx.strokeStyle = ink || hue(angle);
  // Поле штриха (§6.4.3): детектор покрывает край на span клеток в каждую сторону от центра — тонкая линия под
  // отрезком. Центры детекторов стоят на расстоянии от одного до двух таких полуполей (§6.4.5), край покрыт целиком.
  if (span > 0.45) {
    const fx = ux / 0.45 * span, fy = uy / 0.45 * span;
    ctx.globalAlpha = alpha * 0.5; ctx.lineWidth = Math.max(1, unit / 10);
    ctx.beginPath(); ctx.moveTo(px - fx, py - fy); ctx.lineTo(px + fx, py + fy); ctx.stroke();
  }
  ctx.globalAlpha = alpha;
  ctx.lineWidth = Math.max(1, unit / 5);
  ctx.beginPath(); ctx.moveTo(px - ux, py - uy); ctx.lineTo(px + ux, py + uy); ctx.stroke();
  ctx.globalAlpha = 1;
}

// Содержимое точки карты: углопозиции её обучающей картинки в поле окна — ярко то, что видит окно (§5, П.2).
function glyph(ctx, M, i, left, top, size) {
  const sticks = S.model.sticks, p = M.pointPicture[i];
  if (p < 0) return;
  const cy = M.pointCentre[2 * i], cx = M.pointCentre[2 * i + 1], extent = M.pointExtent[i], half = M.half + 0.5;
  const unit = size / (2 * half);
  for (let j = sticks.start[p]; j < sticks.start[p + 1]; j++) {
    const dy = sticks.y[j] - cy, dx = sticks.x[j] - cx;
    if (Math.abs(dy) > half || Math.abs(dx) > half) continue;
    const inside = Math.hypot(dy, dx) <= extent + 1e-5;  // места и поле приходят в float32
    stick(ctx, left + size / 2 + dx * unit, top + size / 2 + dy * unit, unit, sticks.angle[j], inside ? 1 : 0.22, null, sticks.span[j]);
  }
}

function glyphCanvas(M, i, size, frame) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size * DPR;
  canvas.style.width = canvas.style.height = size + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(DPR, DPR);
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, size, size);
  ctx.lineCap = "round";
  glyph(ctx, M, i, 0, 0, size);
  if (frame) { ctx.strokeStyle = CLASS[M.labels[i]]; ctx.lineWidth = 2; ctx.strokeRect(1, 1, size - 2, size - 2); }
  return canvas;
}

// ================================================================== холст и кисть
const pad = $("pad"), padCtx = pad.getContext("2d");
// strokes — штрихи в пикселях поля 28×28; move — точка, за которую тянут рисунок; tool — кисть или «двигать»;
// shift — на сколько рисунок сдвинут с места, где его нарисовали.
const DRAW = {side: 28, sample: 8, frame: 20, strokes: [], stroke: null, move: null, tool: "draw", shift: [0, 0]};
const ink = document.createElement("canvas");
ink.width = ink.height = DRAW.side * DRAW.sample;
const inkCtx = ink.getContext("2d", {willReadFrequently: true});

function strokePath(ctx, width) {
  ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = width; ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff";
  for (const s of DRAW.strokes) {
    ctx.beginPath();
    if (s.length === 2) { ctx.arc(s[0], s[1], width / 2, 0, 2 * Math.PI); ctx.fill(); continue; }
    ctx.moveTo(s[0], s[1]);
    for (let k = 2; k < s.length; k += 2) ctx.lineTo(s[k], s[k + 1]);
    ctx.stroke();
  }
}

// Покрытие клеток 28×28 штрихами (сдвиг dx, dy пикселей): доля каждой клетки, суперсэмплинг ×8.
function coverage(dx, dy) {
  const n = DRAW.side, q = DRAW.sample, size = n * q;
  inkCtx.setTransform(1, 0, 0, 1, 0, 0); inkCtx.clearRect(0, 0, size, size);
  inkCtx.setTransform(q, 0, 0, q, dx * q, dy * q);
  strokePath(inkCtx, S.model.brush.width);
  const alpha = inkCtx.getImageData(0, 0, size, size).data, cover = new Float32Array(n * n);
  for (let y = 0; y < size; y++) {
    const row = Math.floor(y / q) * n;
    for (let x = 0; x < size; x++) cover[row + Math.floor(x / q)] += alpha[(y * size + x) * 4 + 3];
  }
  const part = 1 / (255 * q * q);
  for (let i = 0; i < cover.length; i++) cover[i] *= part;
  return cover;
}

// Гауссово размытие σ (как у штриха MNIST) и перевод в байты 0…255.
function bytesOf(cover) {
  const n = DRAW.side, sigma = S.model.brush.sigma, reach = Math.floor(3 * sigma + 0.5), kernel = [];
  let total = 0;
  for (let k = -reach; k <= reach; k++) { const w = Math.exp(-k * k / (2 * sigma * sigma)); kernel.push(w); total += w; }
  const pass = (source, sx, sy) => {
    const out = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      let sum = 0;
      for (let k = -reach; k <= reach; k++) {
        const xx = x + k * sx, yy = y + k * sy;
        if (xx >= 0 && yy >= 0 && xx < n && yy < n) sum += source[yy * n + xx] * kernel[k + reach];
      }
      out[y * n + x] = sum / total;
    }
    return out;
  };
  const soft = pass(pass(cover, 1, 0), 0, 1), bytes = new Uint8Array(n * n);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.round(Math.min(1, Math.max(0, soft[i])) * 255);
  return bytes;
}

// Рисунок для сети: как нарисован либо поставленный по центру масс в точку цифр MNIST (замер — DASHBOARD_CENTRE).
function padPicture() {
  if (S.example) return S.example.image;
  const n = DRAW.side, cover = coverage(0, 0);
  if (!$("centre").checked) return bytesOf(cover);
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const v = cover[y * n + x]; mass += v; mx += v * (x + 0.5); my += v * (y + 0.5); }
  if (mass <= 0) return bytesOf(cover);
  const c = S.model.brush.centre;
  return bytesOf(coverage(c - mx / mass, c - my / mass));
}

function drawPad() {
  const size = pad.width, k = size / DRAW.side, edge = (DRAW.side - DRAW.frame) / 2;
  padCtx.setTransform(1, 0, 0, 1, 0, 0); padCtx.fillStyle = "#000"; padCtx.fillRect(0, 0, size, size);
  if (S.example) {
    const tile = document.createElement("canvas"); tile.width = tile.height = DRAW.side;
    const t = tile.getContext("2d"), data = t.createImageData(DRAW.side, DRAW.side);
    for (let i = 0; i < S.example.image.length; i++) { const v = S.example.image[i]; data.data.set([v, v, v, 255], 4 * i); }
    t.putImageData(data, 0, 0);
    padCtx.imageSmoothingEnabled = true; padCtx.drawImage(tile, 0, 0, size, size);
  }
  padCtx.setTransform(k, 0, 0, k, 0, 0);
  padCtx.strokeStyle = "rgba(255,255,255,0.22)"; padCtx.lineWidth = 1.5 / k; padCtx.setLineDash([6 / k, 6 / k]);
  padCtx.strokeRect(edge, edge, DRAW.frame, DRAW.frame); padCtx.setLineDash([]);
  if (!DRAW.strokes.length && !S.example) {
    padCtx.setTransform(1, 0, 0, 1, 0, 0); padCtx.fillStyle = "rgba(255,255,255,0.4)"; padCtx.textAlign = "center";
    padCtx.font = `${Math.round(size / 19)}px Segoe UI, Arial`;
    padCtx.fillText("рисуйте цифру во всю рамку", size / 2, size / 2);
    return;
  }
  if (DRAW.strokes.length) {
    padCtx.filter = `blur(${(S.model.brush.sigma * k).toFixed(2)}px)`;
    strokePath(padCtx, S.model.brush.width);
    padCtx.filter = "none";
  }
}

function padPoint(event) {
  const box = pad.getBoundingClientRect();
  return [(event.clientX - box.left) / box.width * DRAW.side, (event.clientY - box.top) / box.height * DRAW.side];
}
pad.addEventListener("contextmenu", event => event.preventDefault()); // правая кнопка — двигать рисунок
pad.addEventListener("pointerdown", event => {
  if (!S.model) return;
  const moving = event.button === 2 || event.shiftKey || DRAW.tool === "move";
  if (event.button !== 0 && !moving) return;
  event.preventDefault();
  pad.setPointerCapture?.(event.pointerId);
  if (moving) { DRAW.move = padPoint(event); pad.classList.add("grabbing"); return; }
  if (S.example) { S.example = null; $("padNote").textContent = ""; DRAW.shift = [0, 0]; showShift(); }
  DRAW.stroke = padPoint(event); DRAW.strokes.push(DRAW.stroke);
  changed(false);
});
pad.addEventListener("pointermove", event => {
  if (DRAW.move) {
    const point = padPoint(event);
    moveBy(point[0] - DRAW.move[0], point[1] - DRAW.move[1], false);
    DRAW.move = point;
    return;
  }
  const stroke = DRAW.stroke;
  if (!stroke) return;
  const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [];
  for (const one of events.length ? events : [event]) {
    const point = padPoint(one), n = stroke.length;
    if (Math.hypot(point[0] - stroke[n - 2], point[1] - stroke[n - 1]) >= 0.03) stroke.push(point[0], point[1]);
  }
  changed(false);
});
for (const type of ["pointerup", "pointercancel"]) pad.addEventListener(type, () => {
  if (DRAW.move) { DRAW.move = null; pad.classList.remove("grabbing"); changed(true); return; }
  if (DRAW.stroke) { DRAW.stroke = null; changed(true); }
});
$("clear").addEventListener("click", () => {
  DRAW.strokes = []; DRAW.stroke = null; DRAW.shift = [0, 0]; S.example = null; S.pin = null; $("padNote").textContent = ""; showShift(); changed(true);
});
$("undo").addEventListener("click", () => { DRAW.strokes.pop(); DRAW.stroke = null; changed(true); });
$("centre").addEventListener("change", () => { showShift(); changed(true); });
$("example").addEventListener("click", loadExample);
$("moveTool").addEventListener("click", () => {
  DRAW.tool = DRAW.tool === "move" ? "draw" : "move";
  $("moveTool").classList.toggle("on", DRAW.tool === "move");
  pad.classList.toggle("moving", DRAW.tool === "move");
});
$("home").addEventListener("click", home);
window.addEventListener("keydown", event => {
  // Стрелки — сдвиг рисунка на 1 пиксель поля 28×28 (если фокус не в поле ввода).
  if (!S.model || event.target.closest?.("input, textarea, select")) return;
  const step = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
  if (!step) return;
  event.preventDefault();
  moveBy(step[0], step[1], true);
});

// Сдвиг рисунка по холсту на (dx, dy) пикселей поля 28×28. Штрихи двигаются как есть (дробно); пример MNIST —
// целыми пикселями, с нулями на освободившемся месте, как сдвиг картинки MNIST. Центр масс при сдвиге выключается:
// иначе сеть видела бы рисунок на прежнем месте и сдвиг ничего бы не менял.
function moveBy(dx, dy, final) {
  if (!dx && !dy) return;
  if (S.example) {
    const e = S.example;
    e.offset[0] += dx; e.offset[1] += dy;
    const ix = Math.round(e.offset[0]), iy = Math.round(e.offset[1]);
    if (ix === e.shift[0] && iy === e.shift[1]) return;
    e.shift = [ix, iy];
    e.image = shifted(e.original, ix, iy);
  } else if (DRAW.strokes.length) {
    for (const s of DRAW.strokes) for (let k = 0; k < s.length; k += 2) { s[k] += dx; s[k + 1] += dy; }
    DRAW.shift[0] += dx; DRAW.shift[1] += dy;
    if ($("centre").checked) $("centre").checked = false;
  } else return;
  showShift();
  changed(final);
}

// Картинка 28×28, сдвинутая на целые (dx, dy) пикселей; ушедшее за край пропадает, пустое — нули.
function shifted(source, dx, dy) {
  const n = DRAW.side, out = new Uint8Array(n * n);
  for (let y = 0; y < n; y++) {
    const sy = y - dy;
    if (sy < 0 || sy >= n) continue;
    for (let x = 0; x < n; x++) { const sx = x - dx; if (sx >= 0 && sx < n) out[y * n + x] = source[sy * n + sx]; }
  }
  return out;
}

// Вернуть рисунок туда, где его нарисовали (пример — на место в MNIST).
function home() {
  if (S.example) {
    S.example.offset = [0, 0]; S.example.shift = [0, 0]; S.example.image = S.example.original;
  } else {
    const [sx, sy] = DRAW.shift;
    for (const s of DRAW.strokes) for (let k = 0; k < s.length; k += 2) { s[k] -= sx; s[k + 1] -= sy; }
    DRAW.shift = [0, 0];
  }
  showShift();
  changed(true);
}

function showShift() {
  const sign = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(S.example ? 0 : 1);
  const [sx, sy] = S.example ? S.example.shift : DRAW.shift;
  let text = "";
  if (sx || sy) {
    text = `Рисунок сдвинут на (${sign(sx)}, ${sign(sy)}) px по x и y` + (S.example ? " — целыми пикселями, как сдвиг MNIST." : ".");
    if (!S.example) text += $("centre").checked ? " Центр масс включён: сеть видит рисунок в центре, сдвиг не виден ей." :
      " Центр масс выключен: сеть видит рисунок там, где он стоит.";
  }
  $("moveNote").textContent = text;
}

async function loadExample() {
  const answer = await (await fetch("/api/example")).json();
  if (!answer.ok) { $("padNote").textContent = answer.error; return; }
  DRAW.strokes = []; DRAW.stroke = null; DRAW.shift = [0, 0]; S.pin = null;
  const image = dec(answer.image, Uint8Array);
  S.example = {image, original: image, mnist: answer.mnist, offset: [0, 0], shift: [0, 0]};
  showShift();
  $("padNote").innerHTML = `Пример: цифра «${answer.digit}», MNIST t10k № <b>${answer.mnist}</b> (каждый раз другая).`;
  changed(true);
}

function changed(final) { drawPad(); schedule(final); }

// Рисунки считают рабочие процессы сервера, поэтому сразу идёт столько запросов, сколько процессов готово (S.parallel;
// без них — один). Запросы уходят равномерно — через долю времени круга, — иначе ответы приходили бы пачками. У
// запроса свой номер: ответ старше уже показанного отбрасывается. Последний рисунок штриха (final) уходит сразу.
function schedule(final) {
  S.final = S.final || final;
  S.dirty = true;
  pump();
}

function pump() {
  if (!S.dirty) return;
  clearTimeout(S.pumpTimer);
  const parallel = Math.max(1, S.parallel || 1);
  if (S.inflight >= parallel) return;  // свободного места нет: рисунок уйдёт, когда вернётся ответ
  const wait = S.inflight ? S.lastSend + S.round / parallel - performance.now() : 0;
  if (wait > 1 && !S.final) { S.pumpTimer = setTimeout(pump, wait); return; }
  send();
}

async function send() {
  S.dirty = false;
  if (!DRAW.strokes.length && !S.example) {
    // Холст пуст: ответы на прежние рисунки, которые ещё в пути, больше не нужны.
    S.final = false; S.sight = null; S.image = null; S.applied = ++S.sequence; render(); return;
  }
  const number = ++S.sequence, final = S.final, image = padPicture(), sent = performance.now();
  S.final = false;
  S.inflight++;
  S.lastSend = sent;
  try {
    const point = S.pin && S.pin.kind === "point" ? [S.pin.level, S.pin.index] : null;
    const body = JSON.stringify({image: btoa(String.fromCharCode.apply(null, image)), final, point});
    const answer = await (await fetch("/api/see", {method: "POST", body})).json();
    if (!answer.ok) throw new Error(answer.error);
    S.round = 0.7 * S.round + 0.3 * (performance.now() - sent);  // время круга «рисунок → ответ», сглаженное
    if (number > S.applied && (DRAW.strokes.length || S.example)) {
      S.applied = number;
      S.updates.push(performance.now());
      S.image = image;
      remember();
      S.sight = prepareSight(answer);
      S.sightId = (S.sightId || 0) + 1;  // номер рисунка: по нему помнятся слои карт и картинки кусочков
      S.sprites = null;
      // Вспышка новых кластеров — только когда штрих закончен: пока кисть ведут, карты не перерисовываются по кадрам.
      if (final) S.flashAt = performance.now();
      // У выбранной точки карты кусочки нового рисунка другие: их список приходит в том же ответе.
      if (S.pin && S.pin.kind === "point") {
        const same = point && point[0] === S.pin.level && point[1] === S.pin.index;
        S.pin.keys = same && answer.point ? Array.from(dec(answer.point.keyholes, Uint16Array)) : null;
        if (!S.pin.keys) await loadPointPieces();
      }
      if (S.wishPin) {
        // ?pin=уровень:кусочек — выбрать кусочек сразу (для снимков страницы).
        const [level, keyhole] = S.wishPin.split(":").map(Number);
        if (level >= 1 && level <= S.sight.levels.length) pin(level - 1, Math.min(keyhole || 0, S.sight.levels[level - 1].K - 1));
        S.wishPin = null;
      } else if (S.pin && S.pin.kind === "piece") {
        // Выбранный кусочек держится за место на рисунке: после штриха — ближайший кусочек того же уровня.
        // Выбранные кластер и точка карты от рисунка не зависят: их кусочки находятся заново (pinnedPieces).
        const V = S.sight.levels[S.pin.level];
        if (!V || !V.K) S.pin = null;
        else {
          let best = 0, least = Infinity;
          for (let k = 0; k < V.K; k++) {
            const d = (V.centre[2 * k] - S.pin.y) ** 2 + (V.centre[2 * k + 1] - S.pin.x) ** 2;
            if (d < least) { least = d; best = k; }
          }
          S.pin.keyhole = best;
        }
      }
      render();
    }
  } catch (error) {
    $("status").textContent = `ошибка сервера: ${error.message}`;
  }
  S.inflight--;
  pump();
}

// Сколько рабочих процессов сервера готово: столько запросов рисунка идёт сразу. Процессы поднимаются в фоне после
// запуска — страница спрашивает, пока не поднимутся все.
async function watchWorkers() {
  try {
    const status = await (await fetch("/api/status")).json();
    S.workers = status;
    S.parallel = Math.max(1, status.ready || 0);
    renderStatus();
    if (status.ready < status.workers && !status.failed) setTimeout(watchWorkers, 2000);
  } catch (error) { setTimeout(watchWorkers, 5000); }
}

// Кластеры, которых не было до этого штриха, вспыхивают: видно, как кусочки набиваются в кластеры.
function remember() {
  if (!S.sight) { S.previous = []; return; }
  S.previous = S.sight.levels.map(V => V.detCount);
}

// ================================================================== уровни: карточки
function buildCards() {
  const box = $("levels");
  box.innerHTML = "";
  box.style.gridTemplateColumns = `repeat(${S.model.levels.length + (S.model.primary ? 1 : 0)}, minmax(0, 1fr))`;
  S.primaryCard = S.model.primary ? buildPrimaryCard(box) : null;
  S.cards = S.model.levels.map((M, n) => {
    const card = document.createElement("article");
    card.className = "level";
    card.innerHTML = `<div class="head"><span class="badge">${M.number}</span><b>Уровень ${M.number}</b><span class="chips"></span></div>
      <div class="top"><canvas class="pieces"></canvas><div class="stats"></div></div>
      <div class="mapwrap"><canvas class="map"></canvas><div class="zoom"><button data-z="1.4">+</button><button data-z="0.7">−</button><button data-z="0">⤢</button></div></div>`;
    box.appendChild(card);
    const item = {card, n, M, pieces: card.querySelector(".pieces"), map: card.querySelector(".map"), chips: card.querySelector(".chips"),
      stats: card.querySelector(".stats"), view: null, keep: {}};
    interactive(item);
    card.querySelectorAll(".zoom button").forEach(button => button.addEventListener("click", () => {
      const z = +button.dataset.z;
      if (!z) item.view = fit(item); else zoomAt(item, item.map.width / 2, item.map.height / 2, z);
      drawMap(item);
    }));
    return item;
  });
  resize();
}

function resize() {
  for (const item of (S.primaryCard ? [S.primaryCard] : []).concat(S.cards)) {
    const wrap = item.map.parentElement, side = Math.max(120, Math.min(wrap.clientWidth, wrap.clientHeight));
    item.map.style.width = item.map.style.height = side + "px";
    item.map.width = item.map.height = Math.round(side * DPR);
    const pieceSide = item.pieces.clientWidth;
    item.pieces.width = item.pieces.height = Math.round(pieceSide * DPR);
    item.view = fit(item);
    item.keep = {};
    item.layer = null;
  }
  render();
}
window.addEventListener("resize", () => { clearTimeout(S.resizeTimer); S.resizeTimer = setTimeout(resize, 120); });

function fit(item) {
  const M = item.M, size = item.map.width, s = size / (Math.max(M.width, M.height) * 1.02);
  return {s, ox: (size - M.width * s) / 2, oy: (size - M.height * s) / 2};
}

function zoomAt(item, mx, my, factor) {
  const M = item.M, view = item.view;
  const next = Math.min(120 * DPR, Math.max(item.map.width / (Math.max(M.width, M.height) * 1.6), view.s * factor));
  view.ox = mx - (mx - view.ox) * next / view.s; view.oy = my - (my - view.oy) * next / view.s; view.s = next;
}

// ================================================================== уровень 0: первичная карта (пинвилы)
// Точка первичной карты — углопозиция обучающей цифры: её первичный код — угол и место (§4.6). Раскладка ставит рядом
// штрихи с близкими углом и местом (§5); цвет точки — угол её штриха, поэтому на карте видны пинвилы (§5.13.2).
// Штрихи рисунка зажигают точки карты (§6.1): зажжённые горят своим цветом, остальные приглушены; кольца — сработавшие
// детекторы (§6.5). В счёте уровней первичная карта не участвует — она показывает первичное кодирование.
function buildPrimaryCard(box) {
  const card = document.createElement("article");
  card.className = "level";
  card.innerHTML = `<div class="head"><span class="badge">0</span><b>Уровень 0</b><span class="chips"></span></div>
    <div class="top"><canvas class="pieces"></canvas><div class="stats"></div></div>
    <div class="mapwrap"><canvas class="map"></canvas><div class="zoom"><button data-z="1.4">+</button><button data-z="0.7">−</button><button data-z="0">⤢</button></div></div>`;
  box.appendChild(card);
  const item = {card, M: S.model.primary, pieces: card.querySelector(".pieces"), map: card.querySelector(".map"),
    chips: card.querySelector(".chips"), stats: card.querySelector(".stats"), view: null, keep: {}, under: -1};
  const canvas = item.map;
  let drag = null;
  card.querySelectorAll(".zoom button").forEach(button => button.addEventListener("click", () => {
    const z = +button.dataset.z;
    if (!z) item.view = fit(item); else zoomAt(item, canvas.width / 2, canvas.height / 2, z);
    drawPrimaryMap(item);
  }));
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect(), k = canvas.width / rect.width;
    zoomAt(item, (event.clientX - rect.left) * k, (event.clientY - rect.top) * k, Math.pow(1.0015, -event.deltaY));
    drawPrimaryMap(item);
  }, {passive: false});
  canvas.addEventListener("pointerdown", event => { drag = [event.clientX, event.clientY]; canvas.style.cursor = "grabbing"; });
  window.addEventListener("pointerup", () => { if (drag) { drag = null; canvas.style.cursor = "grab"; } });
  window.addEventListener("pointermove", event => {
    if (!drag) return;
    const k = canvas.width / canvas.getBoundingClientRect().width;
    item.view.ox += (event.clientX - drag[0]) * k; item.view.oy += (event.clientY - drag[1]) * k; drag = [event.clientX, event.clientY];
    drawPrimaryMap(item);
  });
  // Точка под курсором: её штрих (угол, место, цифра) — в строке заголовка карточки, без всплывающих окон.
  canvas.addEventListener("pointermove", event => {
    if (drag) return;
    const rect = canvas.getBoundingClientRect(), k = canvas.width / rect.width, view = item.view, M = item.M;
    const x = Math.floor(((event.clientX - rect.left) * k - view.ox) / view.s), y = Math.floor(((event.clientY - rect.top) * k - view.oy) / view.s);
    const i = x >= 0 && y >= 0 && x < M.width && y < M.height ? M.index[y * M.width + x] : -1;
    if (i !== item.under) { item.under = i; renderPrimaryStats(item); drawPrimaryPicture(item); }
  });
  canvas.addEventListener("pointerleave", () => { if (item.under !== -1) { item.under = -1; renderPrimaryStats(item); drawPrimaryPicture(item); } });
  canvas.addEventListener("dblclick", () => { item.view = fit(item); drawPrimaryMap(item); });
  return item;
}

// Рисунок уровня 0: сам рисунок и его углопозиции, цвет — угол (тот же, что у точек первичной карты). Точка карты под
// курсором показана своим штрихом — белым, на своём месте картинки.
function drawPrimaryPicture(item) {
  const ctx = item.pieces.getContext("2d"), {W, unit, px} = pieceGeometry(item), M = item.M;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, W);
  if (S.image) {
    const tile = document.createElement("canvas"); tile.width = tile.height = DRAW.side;
    const t = tile.getContext("2d"), data = t.createImageData(DRAW.side, DRAW.side);
    for (let i = 0; i < S.image.length; i++) { const v = S.image[i] * 0.45; data.data.set([v, v, v, 255], 4 * i); }
    t.putImageData(data, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.drawImage(tile, 0, 0, W, W);
  }
  ctx.lineCap = "round";
  const sight = S.sight;
  if (sight) {
    const st = sight.sticks;
    for (let i = 0; i < st.count; i++) stick(ctx, px(st.x[i]), px(st.y[i]), unit * S.model.cell, st.angle[i], 0.95, null, st.span[i]);
  }
  if (item.under >= 0) stick(ctx, px(M.x[item.under]), px(M.y[item.under]), unit * S.model.cell * 1.6, M.angle[item.under], 1, "#ffffff", 0);
}

function drawPrimaryMap(item) {
  const canvas = item.map, ctx = canvas.getContext("2d"), M = item.M, view = item.view, W = canvas.width, H = canvas.height, s = view.s;
  const V = S.sight && S.sight.primary;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  // Без рисунка карта видна целиком; с рисунком точки приглушены, а зажжённые горят: яркость растёт от доли
  // DASHBOARD_LIT_FLOOR диапазона «λ_a … сильнейшая активация» до его верха (как у карт уровней).
  const low = M.activation, span = V ? Math.max(1e-6, V.actHigh - low) : 1, floor = S.model.litFloor, dim = V ? 0.24 : 0.9;
  const x0 = Math.max(0, Math.floor(-view.ox / s)), x1 = Math.min(M.width - 1, Math.floor((W - view.ox) / s));
  const y0 = Math.max(0, Math.floor(-view.oy / s)), y1 = Math.min(M.height - 1, Math.floor((H - view.oy) / s));
  const size = Math.max(1, s), marks = s >= 12 * DPR;
  ctx.lineCap = "round";
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = M.index[y * M.width + x];
    if (i < 0) continue;
    let alpha = dim;
    if (V && V.pointAct[i] > 0) {
      const t = (V.pointAct[i] - low) / span;
      if (t >= floor) alpha = dim + (1 - dim) * Math.pow(Math.min(1, (t - floor) / Math.max(1e-6, 1 - floor)), 1.4);
    }
    const left = view.ox + x * s, top = view.oy + y * s;
    ctx.globalAlpha = alpha; ctx.fillStyle = M.colour[i]; ctx.fillRect(left, top, size, size);
    if (marks) {
      // Крупно: в клетке — сам штрих точки (отрезок вдоль линии).
      const along = (M.angle[i] + 90) * Math.PI / 180, half = s * 0.36;
      ctx.globalAlpha = Math.min(1, alpha + 0.35); ctx.strokeStyle = "#000"; ctx.lineWidth = Math.max(1, s / 7);
      ctx.beginPath(); ctx.moveTo(left + s / 2 - Math.cos(along) * half, top + s / 2 + Math.sin(along) * half);
      ctx.lineTo(left + s / 2 + Math.cos(along) * half, top + s / 2 - Math.sin(along) * half); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  if (item.under >= 0) {
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * DPR;
    ctx.strokeRect(view.ox + M.cells[2 * item.under + 1] * s - 1, view.oy + M.cells[2 * item.under] * s - 1, size + 2, size + 2);
  }
  if (!V) return;
  // Кластеры — сработавшие детекторы (§6.5): цвет — слой, толщина — сколько штрихов рисунка в него набилось.
  const cx = d => view.ox + (M.det[5 * d + 1] + 0.5) * s, cy = d => view.oy + (M.det[5 * d] + 0.5) * s;
  for (const d of V.ranked) {
    if (!V.top.has(d)) continue;
    ctx.strokeStyle = LAYER[M.layer[d] % LAYER.length];
    ctx.globalAlpha = 0.55 + 0.45 * Math.min(1, V.detLevel[d]);
    ctx.lineWidth = Math.max(1.5, Math.min(6, 1 + Math.log2(1 + V.detCount[d]))) * DPR;
    ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(2, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function renderPrimaryStats(item) {
  const M = item.M, V = S.sight && S.sight.primary;
  const under = item.under >= 0
    ? `точка: угол ${M.angle[item.under].toFixed(0)}° · место x ${M.x[item.under].toFixed(1)}, y ${M.y[item.under].toFixed(1)} · из цифры «${M.labels[item.under]}»` : "";
  item.chips.textContent = under || (V ? `штрихов ${V.sticks} · точек ${V.litPoints} · кластеров ${V.ranked.length}` : "первичная карта: цвет — угол штриха");
  const rows = [
    `<div class="row"><span>точек на карте</span><b>${M.points}</b></div>`,
    `<div class="row"><span>сетка карты</span><b>${M.width} × ${M.height}</b></div>`,
    `<div class="row"><span>период угла</span><b>${M.period.toFixed(0)}°</b></div>`,
  ];
  if (V) rows.push(
    `<div class="row"><span>штрихов рисунка</span><b>${V.sticks}</b></div>`,
    `<div class="row"><span>зажжено точек</span><b>${V.litPoints} из ${M.points}</b></div>`,
    `<div class="row"><span>пар «штрих → точка»</span><b>${V.litPairs}</b></div>`,
    `<div class="row"><span>срабатываний</span><b>${V.firedTotal}</b></div>`,
    `<div class="row"><span>кластеров набито</span><b>${V.ranked.length}</b></div>`);
  rows.push(`<div class="clusters"><div class="cl">точка — штрих: угол и место</div><div class="cl">цвет — угол штриха</div></div>`);
  item.stats.innerHTML = rows.join("");
}

// ================================================================== рисунок уровня: кусочки
function pieceGeometry(item) {
  const W = item.pieces.width, unit = W / DRAW.side, cell = S.model.cell;
  return {W, unit, px: x => (x + 0.5) * cell * unit, r: e => e * cell * unit};
}

function drawPieces(item) {
  const ctx = item.pieces.getContext("2d"), {W, unit, px, r} = pieceGeometry(item);
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, W);
  if (!S.image) return;
  const tile = document.createElement("canvas"); tile.width = tile.height = DRAW.side;
  const t = tile.getContext("2d"), data = t.createImageData(DRAW.side, DRAW.side);
  for (let i = 0; i < S.image.length; i++) { const v = S.image[i] * 0.45; data.data.set([v, v, v, 255], 4 * i); }
  t.putImageData(data, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(tile, 0, 0, W, W);
  const sight = S.sight;
  if (!sight) return;
  ctx.lineCap = "round";
  const V = sight.levels[item.n];
  const marks = highlights();
  // Штрихи — главное на рисунке: рисуются последними, поверх полей, точек кусочков и подсветки (автор 2026-11-17:
  // «штрихи поверх всего, а точки уменьши»). Элементы уровня — детали уровня ниже (П.1, §6.6); на уровне 1 с уровнем 0
  // в цикле это детали уровня 0, без него — сами углопозиции.
  // Картинка уровня выше первого собрана из деталей уровня ниже (П.1, §6.6): элементы её окон — детали, прошедшие
  // торможение. Каждая деталь — штрихи окна памяти, на которое её кусочек больше всего похож, на месте кусочка;
  // цвет — цифра этого окна, яркость — сила детали. Когда выбран путь, детали вне его гаснут.
  const strokes = () => {
    if (item.n === 0 && levelZeroDetails(sight)) {
      // Деталь уровня 0 — штрих памяти уровня 0, который штрих рисунка зажёг сильнее всего: его наклон на месте штриха
      // рисунка; цвет — цифра его картинки, яркость — сила детали. Так же, как детали уровней выше; вне пути гаснут.
      const st = sight.sticks, M0 = S.model.primary, Q = sight.primary, focus = marks.size > 0;
      const path = focus ? levelOnePath(marks) : null, cellPx = unit * S.model.cell;
      let strongest = 0;
      for (let i = 0; i < st.count; i++) strongest = Math.max(strongest, Q.stickStrength[i]);
      const order = Array.from({length: st.count}, (_, i) => i).sort((a, b) => Q.stickStrength[a] - Q.stickStrength[b]);
      for (const i of order) {
        if (focus && !path.has(i)) continue;
        const p = Q.stickPoint[i];
        if (p < 0) { stick(ctx, px(st.x[i]), px(st.y[i]), cellPx, st.angle[i], 0.25, "#555", 0); continue; }
        const weight = strongest > 0 ? Math.pow(Q.stickStrength[i] / strongest, 2.5) : 0;
        stick(ctx, px(st.x[i]), px(st.y[i]), cellPx, M0.angle[p], 0.03 + 0.92 * weight, CLASS[M0.labels[p]], 0);
      }
      return;
    }
    if (item.n === 0) {
      const st = sight.sticks;
      for (let i = 0; i < st.count; i++) stick(ctx, px(st.x[i]), px(st.y[i]), unit * S.model.cell, st.angle[i], 0.85, null, st.span[i]);
      return;
    }
    if (!V) return;
    const below = sight.levels[item.n - 1], Mb = S.model.levels[item.n - 1], path = marks.get(item.n - 1), focus = marks.size > 0;
    const cellPx = unit * S.model.cell;
    // Слабые детали — снизу и почти не видны, сильные — сверху и ярко: прозрачность растёт как (сила / сильнейшая)^2.5;
    // толщина у всех одна (автор: «толщину не увеличивай — только прозрачность»). При выборе и наведении детали вне
    // пути не рисуются вовсе, а детали пути видны так же, как без выбора, — по своей силе: окно верхнего уровня
    // вмещает почти все детали (74 из 100), и подсветка «всего пути» ярче обычного заливала бы рисунок шумом.
    const used = [...new Set(V.entryElement)].sort((a, b) => below.strength[a] - below.strength[b]);
    let strongest = 0;
    for (const e of used) strongest = Math.max(strongest, below.strength[e]);
    for (const e of used) {
      if (focus && !(path && path.has(e))) continue;
      const i = strongestPoint(below, e);
      if (i < 0) continue;
      const weight = strongest > 0 ? Math.pow(below.strength[e] / strongest, 2.5) : 0;
      fragment(ctx, Mb, i, px(below.centre[2 * e + 1]), px(below.centre[2 * e]), cellPx, 0.03 + 0.92 * weight, CLASS[Mb.labels[i]],
        Math.max(1, cellPx / 5));
    }
  };
  if (!V) {
    strokes();
    ctx.fillStyle = "#9aa3b2"; ctx.font = `${Math.round(W / 16)}px Segoe UI`; ctx.textAlign = "center"; ctx.fillText("до уровня не дошло", W / 2, W / 2);
    return;
  }
  let top = 0;
  for (let k = 0; k < V.K; k++) top = Math.max(top, V.strength[k]);
  // Поля кусочков — еле видно (их десятки, и они перекрываются); сам кусочек — маленькая точка в центре окна, под
  // штрихами; цвет — сила его детали (П.1): синий — слабая, жёлтый — сильная, серый — ни один детектор не сработал.
  ctx.strokeStyle = "rgba(110,168,255,0.07)"; ctx.lineWidth = Math.max(1, W / 360);
  for (let k = 0; k < V.K; k++) {
    ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), r(V.extent[k]), 0, 2 * Math.PI); ctx.stroke();
  }
  const dot = Math.max(1.2, W / 280);
  for (let k = 0; k < V.K; k++) {
    const share = top > 0 ? V.strength[k] / top : 0;
    ctx.fillStyle = V.strength[k] > 0 ? `hsl(${(220 - 170 * share).toFixed(0)},90%,${(45 + 20 * share).toFixed(0)}%)` : "#555";
    ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), dot, 0, 2 * Math.PI); ctx.fill();
  }
  const levelMarks = marks.get(item.n);
  if (levelMarks) {
    // Цепочка (из чего собран, куда вошёл) — точка и тонкий круг; выбранный и под курсором — жирный круг.
    const ordered = [...levelMarks].sort((a, b) => (a[1] === "member" ? 0 : 1) - (b[1] === "member" ? 0 : 1));
    for (const [k, kind] of ordered) {
      const colour = kind === "pin" ? PIN : kind === "hover" ? HOVER : MEMBER, member = kind === "member";
      ctx.strokeStyle = colour; ctx.globalAlpha = member ? 0.35 : 1; ctx.lineWidth = member ? Math.max(1, W / 300) : Math.max(2.5, W / 120);
      ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), r(V.extent[k]), 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillStyle = colour;
      ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), Math.max(1.8, W / (member ? 220 : 150)), 0, 2 * Math.PI); ctx.fill();
    }
  }
  // Кусочки, набившиеся в кластер под курсором.
  if (S.hover && S.hover.kind === "cluster" && S.hover.level === item.n) {
    const d = S.hover.index, B = V.byDetector;
    ctx.fillStyle = "rgba(255,90,209,0.18)"; ctx.strokeStyle = HOVER; ctx.lineWidth = Math.max(2, W / 160);
    for (let j = B.start[d]; j < B.start[d + 1]; j++) {
      const k = V.firedKeyhole[B.order[j]];
      ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), r(V.extent[k]), 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
  }
  strokes();
  // Углопозиции выбранного кусочка уровня 1 (без уровня 0 в цикле; с ним путь показывают сами детали уровня 0).
  if (item.n === 0 && levelMarks && !levelZeroDetails(sight)) {
    for (const [k, kind] of levelMarks) {
      if (kind === "member") continue;
      for (let j = V.entries.start[k]; j < V.entries.start[k + 1]; j++) {
        const e = V.entryElement[V.entries.order[j]], st = sight.sticks;
        stick(ctx, px(st.x[e]), px(st.y[e]), unit * S.model.cell * 1.2, st.angle[e], 1, "#ffffff", st.span[e] / 1.2);
      }
    }
  }
}

// Есть ли у рисунка детали уровня 0: уровень 0 в цикле, и сервер прислал у штрихов их точки карты.
function levelZeroDetails(sight) {
  return Boolean(S.model.primary && S.model.primary.inCycle && sight && sight.primary && sight.primary.stickPoint);
}

// Штрихи на пути выбранного: элементы отмеченных кусочков уровня 1 (выбранного, под курсором и их состава).
function levelOnePath(marks) {
  const V = S.sight.levels[0], path = new Set(), level = marks.get(0);
  if (!V || !level) return path;
  for (const k of level.keys()) for (let j = V.entries.start[k]; j < V.entries.start[k + 1]; j++) path.add(V.entryElement[V.entries.order[j]]);
  return path;
}

// Подсветка кусочков по уровням: выбранный (pin), под курсором (hover), его состав ниже и куда он вошёл выше.
function highlights(withHover = true) {
  const sight = S.sight;
  if (!sight) return new Map();
  // Подсветка одна на рисунок, выбор и наведение, а спрашивают её все рисунки и карты уровней: ответ помнится.
  const hoverKey = withHover && S.hover && S.hover.kind === "piece" ? `${S.hover.level}:${S.hover.index}` : "";
  const slot = withHover ? "marksHover" : "marksPin", key = pinKey() + "|" + hoverKey, known = S[slot];
  if (known && known.sight === sight && known.key === key) return known.marks;
  const marks = new Map();
  S[slot] = {sight, key, marks};
  const add = (n, k, kind) => { if (!marks.has(n)) marks.set(n, new Map()); const m = marks.get(n); if (!m.has(k) || kind !== "member") m.set(k, kind); };
  const chain = (n, k, kind) => {
    add(n, k, kind);
    let level = [k];
    for (let m = n; m > 0; m--) { // вниз: из каких кусочков уровня m − 1 собран
      const V = sight.levels[m], next = new Set();
      for (const key of level) for (let j = V.entries.start[key]; j < V.entries.start[key + 1]; j++) next.add(V.entryElement[V.entries.order[j]]);
      for (const e of next) add(m - 1, e, "member");
      level = [...next];
    }
    level = [k];
    for (let m = n + 1; m < sight.levels.length; m++) { // вверх: в какие кусочки уровня m вошёл
      const V = sight.levels[m], next = new Set();
      for (const key of level) for (let j = V.parents.start[key]; j < V.parents.start[key + 1]; j++) next.add(V.entryKeyhole[V.parents.order[j]]);
      for (const e of next) add(m, e, "member");
      level = [...next];
    }
  };
  const pinned = pinnedPieces();
  if (pinned) for (const k of pinned.keys) chain(pinned.level, k, "pin");
  if (withHover && S.hover && S.hover.kind === "piece") chain(S.hover.level, S.hover.index, "hover");
  return marks;
}

function pieceAt(item, event) {
  const V = S.sight?.levels[item.n];
  if (!V) return -1;
  const box = item.pieces.getBoundingClientRect(), k = item.pieces.width / box.width, {px, r} = pieceGeometry(item);
  const x = (event.clientX - box.left) * k, y = (event.clientY - box.top) * k;
  let best = -1, least = Infinity;
  for (let i = 0; i < V.K; i++) {
    const d = Math.hypot(px(V.centre[2 * i + 1]) - x, px(V.centre[2 * i]) - y);
    if (d <= Math.max(r(V.extent[i]), item.pieces.width / 40) && d < least) { least = d; best = i; }
  }
  return best;
}

// ================================================================== карта уровня: кластеры
function drawMap(item) {
  const canvas = item.map, ctx = canvas.getContext("2d"), M = item.M, view = item.view, W = canvas.width, H = canvas.height, s = view.s;
  // Фон карты от рисунка не зависит: один раз на положение и масштаб.
  const key = [W, H, s.toFixed(4), view.ox.toFixed(2), view.oy.toFixed(2)].join("|");
  if (item.keep.key !== key) {
    const back = item.keep.canvas || document.createElement("canvas");
    back.width = W; back.height = H;
    const b = back.getContext("2d");
    b.fillStyle = "#000"; b.fillRect(0, 0, W, H); b.lineCap = "round";
    const x0 = Math.max(0, Math.floor(-view.ox / s)), x1 = Math.min(M.width - 1, Math.floor((W - view.ox) / s));
    const y0 = Math.max(0, Math.floor(-view.oy / s)), y1 = Math.min(M.height - 1, Math.floor((H - view.oy) / s));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = M.index[y * M.width + x];
      if (i < 0) continue;
      const left = view.ox + x * s, top = view.oy + y * s;
      if (s >= 16 * DPR) {
        b.strokeStyle = CLASS[M.labels[i]]; b.globalAlpha = 0.35; b.lineWidth = 1; b.strokeRect(left + 0.5, top + 0.5, s - 1, s - 1); b.globalAlpha = 1;
        glyph(b, M, i, left, top, s);
      } else {
        b.fillStyle = CLASS[M.labels[i]]; b.globalAlpha = 0.2; b.fillRect(left, top, Math.max(1, s), Math.max(1, s)); b.globalAlpha = 1;
      }
    }
    item.keep = {key, canvas: back};
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const V = S.sight?.levels[item.n];
  if (!V) { ctx.drawImage(item.keep.canvas, 0, 0); return; }
  const cx = d => view.ox + (M.det[5 * d + 1] + 0.5) * s, cy = d => view.oy + (M.det[5 * d] + 0.5) * s;
  // Отклик карты на рисунок (зажжённые точки и кластеры) от наведения и выбора не зависит: он рисуется один раз на
  // рисунок и вид карты в свой слой, дальше слой копируется. Пока гаснет вспышка новых кластеров — рисуется прямо.
  const glow = S.flashAt ? Math.max(0, 1 - (performance.now() - S.flashAt) / 900) : 0;
  if (glow > 0) {
    ctx.drawImage(item.keep.canvas, 0, 0);
    drawResponse(ctx, item, V, glow);
  } else {
    const layerKey = key + "|" + S.sightId;
    if (!item.layer || item.layer.key !== layerKey) {
      const layer = (item.layer && item.layer.canvas) || document.createElement("canvas");
      layer.width = W; layer.height = H;
      const l = layer.getContext("2d");
      l.drawImage(item.keep.canvas, 0, 0);
      drawResponse(l, item, V, 0);
      item.layer = {key: layerKey, canvas: layer};
    }
    ctx.drawImage(item.layer.canvas, 0, 0);
  }
  drawMapMarks(ctx, item, V);
}

// Отклик карты на рисунок: зажжённые точки (§6.1) и сработавшие детекторы — кластеры (§6.5).
function drawResponse(ctx, item, V, glow) {
  const M = item.M, view = item.view, s = view.s;
  const cx = d => view.ox + (M.det[5 * d + 1] + 0.5) * s, cy = d => view.oy + (M.det[5 * d] + 0.5) * s;
  // Зажжённые точки (§6.1). Рисунок зажигает почти всю карту (порог λ_a уровня низкий), поэтому светятся только
  // сильные совпадения: яркость растёт от доли DASHBOARD_LIT_FLOOR диапазона «λ_a … сильнейшая активация» до его верха.
  const low = M.activation, span = Math.max(1e-6, V.actHigh - low), floor = S.model.litFloor;
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#fff";
  for (let p = 0; p < M.points; p++) {
    const a = V.pointAct[p];
    if (a <= 0) continue;
    const t = (a - low) / span;
    if (t < floor) continue;
    const left = view.ox + M.cells[2 * p + 1] * s, top = view.oy + M.cells[2 * p] * s;
    ctx.globalAlpha = Math.pow(Math.min(1, (t - floor) / Math.max(1e-6, 1 - floor)), 1.4);
    if (s >= 16 * DPR) { ctx.lineWidth = Math.max(1.5, s / 10); ctx.strokeRect(left + 1, top + 1, s - 2, s - 2); }
    else ctx.fillRect(left, top, Math.max(1.5, s), Math.max(1.5, s));
  }
  ctx.globalAlpha = 1;
  // Кластеры — сработавшие детекторы (§6.5): цвет — слой, толщина — сколько кусочков рисунка в него набилось.
  // Жирно — DASHBOARD_TOP_CLUSTERS самых набитых, остальные — тонкой линией.
  const fresh = S.previous[item.n];
  ctx.lineWidth = DPR;
  for (const d of V.ranked) {
    if (V.top.has(d)) continue;
    ctx.strokeStyle = LAYER[M.layer[d] % LAYER.length]; ctx.globalAlpha = 0.13;
    ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(2, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
  }
  for (const d of V.ranked) {
    if (!V.top.has(d)) continue;
    const count = V.detCount[d], isNew = glow > 0 && (!fresh || !fresh[d]);
    ctx.strokeStyle = LAYER[M.layer[d] % LAYER.length];
    ctx.globalAlpha = 0.55 + 0.45 * Math.min(1, V.detLevel[d]);
    ctx.lineWidth = Math.max(1.5, Math.min(6, 1 + Math.log2(1 + count))) * DPR;
    if (isNew) { ctx.shadowColor = "#fff"; ctx.shadowBlur = 14 * glow * DPR; }
    ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(2, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
    ctx.shadowBlur = 0;
    if (s * M.det[5 * d + 2] > 22 * DPR) {
      ctx.globalAlpha = 1; ctx.fillStyle = ctx.strokeStyle; ctx.font = `${Math.round(11 * DPR)}px Segoe UI`; ctx.textAlign = "center";
      ctx.fillText(String(count), cx(d), cy(d) - M.det[5 * d + 2] * s - 3 * DPR);
    }
  }
  ctx.globalAlpha = 1;
}

// Выбор и наведение поверх отклика карты.
function drawMapMarks(ctx, item, V) {
  const M = item.M, view = item.view, s = view.s;
  const cx = d => view.ox + (M.det[5 * d + 1] + 0.5) * s, cy = d => view.oy + (M.det[5 * d] + 0.5) * s;
  // Выбранный кусочек и кусочек под курсором: их точки и кластеры.
  const own = [];
  if (S.pin && S.pin.kind === "piece" && S.pin.level === item.n) own.push([S.pin.keyhole, PIN]);
  if (S.hover && S.hover.kind === "piece" && S.hover.level === item.n) own.push([S.hover.index, HOVER]);
  for (const [k, colour] of own) {
    ctx.strokeStyle = colour; ctx.fillStyle = colour;
    for (let j = V.lit.start[k]; j < V.lit.start[k + 1]; j++) {
      const i = V.lit.order[j], p = V.litPoint[i], left = view.ox + M.cells[2 * p + 1] * s, top = view.oy + M.cells[2 * p] * s;
      ctx.globalAlpha = 0.4 + 0.6 * V.litValue[i];
      if (s >= 6 * DPR) { ctx.lineWidth = Math.max(1.5, s / 7); ctx.strokeRect(left + 1, top + 1, s - 2, s - 2); }
      else ctx.fillRect(left - 1, top - 1, Math.max(3, s + 2), Math.max(3, s + 2));
    }
    // Кусочек зажигает сотни кластеров: на карте — его DASHBOARD_OWN_CLUSTERS сильнейших (по уровню активации E).
    ctx.globalAlpha = 1; ctx.lineWidth = 2.5 * DPR;
    const mine = [];
    for (let j = V.fired.start[k]; j < V.fired.start[k + 1]; j++) mine.push(V.fired.order[j]);
    mine.sort((a, b) => V.firedLevel[b] - V.firedLevel[a]);
    for (const i of mine.slice(0, S.model.ownClusters)) {
      const d = V.firedDetector[i];
      ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(3, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
    }
  }
  if (S.hover && S.hover.kind === "cluster" && S.hover.level === item.n) {
    const d = S.hover.index;
    ctx.setLineDash([6 * DPR, 4 * DPR]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * DPR;
    ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(3, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
  }
  // Путь выбранного по этой карте: сильнейший кластер каждого кусочка цепочки.
  const path = S.pin ? highlights(false).get(item.n) : null;
  if (path) {
    const seen = new Set();
    ctx.strokeStyle = MEMBER; ctx.lineWidth = 1.5 * DPR; ctx.globalAlpha = 0.85;
    for (const [k] of path) {
      const d = V.best[k];
      if (d < 0 || seen.has(d)) continue;
      seen.add(d);
      ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(3, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  // Выбранные кластер или точка карты.
  if (S.pin && S.pin.kind !== "piece" && S.pin.level === item.n) {
    ctx.strokeStyle = PIN; ctx.lineWidth = 3 * DPR;
    if (S.pin.kind === "cluster") {
      const d = S.pin.index;
      ctx.beginPath(); ctx.arc(cx(d), cy(d), Math.max(4, M.det[5 * d + 2] * s), 0, 2 * Math.PI); ctx.stroke();
    } else {
      const p = S.pin.index, left = view.ox + M.cells[2 * p + 1] * s, top = view.oy + M.cells[2 * p] * s;
      ctx.strokeRect(left - 2 * DPR, top - 2 * DPR, s + 4 * DPR, s + 4 * DPR);
    }
  }
}

function mapAt(item, event) {
  const canvas = item.map, box = canvas.getBoundingClientRect(), k = canvas.width / box.width, view = item.view, M = item.M;
  const mx = (event.clientX - box.left) * k, my = (event.clientY - box.top) * k;
  const fx = (mx - view.ox) / view.s, fy = (my - view.oy) / view.s, x = Math.floor(fx), y = Math.floor(fy);
  const V = S.sight?.levels[item.n];
  if (V) {
    // Кластер под курсором — самый узкий из сработавших, в поле которого курсор.
    let best = -1, radius = Infinity;
    for (let d = 0; d < M.count; d++) {
      if (!V.detCount[d]) continue;
      const r = M.det[5 * d + 2], dist = Math.hypot(fy - 0.5 - M.det[5 * d], fx - 0.5 - M.det[5 * d + 1]);
      if (dist <= Math.max(r, 0.8) && r < radius) { radius = r; best = d; }
    }
    if (best >= 0) return {kind: "cluster", level: item.n, index: best};
  }
  const i = x >= 0 && y >= 0 && x < M.width && y < M.height ? M.index[y * M.width + x] : -1;
  return i >= 0 ? {kind: "point", level: item.n, index: i} : null;
}

function interactive(item) {
  const canvas = item.map;
  let drag = null;
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const box = canvas.getBoundingClientRect(), k = canvas.width / box.width;
    zoomAt(item, (event.clientX - box.left) * k, (event.clientY - box.top) * k, Math.pow(1.0015, -event.deltaY));
    drawMap(item);
  }, {passive: false});
  canvas.addEventListener("pointerdown", event => { drag = [event.clientX, event.clientY]; item.moved = 0; canvas.style.cursor = "grabbing"; hideTip(); });
  window.addEventListener("pointerup", () => { if (drag) { drag = null; canvas.style.cursor = "grab"; } });
  window.addEventListener("pointermove", event => {
    if (!drag) return;
    const k = canvas.width / canvas.getBoundingClientRect().width;
    item.moved += Math.abs(event.clientX - drag[0]) + Math.abs(event.clientY - drag[1]);
    item.view.ox += (event.clientX - drag[0]) * k; item.view.oy += (event.clientY - drag[1]) * k; drag = [event.clientX, event.clientY];
    drawMap(item);
  });
  // Щелчок (без перетаскивания) по карте выбирает точку раскладки под курсором: путь кусочков рисунка, которые её
  // зажгли, по всем уровням (список кусочков точки приходит с сервера). В пустой клетке — кластер под курсором.
  canvas.addEventListener("click", async event => {
    if (item.moved > 4 || !S.sight) return;
    const box = canvas.getBoundingClientRect(), k = canvas.width / box.width, view = item.view, M = item.M;
    const x = Math.floor(((event.clientX - box.left) * k - view.ox) / view.s), y = Math.floor(((event.clientY - box.top) * k - view.oy) / view.s);
    const i = x >= 0 && y >= 0 && x < M.width && y < M.height ? M.index[y * M.width + x] : -1;
    const target = i >= 0 ? {kind: "point", level: item.n, index: i} : mapAt(item, event);
    const same = S.pin && target && S.pin.kind === target.kind && S.pin.level === target.level && S.pin.index === target.index;
    S.pin = target && !same ? {kind: target.kind, level: target.level, index: target.index} : null;
    if (S.pin && S.pin.kind === "point") await loadPointPieces();
    render();
  });
  canvas.addEventListener("pointermove", event => { if (!drag) hover(mapAt(item, event), event); });
  canvas.addEventListener("pointerleave", () => hover(null));
  canvas.addEventListener("dblclick", () => { item.view = fit(item); drawMap(item); });
  item.pieces.addEventListener("pointermove", event => { const k = pieceAt(item, event); hover(k >= 0 ? {kind: "piece", level: item.n, index: k} : null, event); });
  item.pieces.addEventListener("pointerleave", () => hover(null));
  item.pieces.addEventListener("click", event => {
    const k = pieceAt(item, event);
    if (k >= 0 && !(S.pin && S.pin.kind === "piece" && S.pin.level === item.n && S.pin.keyhole === k)) pin(item.n, k); else S.pin = null;
    render();
  });
}

// Выбрать кусочек k уровня n; его место на рисунке запоминается, чтобы выбор пережил следующий штрих.
function pin(n, k) {
  const V = S.sight.levels[n];
  S.pin = {kind: "piece", level: n, keyhole: k, y: V.centre[2 * k], x: V.centre[2 * k + 1]};
}

function hover(target, event) {
  const same = (a, b) => (a === b) || (a && b && a.kind === b.kind && a.level === b.level && a.index === b.index);
  if (!same(target, S.hover)) {
    // Перерисовывается только то, что меняется: точка карты — ничего (лишь подсказка); кластер — рисунок и карта
    // своего уровня; кусочек — рисунки всех уровней (его цепочка) и карта своего уровня.
    const need = S.redraw || (S.redraw = {pieces: new Set(), maps: new Set()});
    for (const h of [S.hover, target]) {
      if (!h || h.kind === "point") continue;
      need.maps.add(h.level);
      if (h.kind === "piece") for (const item of S.cards) need.pieces.add(item.n); else need.pieces.add(h.level);
    }
    S.hover = target;
    cancelAnimationFrame(S.frame);
    S.frame = requestAnimationFrame(() => {
      const todo = S.redraw;
      S.redraw = null;
      if (!todo) return;
      for (const item of S.cards) { if (todo.pieces.has(item.n)) drawPieces(item); if (todo.maps.has(item.n)) drawMap(item); }
    });
    if (target && event) showTip(target, event); else hideTip();
  } else if (target && event) placeTip(event);
}

// ================================================================== подсказка
const tip = $("tip");
function hideTip() { tip.hidden = true; }
function showTip(target, event) {
  const M = S.model.levels[target.level], V = S.sight?.levels[target.level], sticks = S.model.sticks;
  tip.innerHTML = "";
  const text = document.createElement("div");
  if (target.kind === "point") {
    const i = target.index, p = M.pointPicture[i];
    tip.appendChild(glyphCanvas(M, i, 110, true));
    const view = p >= S.model.originals ? " (сдвинутый вид)" : "";
    const lit = V ? V.pointAct[i] : 0;
    text.innerHTML = `<b>Точка карты уровня ${M.number}</b><br>окно обучающей цифры <b>«${M.labels[i]}»</b>, MNIST train № ${sticks.index[p]}${view}<br>` +
      `энергия Ê = ${M.energies[i].toFixed(2)}<br>` +
      (lit > 0 ? `рисунок зажёг: a = <b>${lit.toFixed(2)}</b>, кусочков: ${V.pointCount[i]}` : "рисунок её не зажёг");
  } else if (target.kind === "cluster") {
    const d = target.index, inside = members(M, d), parts = composition(M, d).filter(c => c.share > 0).slice(0, 3);
    text.innerHTML = `<b>Кластер — детектор ${d}</b> (§6.5)<br>слой ${M.layer[d]} из ${M.thresholds.length}: λ_d = ${M.det[5 * d + 3].toFixed(3)}, ` +
      `радиус ${M.det[5 * d + 2].toFixed(1)} кл., точек ${inside.length}<br>` +
      `набилось кусочков рисунка: <b>${V.detCount[d]}</b>, сильнейший E = <b>${V.detLevel[d].toFixed(2)}</b><br>` +
      `состав: ${parts.map(c => `«${c.digit}» ${(100 * c.share).toFixed(0)} %`).join(", ") || "—"}`;
    tip.appendChild(text);
    const mosaic = document.createElement("div");
    mosaic.className = "mosaic";
    const shown = inside.slice().sort((a, b) => (V.pointAct[b] - V.pointAct[a]) || (M.energies[b] - M.energies[a])).slice(0, 12);
    for (const i of shown) mosaic.appendChild(glyphCanvas(M, i, 44, true));
    tip.appendChild(mosaic);
  } else if (target.kind === "piece" && V) {
    const k = target.index, elements = V.entries.start[k + 1] - V.entries.start[k];
    const lit = V.litCount[k], fired = V.fired.start[k + 1] - V.fired.start[k];
    let bestPoint = -1, bestValue = 0;
    for (let j = V.lit.start[k]; j < V.lit.start[k + 1]; j++) { const i = V.lit.order[j]; if (V.litValue[i] > bestValue) { bestValue = V.litValue[i]; bestPoint = V.litPoint[i]; } }
    if (bestPoint >= 0) tip.appendChild(glyphCanvas(M, bestPoint, 110, true));
    const d = V.best[k];
    text.innerHTML = `<b>Кусочек ${k} уровня ${M.number}</b> (окно П.2)<br>${elements} ${M.number === 1 ? "углопозиций" : "деталей уровня " + (M.number - 1)}, ` +
      `поле ${V.extent[k].toFixed(1)} кл.<br>зажёг точек: <b>${lit}</b>, сработало детекторов: <b>${fired}</b><br>` +
      (d >= 0 ? `сильнейший кластер: слой ${M.layer[d]}, E = ${V.bestLevel[k].toFixed(2)}` : "ни один детектор не сработал") +
      (bestPoint >= 0 ? `<br>похоже на окно цифры «${M.labels[bestPoint]}» (a = ${bestValue.toFixed(2)}) — слева` : "");
  }
  if (!text.parentElement) tip.appendChild(text);
  tip.hidden = false;
  placeTip(event);
}

function placeTip(event) {
  const w = 300, h = tip.offsetHeight || 160, vw = window.innerWidth, vh = window.innerHeight;
  tip.style.left = (event.clientX + 14 + w > vw ? event.clientX - w - 14 : event.clientX + 14) + "px";
  tip.style.top = Math.max(8, Math.min(vh - h - 8, event.clientY + 14)) + "px";
}

// ================================================================== сводки
function renderStats(item) {
  const M = item.M, V = S.sight?.levels[item.n];
  if (!V) { item.chips.textContent = S.sight ? "рисунок до уровня не дошёл" : "нарисуйте цифру"; item.stats.innerHTML = ""; return; }
  let silent = 0, mean = 0;
  for (let k = 0; k < V.K; k++) { if (V.fired.start[k + 1] === V.fired.start[k]) silent++; mean += V.strength[k]; }
  mean /= Math.max(1, V.K);
  const fresh = S.previous[item.n];
  let added = 0;
  for (let d = 0; d < M.count; d++) if (V.detCount[d] && (!fresh || !fresh[d])) added++;
  item.chips.textContent = `кусочков ${V.K} · точек ${V.litPoints} · кластеров ${V.clusters}`;
  const top = V.ranked;
  item.stats.innerHTML =
    `<div class="row"><span>кусочков (окон)</span><b>${V.K}</b></div>` +
    `<div class="row"><span>зажжено точек</span><b>${V.litPoints} из ${M.points}</b></div>` +
    `<div class="row"><span>пар «кусочек → точка»</span><b>${V.litPairs}</b></div>` +
    `<div class="row"><span>срабатываний</span><b>${V.firedTotal}</b></div>` +
    `<div class="row"><span>кластеров набито</span><b>${V.clusters}${added && fresh ? ` <span style="color:${PIN}">+${added}</span>` : ""}</b></div>` +
    `<div class="row"><span>кусочков без отклика</span><b>${silent}</b></div>` +
    `<div class="row"><span>сила деталей, средняя</span><b>${mean.toFixed(2)}</b></div>` +
    `<div class="clusters">${top.slice(0, 5).map(d => {
      const c = composition(M, d)[0];
      return `<div class="cl"><i style="border-color:${LAYER[M.layer[d] % LAYER.length]}"></i>слой ${M.layer[d]} · кусочков <b>${V.detCount[d]}</b> · «${c.digit}» ${(100 * c.share).toFixed(0)} %</div>`;
    }).join("")}</div>`;
}

// Ответ памяти (§2.2.5, §7.1.13.3): цифра с наибольшей средней долей голосов кусочков по уровням; уверенность —
// согласие уровней: все уровни назвали эту цифру, её назвали два верхних, уровни расходятся. Под ним — уровни: полоска
// знания (насколько память знает кусочки уровня) с риской порога и победитель голосов кусочков (цифра самой похожей
// точки памяти у каждого); уровни, согласные с ответом, выделены.
function renderMemory() {
  const sight = S.sight, memory = sight && sight.memory, box = $("memory");
  box.className = memory ? memory.kind : "unknown";
  if (!memory) {
    $("memoryDigit").textContent = "—"; $("memoryKind").textContent = ""; $("memoryWhy").textContent = ""; $("voteRows").innerHTML = "";
    return;
  }
  const percent = value => `${(100 * value).toFixed(0)} %`;
  $("memoryDigit").textContent = memory.digit >= 0 ? String(memory.digit) : "?";
  $("memoryKind").textContent = memory.kind === "agree" ? "все уровни согласны" : memory.kind === "mostly" ? "два верхних уровня согласны"
    : memory.kind === "split" ? "уровни расходятся" : "не знаю";
  $("memoryWhy").textContent = memory.digit >= 0
    ? `в среднем по уровням ${percent(memory.share)} кусочков за «${memory.digit}» · согласны ${memory.agreed} из ${S.model.levels.length}`
    : "ни одна точка памяти не зажглась";
  $("voteRows").innerHTML = sight.votes.map(level => {
    const used = level.digit === memory.digit && memory.digit >= 0 ? " used" : "";
    const title = `знание ${level.knowledge.toFixed(2)}, порог ${level.threshold.toFixed(2)}: память ${level.knows ? "знает" : "не знает"} кусочки уровня; голосовало ${level.voted} из ${level.keyholes}`;
    return `<span class="${used.trim()}" title="${title}">ур. ${level.number}</span>` +
      `<div class="know${level.knows ? " yes" : ""}" title="${title}"><span style="width:${(100 * Math.min(1, level.knowledge)).toFixed(1)}%"></span><i style="left:${(100 * Math.min(1, level.threshold)).toFixed(1)}%"></i></div>` +
      `<span class="${used.trim()}" title="${title}">${level.digit >= 0 ? `<b>«${level.digit}»</b> ${percent(level.share)}` : "—"}</span>`;
  }).join("");
}

function renderAnswer() {
  renderMemory();
  const bars = $("bars"), sight = S.sight;
  const p = sight ? sight.probabilities : new Array(10).fill(0), best = sight ? sight.digit : -1;
  $("digit").textContent = best >= 0 ? String(best) : "—";
  bars.innerHTML = p.map((value, digit) => `<span class="${digit === best ? "top" : ""}">${digit}</span><div class="bar"><span style="width:${(100 * value).toFixed(1)}%"></span></div><span>${(100 * value).toFixed(1)} %</span>`).join("");
  $("answerNote").textContent = sight ? (best >= 0 ? `ответ за ${Math.round(sight.ms)} мс; ${sight.sticks.count} углопозиций` : "ни один детектор не сработал — ответа нет") : "";
}

function renderTrace() {
  const body = $("traceBody"), sight = S.sight, pinned = pinnedPieces();
  const want = Boolean(sight && pinned && pinned.keys.length);
  // Пока что-то выбрано, полоса пути выше: в ней кусочки всех уровней. Высота меняется — карты пересчитываются.
  if (document.body.classList.contains("pinned") !== want) { document.body.classList.toggle("pinned", want); resize(); return; }
  body.innerHTML = "";
  if (!want) {
    body.innerHTML = pinned
      ? `<div class="note">Выбранную ${S.pin.kind === "point" ? "точку карты" : "область"} не зажёг ни один кусочек рисунка — пути нет. ` +
        `Яркие (белые) точки карты — те, что рисунок зажёг сильно.</div>`
      : `<div class="note">Щёлкните по точке на карте уровня (или по кусочку на рисунке): здесь по уровням появятся кусочки рисунка, ` +
        `которые её зажгли, и кусочки, из которых они собраны, — вплотную, в том порядке, в каком стоят в уровне выше. ` +
        `Колесо — приблизить, тянуть — сдвинуть, двойной щелчок — всё.</div>`;
    return;
  }
  const marks = highlights(false), plan = tracePlan(pinned);
  S.traceSteps = [];
  for (let n = 0; n < sight.levels.length; n++) {
    const step = document.createElement("div");
    step.className = "step";
    const title = document.createElement("div");
    title.className = "steptitle";
    title.textContent = `Уровень ${n + 1}` + (n === pinned.level ? " · выбранное" : n < pinned.level ? " · из чего собрано" : " · куда вошло");
    const canvas = document.createElement("canvas");
    step.appendChild(title);
    step.appendChild(canvas);
    body.appendChild(step);
    const tiles = n <= pinned.level ? plan.tiles(n) : aboveTiles(n, [...(marks.get(n) || new Map()).keys()]);
    const entry = {n, canvas, tiles: tiles.list, span: tiles.span};
    S.traceSteps.push(entry);
    stepEvents(entry);
    drawStep(entry);
  }
}

// Плитки уровня n в свёртке выбранного: кусочки вплотную, вложенные так, как они стоят в уровне выше. Дети кусочка —
// элементы его окна (кусочки уровня ниже, П.2, §6.6) в сетке g×g; клетка ребёнка — по его месту на рисунке
// относительно родителя (слева вверху — в левой верхней клетке). Зазор между группами больше, чем внутри группы.
const TREE_ROOTS = 4;  // у кластера и точки карты — столько сильнейших кусочков

function elementsOf(n, k) {
  const V = S.sight.levels[n], found = new Set();
  for (let j = V.entries.start[k]; j < V.entries.start[k + 1]; j++) found.add(V.entryElement[V.entries.order[j]]);
  return [...found];
}

function treeNode(n, k, sides, known) {
  // Один кусочек входит во многие окна уровня выше: его узел строится один раз.
  const key = n * 65536 + k;
  if (known.has(key)) return known.get(key);
  const node = {n, k, children: n > 0 ? elementsOf(n, k).map(e => treeNode(n - 1, e, sides, known)) : []};
  if (n > 0) sides[n] = Math.max(sides[n] || 1, Math.ceil(Math.sqrt(Math.max(1, node.children.length))));
  known.set(key, node);
  return node;
}

// Места в сетке g×g: жадно, от ближайших пар «кусочек — клетка» по нормированному месту на рисунке.
function gridPlaces(points, g) {
  const ys = points.map(p => p[0]), xs = points.map(p => p[1]);
  const y0 = Math.min(...ys), x0 = Math.min(...xs), hy = Math.max(1e-6, Math.max(...ys) - y0), hx = Math.max(1e-6, Math.max(...xs) - x0);
  const pairs = [];
  points.forEach((p, i) => {
    const u = points.length > 1 ? (p[1] - x0) / hx : 0.5, v = points.length > 1 ? (p[0] - y0) / hy : 0.5;
    for (let c = 0; c < g * g; c++) {
      const cx = g > 1 ? (c % g) / (g - 1) : 0.5, cy = g > 1 ? Math.floor(c / g) / (g - 1) : 0.5;
      pairs.push([(u - cx) ** 2 + (v - cy) ** 2, i, c]);
    }
  });
  pairs.sort((a, b) => a[0] - b[0]);
  const place = new Array(points.length).fill(-1), taken = new Set();
  for (const [, i, c] of pairs) if (place[i] < 0 && !taken.has(c)) { place[i] = c; taken.add(c); }
  return place;
}

function tracePlan(pinned) {
  const n0 = pinned.level, Vtop = S.sight.levels[n0];
  const roots = pinned.keys.slice(0, TREE_ROOTS);  // кусочки идут от сильнейшего (pinnedPieces)
  const sides = {}, known = new Map(), nodes = roots.map(k => treeNode(n0, k, sides, known)), rootSide = Math.ceil(Math.sqrt(nodes.length));
  // Места детей в сетке зависят только от кусочка: считаются один раз на кусочек.
  const placed = new Map();
  const placesOf = node => {
    if (!placed.has(node)) {
      const below = S.sight.levels[node.n - 1];
      placed.set(node, gridPlaces(node.children.map(c => [below.centre[2 * c.k], below.centre[2 * c.k + 1]]), sides[node.n]));
    }
    return placed.get(node);
  };
  const rootPlaces = gridPlaces(nodes.map(node => [Vtop.centre[2 * node.k], Vtop.centre[2 * node.k + 1]]), rootSide);
  return {
    tiles(n) {
      // Плитка — 1; узел уровня m > n — сетка своих детей; зазор: внутри группы — 0.06, между группами — 0.4 на ступень.
      const gapAt = m => m - n === 1 ? 0.06 : 0.4 * (m - n - 1);
      const sizeAt = m => m === n ? 1 : sides[m] * sizeAt(m - 1) + (sides[m] - 1) * gapAt(m);
      const list = [];
      const place = (node, left, top) => {
        if (node.n === n) { list.push({k: node.k, left, top, size: 1}); return; }
        const g = sides[node.n], child = sizeAt(node.n - 1), gap = gapAt(node.n), places = placesOf(node);
        node.children.forEach((c, i) => place(c, left + (places[i] % g) * (child + gap), top + Math.floor(places[i] / g) * (child + gap)));
      };
      const rootSize = sizeAt(n0), rootGap = 0.4 * (n0 - n + 1);
      nodes.forEach((node, i) => place(node, (rootPlaces[i] % rootSide) * (rootSize + rootGap), Math.floor(rootPlaces[i] / rootSide) * (rootSize + rootGap)));
      return {list, span: rootSide * rootSize + (rootSide - 1) * rootGap};
    },
  };
}

// Уровни выше выбранного: кусочки, куда оно вошло, — одной сеткой по их местам на рисунке.
function aboveTiles(n, keys) {
  const V = S.sight.levels[n], g = Math.max(1, Math.ceil(Math.sqrt(keys.length))), gap = 0.06;
  const places = gridPlaces(keys.map(k => [V.centre[2 * k], V.centre[2 * k + 1]]), g);
  return {list: keys.map((k, i) => ({k, left: (places[i] % g) * (1 + gap), top: Math.floor(places[i] / g) * (1 + gap), size: 1})),
    span: g + (g - 1) * gap};
}

// Картинка кусочка k уровня n: штрихи рисунка в его окне, окно вписано в квадрат. Рисуется один раз на рисунок —
// в свёртке один кусочек стоит много раз (в каждом окне уровня выше, куда он вошёл).
const SPRITE = 64;

function pieceStrokes(ctx, n, k, x, y, size, alpha) {
  const V = S.sight.levels[n], sticks = S.sight.sticks;
  const cy = V.centre[2 * k], cx = V.centre[2 * k + 1], extent = V.extent[k], unit = (size / 2) / (extent + 0.6);
  for (let j = 0; j < sticks.count; j++) {
    const dy = sticks.y[j] - cy, dx = sticks.x[j] - cx;
    if (Math.hypot(dy, dx) > extent + 1e-5) continue;
    stick(ctx, x + size / 2 + dx * unit, y + size / 2 + dy * unit, unit, sticks.angle[j], alpha, null, 0);
  }
}

function pieceSprite(n, k) {
  if (!S.sprites || S.sprites.id !== S.sightId) S.sprites = {id: S.sightId, levels: {}};
  const level = S.sprites.levels[n] || (S.sprites.levels[n] = new Map());
  let sprite = level.get(k);
  if (!sprite) {
    sprite = document.createElement("canvas");
    sprite.width = sprite.height = SPRITE;
    const ctx = sprite.getContext("2d");
    ctx.lineCap = "round";
    pieceStrokes(ctx, n, k, 0, 0, SPRITE, 1);
    level.set(k, sprite);
  }
  return sprite;
}

// Плитка — кусочек рисунка: штрихи рисунка в окне кусочка, окно вписано в плитку; яркость — сила кусочка.
function drawStep(entry) {
  const {n, canvas, tiles, span} = entry, W = Math.round(canvas.clientWidth * DPR), H = Math.round(canvas.clientHeight * DPR);
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  const key = JSON.stringify([S.pin, W, H, tiles.length]);
  S.stepViews = S.stepViews || {};
  if (!S.stepViews[n] || S.stepViews[n].key !== key) {
    const s = Math.min(W, H) * 0.96 / Math.max(1e-6, span);
    S.stepViews[n] = {key, s, ox: (W - span * s) / 2, oy: (H - span * s) / 2};
  }
  const view = S.stepViews[n], V = S.sight.levels[n];
  let strongest = 0;
  for (const t of tiles) strongest = Math.max(strongest, V.strength[t.k]);
  ctx.lineCap = "round";
  for (const t of tiles) {
    const x = view.ox + t.left * view.s, y = view.oy + t.top * view.s, size = t.size * view.s;
    if (x > W || y > H || x + size < 0 || y + size < 0) continue;
    ctx.globalAlpha = 1; ctx.fillStyle = "#15171c"; ctx.fillRect(x, y, size, size);
    const weight = strongest > 0 ? Math.pow(V.strength[t.k] / strongest, 1.5) : 0, alpha = 0.15 + 0.85 * weight;
    if (size < 3 * DPR) continue;
    // Мелкая плитка — готовой картинкой кусочка; крупная (приблизили) — штрихами, чтобы было чётко.
    if (size > 1.6 * SPRITE) pieceStrokes(ctx, n, t.k, x, y, size, alpha);
    else { ctx.globalAlpha = alpha; ctx.drawImage(pieceSprite(n, t.k), x, y, size, size); }
  }
  ctx.globalAlpha = 1;
}

function stepAt(entry, event) {
  const box = entry.canvas.getBoundingClientRect(), k = entry.canvas.width / box.width, view = S.stepViews[entry.n];
  const x = ((event.clientX - box.left) * k - view.ox) / view.s, y = ((event.clientY - box.top) * k - view.oy) / view.s;
  for (const t of entry.tiles) if (x >= t.left && y >= t.top && x <= t.left + t.size && y <= t.top + t.size) return t.k;
  return -1;
}

// Колесо — приблизить, тянуть — сдвинуть, двойной щелчок — всё; наведение подсвечивает кусочек на рисунке уровня.
function stepEvents(entry) {
  const canvas = entry.canvas;
  let pan = null;
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const view = S.stepViews[entry.n], r = canvas.getBoundingClientRect(), k = canvas.width / r.width;
    const mx = (event.clientX - r.left) * k, my = (event.clientY - r.top) * k, f = Math.pow(1.0015, -event.deltaY);
    view.ox = mx - (mx - view.ox) * f; view.oy = my - (my - view.oy) * f; view.s *= f;
    drawStep(entry);
  }, {passive: false});
  canvas.addEventListener("pointerdown", event => { pan = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointerup", () => { pan = null; });
  canvas.addEventListener("dblclick", () => { S.stepViews[entry.n] = null; drawStep(entry); });
  canvas.addEventListener("pointermove", event => {
    if (pan) {
      const view = S.stepViews[entry.n], k = canvas.width / canvas.getBoundingClientRect().width;
      view.ox += (event.clientX - pan[0]) * k; view.oy += (event.clientY - pan[1]) * k; pan = [event.clientX, event.clientY];
      drawStep(entry);
      return;
    }
    const k = stepAt(entry, event);
    hover(k >= 0 ? {kind: "piece", level: entry.n, index: k} : null);
  });
  canvas.addEventListener("pointerleave", () => hover(null));
}

function renderStatus() {
  const m = S.model, levels = m.levels.length;
  const views = m.train > m.originals ? ` и их виды — ${m.train} обучающих картинок` : " без видов";
  $("status").textContent = `цикл: ${m.originals} цифр MNIST${views} · уровней ${levels} · ` +
    `карты ${m.levels.map(L => L.points).join(" / ")} точек · детекторов ${m.levels.map(L => L.count).join(" / ")} · читатель λ = ${m.decay}` +
    (S.sight ? ` · путь рисунка ${Math.round(S.sight.ms)} мс` : "") + rateNote();
  const maxLayers = Math.max(...m.levels.map(L => L.thresholds.length));
  $("legend").innerHTML = Array.from({length: maxLayers}, (_, k) => `<span><i style="border-color:${LAYER[k % LAYER.length]}"></i>слой ${k}</span>`).join("") +
    `<span>□ зажжённая точка</span><span style="color:${PIN}">● выбранное (кусочек, кластер, точка)</span><span style="color:${MEMBER}">● его путь</span>` +
    `<span>цветные штрихи на уровне ${m.primary && m.primary.inCycle ? "1+" : "2+"} — детали уровня ниже, цвет — цифра</span>`;
}

// Частота обновления за последнюю секунду и рабочие процессы сервера.
function rateNote() {
  const now = performance.now();
  S.updates = S.updates.filter(t => now - t < 1000);
  const workers = S.workers && S.workers.workers ? ` · процессов ${S.workers.ready} из ${S.workers.workers}` : "";
  return (S.updates.length > 1 ? ` · ${S.updates.length} обновлений/с` : "") + workers;
}

function render() {
  if (!S.model) return;
  renderStatus();
  renderAnswer();
  if (S.primaryCard) { drawPrimaryPicture(S.primaryCard); drawPrimaryMap(S.primaryCard); renderPrimaryStats(S.primaryCard); }
  for (const item of S.cards) { drawPieces(item); drawMap(item); renderStats(item); }
  renderTrace();
  if (S.flashAt && performance.now() - S.flashAt < 900) {
    cancelAnimationFrame(S.glow);
    S.glow = requestAnimationFrame(glowFrame);
  }
}

// Вспышка новых кластеров гаснет за 0.9 с: карты перерисовываются по кадрам, пока она видна.
function glowFrame() {
  for (const item of S.cards) drawMap(item);
  if (performance.now() - S.flashAt < 900) S.glow = requestAnimationFrame(glowFrame);
}

// ================================================================== старт
async function start() {
  try {
    const raw = await (await fetch("/api/model")).json();
    S.model = prepareModel(raw);
  } catch (error) {
    $("status").textContent = `нет обученного цикла: ${error.message}`;
    return;
  }
  buildCards();
  drawPad();
  render();
  watchWorkers();
  const asked = new URLSearchParams(location.search);
  S.wishPin = asked.get("pin");
  if (asked.has("example")) loadExample();
}
start();
