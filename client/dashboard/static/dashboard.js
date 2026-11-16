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
const hue = a => `hsl(${(((a % 360) + 360) % 360).toFixed(0)},95%,60%)`;

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
  previous: [], flash: [], views: [], cards: [], busy: false, dirty: false, final: false, frame: 0,
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
  return {train: raw.train, originals: raw.originals, decay: raw.decay, brush: raw.brush, cell: raw.grid.cell, sticks, levels,
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
  return {sticks, levels, digit: raw.digit, probabilities: raw.probabilities, ms: raw.ms};
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

// ================================================================== рисование содержимого
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

// Одновременно идёт один запрос; пока он идёт, рисунок меняется — следом уходит свежий.
function schedule(final) {
  S.final = S.final || final;
  if (S.busy) { S.dirty = true; return; }
  send();
}

async function send() {
  if (!DRAW.strokes.length && !S.example) {
    S.final = false; S.sight = null; S.image = null; render(); return;
  }
  S.busy = true; S.dirty = false;
  const final = S.final, image = padPicture();
  S.final = false;
  try {
    const body = JSON.stringify({image: btoa(String.fromCharCode.apply(null, image)), final});
    const answer = await (await fetch("/api/see", {method: "POST", body})).json();
    if (!answer.ok) throw new Error(answer.error);
    if (DRAW.strokes.length || S.example) {
      S.image = image;
      remember();
      S.sight = prepareSight(answer);
      if (S.wishPin) {
        // ?pin=уровень:кусочек — выбрать кусочек сразу (для снимков страницы).
        const [level, keyhole] = S.wishPin.split(":").map(Number);
        if (level >= 1 && level <= S.sight.levels.length) pin(level - 1, Math.min(keyhole || 0, S.sight.levels[level - 1].K - 1));
        S.wishPin = null;
      } else if (S.pin) {
        // Выбранный кусочек держится за место на рисунке: после штриха — ближайший кусочек того же уровня.
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
  S.busy = false;
  if (S.dirty) send();
}

// Кластеры, которых не было до этого штриха, вспыхивают: видно, как кусочки набиваются в кластеры.
function remember() {
  const now = performance.now();
  if (!S.sight) { S.previous = []; return; }
  S.previous = S.sight.levels.map(V => V.detCount);
  S.flashAt = now;
}

// ================================================================== уровни: карточки
function buildCards() {
  const box = $("levels");
  box.innerHTML = "";
  box.style.gridTemplateColumns = `repeat(${S.model.levels.length}, minmax(0, 1fr))`;
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
  for (const item of S.cards) {
    const wrap = item.map.parentElement, side = Math.max(120, Math.min(wrap.clientWidth, wrap.clientHeight));
    item.map.style.width = item.map.style.height = side + "px";
    item.map.width = item.map.height = Math.round(side * DPR);
    const pieceSide = item.pieces.clientWidth;
    item.pieces.width = item.pieces.height = Math.round(pieceSide * DPR);
    item.view = fit(item);
    item.keep = {};
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
  if (item.n === 0) {
    const st = sight.sticks;
    for (let i = 0; i < st.count; i++) stick(ctx, px(st.x[i]), px(st.y[i]), unit * S.model.cell, st.angle[i], 0.85, null, st.span[i]);
  }
  const V = sight.levels[item.n];
  if (!V) { ctx.fillStyle = "#9aa3b2"; ctx.font = `${Math.round(W / 16)}px Segoe UI`; ctx.textAlign = "center"; ctx.fillText("до уровня не дошло", W / 2, W / 2); return; }
  const marks = highlights();
  let top = 0;
  for (let k = 0; k < V.K; k++) top = Math.max(top, V.strength[k]);
  // Поля кусочков — еле видно (их десятки, и они перекрываются); сам кусочек — точка в центре окна, цвет — сила
  // его детали (П.1): синий — слабая, жёлтый — сильная, серый — ни один детектор не сработал.
  ctx.strokeStyle = "rgba(110,168,255,0.07)"; ctx.lineWidth = Math.max(1, W / 360);
  for (let k = 0; k < V.K; k++) {
    ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), r(V.extent[k]), 0, 2 * Math.PI); ctx.stroke();
  }
  const dot = Math.max(2, W / 110);
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
      ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), Math.max(2.5, W / (member ? 110 : 80)), 0, 2 * Math.PI); ctx.fill();
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
  // Углопозиции выбранного кусочка уровня 1.
  if (item.n === 0 && levelMarks) {
    for (const [k, kind] of levelMarks) {
      if (kind === "member") continue;
      for (let j = V.entries.start[k]; j < V.entries.start[k + 1]; j++) {
        const e = V.entryElement[V.entries.order[j]], st = sight.sticks;
        stick(ctx, px(st.x[e]), px(st.y[e]), unit * S.model.cell * 1.2, st.angle[e], 1, "#ffffff", st.span[e] / 1.2);
      }
    }
  }
}

// Подсветка кусочков по уровням: выбранный (pin), под курсором (hover), его состав ниже и куда он вошёл выше.
function highlights(withHover = true) {
  const marks = new Map(), sight = S.sight;
  if (!sight) return marks;
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
  if (S.pin) chain(S.pin.level, S.pin.keyhole, "pin");
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
  ctx.drawImage(item.keep.canvas, 0, 0);
  const V = S.sight?.levels[item.n];
  if (!V) return;
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
  const fresh = S.previous[item.n], glow = S.flashAt ? Math.max(0, 1 - (performance.now() - S.flashAt) / 900) : 0;
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
  // Выбранный кусочек и кусочек под курсором: их точки и кластеры.
  const own = [];
  if (S.pin && S.pin.level === item.n) own.push([S.pin.keyhole, PIN]);
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
  canvas.addEventListener("pointerdown", event => { drag = [event.clientX, event.clientY]; canvas.style.cursor = "grabbing"; hideTip(); });
  window.addEventListener("pointerup", () => { if (drag) { drag = null; canvas.style.cursor = "grab"; } });
  window.addEventListener("pointermove", event => {
    if (!drag) return;
    const k = canvas.width / canvas.getBoundingClientRect().width;
    item.view.ox += (event.clientX - drag[0]) * k; item.view.oy += (event.clientY - drag[1]) * k; drag = [event.clientX, event.clientY];
    drawMap(item);
  });
  canvas.addEventListener("pointermove", event => { if (!drag) hover(mapAt(item, event), event); });
  canvas.addEventListener("pointerleave", () => hover(null));
  canvas.addEventListener("dblclick", () => { item.view = fit(item); drawMap(item); });
  item.pieces.addEventListener("pointermove", event => { const k = pieceAt(item, event); hover(k >= 0 ? {kind: "piece", level: item.n, index: k} : null, event); });
  item.pieces.addEventListener("pointerleave", () => hover(null));
  item.pieces.addEventListener("click", event => {
    const k = pieceAt(item, event);
    if (k >= 0 && !(S.pin && S.pin.level === item.n && S.pin.keyhole === k)) pin(item.n, k); else S.pin = null;
    render();
  });
}

// Выбрать кусочек k уровня n; его место на рисунке запоминается, чтобы выбор пережил следующий штрих.
function pin(n, k) {
  const V = S.sight.levels[n];
  S.pin = {level: n, keyhole: k, y: V.centre[2 * k], x: V.centre[2 * k + 1]};
}

function hover(target, event) {
  const same = (a, b) => (a === b) || (a && b && a.kind === b.kind && a.level === b.level && a.index === b.index);
  if (!same(target, S.hover)) {
    S.hover = target;
    cancelAnimationFrame(S.frame);
    S.frame = requestAnimationFrame(() => { for (const item of S.cards) { drawPieces(item); drawMap(item); } });
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
    `<div class="row"><span>зажжено точек карты</span><b>${V.litPoints} из ${M.points}</b></div>` +
    `<div class="row"><span>пар «кусочек → точка»</span><b>${V.litPairs}</b></div>` +
    `<div class="row"><span>срабатываний детекторов</span><b>${V.firedTotal}</b></div>` +
    `<div class="row"><span>кластеров набито</span><b>${V.clusters}${added && fresh ? ` <span style="color:${PIN}">+${added}</span>` : ""}</b></div>` +
    `<div class="row"><span>кусочков без отклика</span><b>${silent}</b></div>` +
    `<div class="row"><span>сила деталей, средняя</span><b>${mean.toFixed(2)}</b></div>` +
    `<div class="clusters">${top.slice(0, 5).map(d => {
      const c = composition(M, d)[0];
      return `<div class="cl"><i style="border-color:${LAYER[M.layer[d] % LAYER.length]}"></i>слой ${M.layer[d]} · кусочков <b>${V.detCount[d]}</b> · «${c.digit}» ${(100 * c.share).toFixed(0)} %</div>`;
    }).join("")}</div>`;
}

function renderAnswer() {
  const bars = $("bars"), sight = S.sight;
  const p = sight ? sight.probabilities : new Array(10).fill(0), best = sight ? sight.digit : -1;
  $("digit").textContent = best >= 0 ? String(best) : "—";
  bars.innerHTML = p.map((value, digit) => `<span class="${digit === best ? "top" : ""}">${digit}</span><div class="bar"><span style="width:${(100 * value).toFixed(1)}%"></span></div><span>${(100 * value).toFixed(1)} %</span>`).join("");
  $("answerNote").textContent = sight ? (best >= 0 ? `ответ за ${Math.round(sight.ms)} мс; ${sight.sticks.count} углопозиций` : "ни один детектор не сработал — ответа нет") : "";
}

function renderTrace() {
  const body = $("traceBody"), sight = S.sight;
  body.innerHTML = "";
  if (!sight || !S.pin) {
    body.innerHTML = `<div class="note">Щёлкните по кусочку (кругу) на рисунке любого уровня: здесь появится, из чего он собран ниже, ` +
      `в какие кусочки вошёл выше и какие кластеры зажёг на каждом уровне. Наведение на кластер карты подсвечивает кусочки, которые в него набились.</div>`;
    return;
  }
  const marks = highlights(false);
  for (let n = 0; n < sight.levels.length; n++) {
    const V = sight.levels[n], M = S.model.levels[n], chosen = [...(marks.get(n) || new Map()).keys()];
    const step = document.createElement("div");
    step.className = "step";
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 116 * DPR;
    const ctx = canvas.getContext("2d");
    const item = {pieces: canvas, n};
    drawTraceDigit(ctx, item, V, chosen);
    step.appendChild(canvas);
    const text = document.createElement("div");
    text.className = "text";
    const clusters = new Map();
    for (const k of chosen) for (let j = V.fired.start[k]; j < V.fired.start[k + 1]; j++) {
      const i = V.fired.order[j], d = V.firedDetector[i];
      clusters.set(d, Math.max(clusters.get(d) || 0, V.firedLevel[i]));
    }
    const strongest = [...clusters.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
    text.innerHTML = `<b>Уровень ${M.number}</b>${n === S.pin.level ? " — выбранный кусочек" : n < S.pin.level ? " — из чего собран" : " — куда вошёл"}<br>` +
      `кусочков: <b>${chosen.length}</b>, кластеров зажгли: <b>${clusters.size}</b><br>` +
      strongest.map(([d, level]) => { const c = composition(M, d)[0]; return `слой ${M.layer[d]}: E = ${level.toFixed(2)}, «${c.digit}» ${(100 * c.share).toFixed(0)} %`; }).join("<br>");
    const glyphs = document.createElement("div");
    glyphs.className = "glyphs";
    for (const [d] of strongest) {
      const inside = members(M, d).slice().sort((a, b) => V.pointAct[b] - V.pointAct[a]);
      if (inside.length) glyphs.appendChild(glyphCanvas(M, inside[0], 40, true));
    }
    text.appendChild(glyphs);
    step.appendChild(text);
    body.appendChild(step);
  }
}

function drawTraceDigit(ctx, item, V, chosen) {
  const W = item.pieces.width, unit = W / DRAW.side, cell = S.model.cell, px = x => (x + 0.5) * cell * unit;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, W);
  const tile = document.createElement("canvas"); tile.width = tile.height = DRAW.side;
  const t = tile.getContext("2d"), data = t.createImageData(DRAW.side, DRAW.side);
  for (let i = 0; i < S.image.length; i++) { const v = S.image[i] * 0.45; data.data.set([v, v, v, 255], 4 * i); }
  t.putImageData(data, 0, 0);
  ctx.drawImage(tile, 0, 0, W, W);
  for (const k of chosen) {
    const own = item.n === S.pin.level && k === S.pin.keyhole;
    ctx.strokeStyle = own ? PIN : MEMBER; ctx.fillStyle = ctx.strokeStyle;
    ctx.globalAlpha = own ? 1 : 0.3; ctx.lineWidth = own ? Math.max(2, W / 50) : Math.max(1, W / 160);
    ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), V.extent[k] * cell * unit, 0, 2 * Math.PI); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(px(V.centre[2 * k + 1]), px(V.centre[2 * k]), Math.max(2, W / 60), 0, 2 * Math.PI); ctx.fill();
  }
}

function renderStatus() {
  const m = S.model, levels = m.levels.length;
  $("status").textContent = `цикл: ${m.originals} цифр MNIST и их виды — ${m.train} обучающих картинок · уровней ${levels} · ` +
    `карты ${m.levels.map(L => L.points).join(" / ")} точек · детекторов ${m.levels.map(L => L.count).join(" / ")} · читатель λ = ${m.decay}` +
    (S.sight ? ` · ответ сервера ${Math.round(S.sight.ms)} мс` : "");
  const maxLayers = Math.max(...m.levels.map(L => L.thresholds.length));
  $("legend").innerHTML = Array.from({length: maxLayers}, (_, k) => `<span><i style="border-color:${LAYER[k % LAYER.length]}"></i>слой ${k}</span>`).join("") +
    `<span>□ зажжённая точка</span><span style="color:${PIN}">● выбранный кусочек</span><span style="color:${MEMBER}">● его цепочка</span>`;
}

function render() {
  if (!S.model) return;
  renderStatus();
  renderAnswer();
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
  const asked = new URLSearchParams(location.search);
  S.wishPin = asked.get("pin");
  if (asked.has("example")) loadExample();
}
start();
