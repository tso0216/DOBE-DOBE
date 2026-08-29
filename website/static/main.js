"use strict";

let INIT = null;

const state = {
  idx: -1,          // test-local patch 索引
  gid: -1,          // 全域 patch 編號（顯示用）
  nPoi: 0,
  base: null,       // {dx, dy, cat}
  added: [],        // [{dx:[], dy:[], cat}]，一格 = 一個 undo 批次
  traj: [],         // [{z:[x,y], robust, score}]，traj[0] = 未加料基準
  cat: 0,
  busy: false,
};

const $ = (id) => document.getElementById(id);
const geoCv = $("geo"), latCv = $("latent"), poiCv = $("poi");

// ---- canvas 基礎 ----

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

// 建立資料座標 → 像素的轉換；equal=true 時兩軸等比例（置中擴張較短的一軸）
// view={cx, cy, zoom} 時以該中心與倍率覆蓋預設視野
function makeTransform(x0, x1, y0, y1, w, h, pad, equal, view) {
  let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  let sx = (x1 - x0) * (1 + pad * 2) || 1, sy = (y1 - y0) * (1 + pad * 2) || 1;
  if (equal) {
    const unit = Math.max(sx / w, sy / h);
    sx = unit * w;
    sy = unit * h;
  }
  if (view) {
    cx = view.cx;
    cy = view.cy;
    sx /= view.zoom;
    sy /= view.zoom;
  }
  const X = (x) => (x - (cx - sx / 2)) / sx * w;
  const Y = (y) => h - (y - (cy - sy / 2)) / sy * h;
  return {
    X, Y,
    invX: (px) => px / w * sx + (cx - sx / 2),
    invY: (py) => (h - py) / h * sy + (cy - sy / 2),
    spanX: sx, spanY: sy,
  };
}

function drawGrid(ctx, w, h, n) {
  ctx.strokeStyle = "rgba(107,114,128,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < n; i++) {
    const x = (w * i) / n, y = (h * i) / n;
    ctx.moveTo(x, 0); ctx.lineTo(x, h);
    ctx.moveTo(0, y); ctx.lineTo(w, y);
  }
  ctx.stroke();
}

function drawDot(ctx, x, y, r, fill, stroke, lw) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
}

function drawSquare(ctx, x, y, half, color, lw) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(x - half, y - half, half * 2, half * 2);
}

function drawX(ctx, x, y, r, color, lw) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
  ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
  ctx.stroke();
}

function drawStar(ctx, x, y, R, fill, stroke) {
  const r = R * 0.45;
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? R : r;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const px = x + rad * Math.cos(a), py = y + rad * Math.sin(a);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
}

// ---- 三張圖 ----

let geoT = null, latT = null, poiT = null;
let geoView = null;   // {cx, cy, zoom}；null = 預設全景
let poiByCat = [];    // 每個類別的 POI 索引，載入後建一次
const GEO_ZOOM_MIN = 0.4, GEO_ZOOM_MAX = 100;
const POI_SHOW_CELL_PX = 12;   // cell 至少要畫到這個像素寬才顯示 POI；縮小過頭只顯示網格

function geoBounds() {
  const g = INIT.geo;
  return [Math.min(...g.x), Math.max(...g.x), Math.min(...g.y), Math.max(...g.y)];
}

function drawGeo() {
  const [ctx, w, h] = fitCanvas(geoCv);
  const [x0, x1, y0, y1] = geoBounds();
  geoT = makeTransform(x0, x1, y0, y1, w, h, 0.05, true, geoView);

  const g = INIT.geo;
  const step = INIT.center_step, half = step / 2;

  // 背景格線＝資料網格的 cell 邊界；太密（<8px 一格）時改畫 10 倍間距的對齊格線
  let gs = step;
  while ((gs / geoT.spanX) * w < 8) gs *= 10;
  const originX = g.x[0] - half, originY = g.y[0] - half;
  const xs = geoT.invX(0), xe = geoT.invX(w);
  const yb = geoT.invY(h), yt = geoT.invY(0);
  ctx.strokeStyle = "rgba(107,114,128,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = originX + Math.ceil((xs - originX) / gs) * gs; x <= xe; x += gs) {
    const px = geoT.X(x);
    ctx.moveTo(px, 0); ctx.lineTo(px, h);
  }
  for (let y = originY + Math.ceil((yb - originY) / gs) * gs; y <= yt; y += gs) {
    const py = geoT.Y(y);
    ctx.moveTo(0, py); ctx.lineTo(w, py);
  }
  ctx.stroke();

  // 每個 patch 畫成實際大小的 cell（step × step，隨縮放同步縮放）
  // 放大顯示 POI 時轉半透明讓點透出來；縮小只剩網格時用實色
  const cw = Math.max((step / geoT.spanX) * w, 2);
  const showPoi = cw >= POI_SHOW_CELL_PX;
  const rect = (cx, cy) => ctx.fillRect(geoT.X(cx) - cw / 2, geoT.Y(cy) - cw / 2, cw, cw);
  ctx.fillStyle = showPoi ? "rgba(192,57,43,0.22)" : "rgba(192,57,43,0.85)";  // train：紅、鎖定
  for (let i = 0; i < g.x.length; i++) if (!g.is_test[i]) rect(g.x[i], g.y[i]);
  ctx.fillStyle = showPoi ? "rgba(37,99,235,0.25)" : "rgba(37,99,235,0.9)";   // test：藍、可選
  for (let i = 0; i < g.x.length; i++) if (g.is_test[i]) rect(g.x[i], g.y[i]);

  // POI：實際地理位置，依類別上色（只畫視野內的）；縮小到 cell 不夠大就不畫，只留網格
  const P = INIT.poi;
  const pr = Math.min(Math.max(cw * 0.05, 1), 3.5);
  for (let c = 0; showPoi && c < poiByCat.length; c++) {
    ctx.fillStyle = INIT.colors[c];
    ctx.beginPath();
    for (const i of poiByCat[c]) {
      const x = P.x[i], y = P.y[i];
      if (x < xs - step || x > xe + step || y < yb - step || y > yt + step) continue;
      const px = geoT.X(x), py = geoT.Y(y);
      if (pr < 1.5) ctx.rect(px - pr, py - pr, pr * 2, pr * 2);
      else { ctx.moveTo(px + pr, py); ctx.arc(px, py, pr, 0, Math.PI * 2); }
    }
    ctx.fill();
  }

  if (state.gid >= 0) {
    const px = geoT.X(g.x[state.gid]), py = geoT.Y(g.y[state.gid]);
    drawSquare(ctx, px, py, Math.max(cw / 2 + 1.5, 6), "#1d4ed8", 2);
    ctx.fillStyle = "#374151";
    ctx.font = "10px sans-serif";
    ctx.fillText(`Grid #${state.gid}`, px + Math.max(cw / 2 + 4, 9), py - Math.max(cw / 2 + 2, 7));
  }
}

function drawLatent() {
  const [ctx, w, h] = fitCanvas(latCv);
  const lo = INIT.latent_lo, hi = INIT.latent_hi;
  latT = makeTransform(lo[0], hi[0], lo[1], hi[1], w, h, 0.05, false);
  drawGrid(ctx, w, h, 8);

  ctx.globalAlpha = 0.75;
  for (const z of INIT.latent) drawDot(ctx, latT.X(z[0]), latT.Y(z[1]), 2.5, "#6b7280");
  ctx.globalAlpha = 1;

  if (!state.traj.length) return;
  ctx.strokeStyle = "#c0392b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  state.traj.forEach((t, i) => {
    const px = latT.X(t.z[0]), py = latT.Y(t.z[1]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();

  const o = state.traj[0], c = state.traj[state.traj.length - 1];
  drawStar(ctx, latT.X(o.z[0]), latT.Y(o.z[1]), 9, "#f59e0b", "#7c2d12");
  drawDot(ctx, latT.X(c.z[0]), latT.Y(c.z[1]), 5.5, "#c0392b", "#111827", 1.2);
}

function drawPoi() {
  const [ctx, w, h] = fitCanvas(poiCv);
  const R = INIT.half_width, lim = R * 1.1;
  poiT = makeTransform(-lim, lim, -lim, lim, w, h, 0, true);
  drawGrid(ctx, w, h, 8);

  ctx.setLineDash([5, 4]);                        // 100m cell（±HALF_WIDTH）邊界
  ctx.strokeStyle = "#6b7280";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(poiT.X(-R), poiT.Y(R),
                 (2 * R / poiT.spanX) * w, (2 * R / poiT.spanY) * h);
  ctx.setLineDash([]);

  if (!state.base) return;
  const b = state.base;
  ctx.globalAlpha = 0.85;
  for (let i = 0; i < b.dx.length; i++) {
    drawDot(ctx, poiT.X(b.dx[i]), poiT.Y(b.dy[i]), 3.2, INIT.colors[b.cat[i]]);
  }
  ctx.globalAlpha = 1;
  for (const batch of state.added) {
    for (let i = 0; i < batch.dx.length; i++) {
      drawX(ctx, poiT.X(batch.dx[i]), poiT.Y(batch.dy[i]), 4, INIT.colors[batch.cat], 1.6);
    }
  }
}

function drawAll() { drawGeo(); drawLatent(); drawPoi(); }

// ---- 頂部數字 ----

function updateHeader() {
  if (!state.traj.length) return;
  const cur = state.traj[state.traj.length - 1];
  const base = state.traj[0];
  const nAdded = state.added.reduce((s, b) => s + b.dx.length, 0);

  $("score-value").textContent = cur.score.toFixed(3);
  const chip = $("score-chip");
  const tol = Math.max(Math.abs(base.score), 1) * 1e-9;
  const delta = cur.score - base.score;
  if (!state.added.length || Math.abs(delta) <= tol) {
    chip.textContent = "Δ +0.000 · Unchanged";
    chip.className = "chip";
  } else if (delta > 0) {
    chip.textContent = `Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(3)} · 遠離鄰居`;
    chip.className = "chip far";
  } else {
    chip.textContent = `Δ ${delta.toFixed(3)} · 靠近鄰居`;
    chip.className = "chip near";
  }

  $("patch-value").textContent = `#${state.gid}`;
  $("base-value").textContent = state.nPoi;
  $("added-value").textContent = nAdded;
  $("robust-value").textContent = `${cur.robust.toFixed(2)} / ${INIT.thr.toFixed(2)}`;
  const rc = $("robust-chip");
  const out = cur.robust > INIT.thr;
  rc.textContent = out ? "OUTLIER" : "NORMAL";
  rc.className = "chip " + (out ? "bad" : "ok");
  const poiSub = $("poi-sub");   // 副標題可有可無，拿掉也不影響其他更新
  if (poiSub) {
    poiSub.textContent =
      `Grid #${state.gid}｜base ${state.nPoi} 顆｜點一下加一顆目前類別的 POI`;
  }
}

// ---- 與後端互動 ----

async function selectPatch(i) {
  if (state.busy) return;
  state.busy = true;
  try {
    const r = await fetch(`/api/patch/${i}`);
    const d = await r.json();
    state.idx = d.idx;
    state.gid = d.global_id;
    state.nPoi = d.n_poi;
    state.base = d.base;
    state.added = [];
    state.traj = [{ z: d.z, robust: d.robust, score: d.score }];
    drawAll();
    updateHeader();
  } finally {
    state.busy = false;
  }
}

function addedCounts() {
  const c = new Array(INIT.categories.length).fill(0);
  for (const b of state.added) c[b.cat] += b.dx.length;
  return c;
}

async function addBatch(dx, dy) {
  if (state.busy || state.idx < 0) return;
  state.busy = true;
  state.added.push({ dx, dy, cat: state.cat });
  try {
    const r = await fetch("/api/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idx: state.idx, added: addedCounts() }),
    });
    if (!r.ok) throw new Error("encode failed");
    const d = await r.json();
    state.traj.push({ z: d.z, robust: d.robust, score: d.score });
    drawLatent();
    drawPoi();
    updateHeader();
  } catch (e) {
    state.added.pop();
    drawPoi();
  } finally {
    state.busy = false;
  }
}

// ---- 事件 ----

// ---- 地理圖縮放/平移 ----

function currentGeoView() {
  if (!geoView) {
    geoView = {
      cx: geoT.invX(geoCv.clientWidth / 2),
      cy: geoT.invY(geoCv.clientHeight / 2),
      zoom: 1,
    };
  }
  return geoView;
}

// 以 (mx, my) 像素為錨點把倍率乘上 f；未給錨點則以畫面中心
function geoZoomBy(f, mx, my) {
  if (!geoT) return;
  const w = geoCv.clientWidth, h = geoCv.clientHeight;
  if (mx === undefined) { mx = w / 2; my = h / 2; }
  const v = currentGeoView();
  const zoomNew = Math.min(GEO_ZOOM_MAX, Math.max(GEO_ZOOM_MIN, v.zoom * f));
  const dataX = geoT.invX(mx), dataY = geoT.invY(my);
  const sxNew = geoT.spanX * v.zoom / zoomNew;
  const syNew = geoT.spanY * v.zoom / zoomNew;
  v.cx = dataX - (mx / w - 0.5) * sxNew;
  v.cy = dataY - ((h - my) / h - 0.5) * syNew;
  v.zoom = zoomNew;
  drawGeo();
}

geoCv.addEventListener("wheel", (ev) => {
  if (!INIT || !geoT) return;
  ev.preventDefault();
  const rect = geoCv.getBoundingClientRect();
  geoZoomBy(Math.exp(-ev.deltaY * 0.0015),
            ev.clientX - rect.left, ev.clientY - rect.top);
}, { passive: false });

let geoDrag = null, geoDragMoved = false;

geoCv.addEventListener("mousedown", (ev) => {
  if (!INIT || !geoT) return;
  const v = currentGeoView();
  geoDrag = { mx: ev.clientX, my: ev.clientY, cx: v.cx, cy: v.cy };
  geoDragMoved = false;
});

window.addEventListener("mousemove", (ev) => {
  if (!geoDrag) return;
  const dx = ev.clientX - geoDrag.mx, dy = ev.clientY - geoDrag.my;
  if (!geoDragMoved && Math.abs(dx) + Math.abs(dy) <= 3) return;
  geoDragMoved = true;
  geoView.cx = geoDrag.cx - (dx / geoCv.clientWidth) * geoT.spanX;
  geoView.cy = geoDrag.cy + (dy / geoCv.clientHeight) * geoT.spanY;
  drawGeo();
});

window.addEventListener("mouseup", () => { geoDrag = null; });

geoCv.addEventListener("dblclick", () => {
  geoView = null;
  drawGeo();
});

geoCv.addEventListener("click", (ev) => {
  if (!INIT || !geoT) return;
  if (geoDragMoved) { geoDragMoved = false; return; }   // 拖曳平移結束，不當成選取
  const rect = geoCv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const g = INIT.geo;
  let best = -1, bestD = 1e18;
  for (let i = 0; i < g.x.length; i++) {
    if (!g.is_test[i]) continue;                  // train 鎖定：不參與挑選
    const d = (geoT.X(g.x[i]) - mx) ** 2 + (geoT.Y(g.y[i]) - my) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best < 0) return;
  // 點進 cell 範圍內就算選中；縮很小時放寬成 12px 內
  const half = INIT.center_step / 2;
  const inCell = Math.abs(geoT.invX(mx) - g.x[best]) <= half &&
                 Math.abs(geoT.invY(my) - g.y[best]) <= half;
  if (inCell || bestD <= 12 * 12) selectPatch(g.test_pos[best]);
});

latCv.addEventListener("click", (ev) => {
  if (!INIT || !latT) return;
  const rect = latCv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  let best = -1, bestD = 1e18;
  for (let i = 0; i < INIT.latent.length; i++) {
    const z = INIT.latent[i];
    const d = (latT.X(z[0]) - mx) ** 2 + (latT.Y(z[1]) - my) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best >= 0) selectPatch(best);
});

poiCv.addEventListener("click", (ev) => {
  if (!INIT || !poiT || state.idx < 0) return;
  const rect = poiCv.getBoundingClientRect();
  const dx = poiT.invX(ev.clientX - rect.left);
  const dy = poiT.invY(ev.clientY - rect.top);
  if (Math.abs(dx) > INIT.half_width || Math.abs(dy) > INIT.half_width) return;  // 只能加在 cell 內
  addBatch([dx], [dy]);
});

$("geo-zoom-in").addEventListener("click", () => geoZoomBy(1.5));
$("geo-zoom-out").addEventListener("click", () => geoZoomBy(1 / 1.5));
$("geo-zoom-reset").addEventListener("click", () => { geoView = null; drawGeo(); });

$("btn-random").addEventListener("click", () => {
  if (!INIT || state.idx < 0) return;
  const dx = [], dy = [];
  for (let k = 0; k < INIT.batch_n; k++) {        // 100m cell 內均勻
    dx.push((Math.random() * 2 - 1) * INIT.half_width);
    dy.push((Math.random() * 2 - 1) * INIT.half_width);
  }
  addBatch(dx, dy);
});

$("btn-undo").addEventListener("click", () => {
  if (state.busy || !state.added.length) return;
  state.added.pop();
  state.traj.pop();
  drawLatent();
  drawPoi();
  updateHeader();
});

$("btn-reset").addEventListener("click", () => {
  if (state.busy || !state.added.length) return;
  state.added = [];
  state.traj = state.traj.slice(0, 1);
  drawLatent();
  drawPoi();
  updateHeader();
});

window.addEventListener("resize", () => { if (INIT) drawAll(); });

// ---- 啟動 ----

function buildCatButtons() {
  const box = $("cat-list");
  INIT.categories.forEach((name, i) => {
    const btn = document.createElement("button");
    btn.className = "cat-btn" + (i === 0 ? " active" : "");
    btn.innerHTML = `<i class="dot" style="background:${INIT.colors[i]}"></i>${name}`;
    btn.addEventListener("click", () => {
      state.cat = i;
      box.querySelectorAll(".cat-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    });
    box.appendChild(btn);
  });
}

(async function init() {
  const r = await fetch("/api/init");
  INIT = await r.json();
  poiByCat = INIT.categories.map(() => []);
  INIT.poi.cat.forEach((c, i) => poiByCat[c].push(i));
  buildCatButtons();
  $("btn-random").textContent = `+${INIT.batch_n} Random`;
  drawAll();
  await selectPatch(INIT.init_patch);
})();









