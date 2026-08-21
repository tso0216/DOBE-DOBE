"use strict";

const COLORS = {
    ink: "#122c2a",
    muted: "#60716e",
    grid: "#ddd9cf",
    coral: "#e05a47",
    teal: "#1b726a",
    paper: "#f7f5ef",
};

const state = {
    bootstrap: null,
    model: "ddae_fsce",
    patchId: 0,
    patch: null,
    embedding: null,
    simulation: null,
    latentScreenPoints: [],
    loadToken: 0,
};

const el = {};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
    [
        "modelStatus", "modelSelect", "modelDescription", "regionSelect",
        "patchInput", "patchRange", "goPatchButton", "categorySelect",
        "amountRange", "amountOutput", "simulateButton", "scopeNote",
        "errorBanner", "totalPoiMetric", "densityMetric", "shiftMetric",
        "saturationMetric", "percentileMetric", "latentCanvas", "latentLoading",
        "curveCanvas", "curveCategoryBadge", "interpretation", "poiMapCanvas",
        "patchTitle", "coordinateLabel", "topCategoryBadge", "poiStats",
        "datasetFooter",
    ].forEach((id) => { el[id] = document.getElementById(id); });

    bindEvents();
    try {
        state.bootstrap = await fetchJSON("/api/bootstrap");
        populateControls();
        state.patchId = state.bootstrap.presets[0].patch_id;
        el.patchInput.value = state.patchId;
        el.regionSelect.value = String(state.patchId);
        await refreshAll();
    } catch (error) {
        showError(error.message);
        el.modelStatus.textContent = "載入失敗";
    }
}

function bindEvents() {
    el.modelSelect.addEventListener("change", async () => {
        state.model = el.modelSelect.value;
        updateModelDescription();
        await refreshEmbeddingAndSimulation();
    });
    el.regionSelect.addEventListener("change", async () => {
        if (el.regionSelect.value === "custom") return;
        await selectPatch(Number(el.regionSelect.value));
    });
    el.goPatchButton.addEventListener("click", () => {
        selectPatch(Number(el.patchInput.value));
    });
    el.patchInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") selectPatch(Number(el.patchInput.value));
    });
    el.categorySelect.addEventListener("change", runSimulation);
    el.amountRange.addEventListener("input", () => {
        el.amountOutput.textContent = `+${el.amountRange.value}`;
    });
    el.amountRange.addEventListener("change", runSimulation);
    el.simulateButton.addEventListener("click", runSimulation);
    el.latentCanvas.addEventListener("click", onLatentClick);

    const redraw = () => {
        drawLatent();
        drawCurve();
        drawPoiMap();
    };
    window.addEventListener("resize", debounce(redraw, 120));
}

function populateControls() {
    const availableModels = state.bootstrap.models.filter((model) => model.available);
    el.modelSelect.replaceChildren(...availableModels.map((model) => {
        const option = document.createElement("option");
        option.value = model.key;
        option.textContent = model.label;
        return option;
    }));
    if (!availableModels.some((model) => model.key === state.model)) {
        state.model = availableModels[0].key;
    }
    el.modelSelect.value = state.model;
    updateModelDescription();

    el.categorySelect.replaceChildren(...state.bootstrap.categories.map((category) => {
        const option = document.createElement("option");
        option.value = category.id;
        option.textContent = category.name;
        return option;
    }));

    el.regionSelect.replaceChildren();
    const groups = new Map();
    state.bootstrap.presets.forEach((preset) => {
        if (!groups.has(preset.group)) {
            const group = document.createElement("optgroup");
            group.label = preset.group;
            groups.set(preset.group, group);
            el.regionSelect.appendChild(group);
        }
        const option = document.createElement("option");
        option.value = preset.patch_id;
        option.textContent = `${preset.label} · #${preset.patch_id}`;
        groups.get(preset.group).appendChild(option);
    });
    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "自訂／從圖上點選";
    el.regionSelect.appendChild(custom);

    const dataset = state.bootstrap.dataset;
    el.patchInput.max = dataset.patch_count - 1;
    el.patchRange.textContent = `0–${dataset.patch_count - 1}`;
    el.scopeNote.textContent = dataset.scope_note;
    el.datasetFooter.textContent = `${dataset.patch_count.toLocaleString()} patches · ${dataset.poi_records.toLocaleString()} POI records`;
}

function updateModelDescription() {
    const model = state.bootstrap?.models.find((item) => item.key === state.model);
    if (!model) return;
    el.modelDescription.textContent = model.description;
    el.modelStatus.textContent = `${model.short_label} 已選擇`;
}

async function refreshAll() {
    clearError();
    const token = ++state.loadToken;
    setLoading(true);
    try {
        const [patch, embedding] = await Promise.all([
            fetchJSON(`/api/patch/${state.patchId}`),
            fetchJSON(`/api/models/${state.model}/embedding`),
        ]);
        if (token !== state.loadToken) return;
        state.patch = patch;
        state.embedding = embedding;
        renderPatch();
        await runSimulation(token);
    } catch (error) {
        if (token === state.loadToken) showError(error.message);
    } finally {
        if (token === state.loadToken) setLoading(false);
    }
}

async function refreshEmbeddingAndSimulation() {
    clearError();
    const token = ++state.loadToken;
    setLoading(true);
    try {
        state.embedding = await fetchJSON(`/api/models/${state.model}/embedding`);
        if (token !== state.loadToken) return;
        drawLatent();
        await runSimulation(token);
    } catch (error) {
        if (token === state.loadToken) showError(error.message);
    } finally {
        if (token === state.loadToken) setLoading(false);
    }
}

async function selectPatch(patchId) {
    const max = state.bootstrap.dataset.patch_count - 1;
    if (!Number.isInteger(patchId) || patchId < 0 || patchId > max) {
        showError(`Patch 編號必須介於 0 和 ${max}。`);
        return;
    }
    clearError();
    state.patchId = patchId;
    el.patchInput.value = patchId;
    const preset = state.bootstrap.presets.find((item) => item.patch_id === patchId);
    el.regionSelect.value = preset ? String(patchId) : "custom";
    const token = ++state.loadToken;
    setLoading(true, false);
    try {
        state.patch = await fetchJSON(`/api/patch/${patchId}`);
        if (token !== state.loadToken) return;
        renderPatch();
        await runSimulation(token);
    } catch (error) {
        if (token === state.loadToken) showError(error.message);
    } finally {
        if (token === state.loadToken) setLoading(false);
    }
}

async function runSimulation(parentToken = null) {
    if (!state.patch || !state.embedding) return;
    const token = parentToken ?? ++state.loadToken;
    setSimulationBusy(true);
    clearError();
    try {
        const result = await fetchJSON("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: state.model,
                patch_id: state.patchId,
                category: Number(el.categorySelect.value),
                amount: Number(el.amountRange.value),
            }),
        });
        if (token !== state.loadToken) return;
        state.simulation = result;
        renderSimulation();
    } catch (error) {
        if (token === state.loadToken) showError(error.message);
    } finally {
        if (token === state.loadToken) setSimulationBusy(false);
    }
}

function renderPatch() {
    const patch = state.patch;
    el.patchTitle.textContent = `Patch #${patch.id}`;
    el.coordinateLabel.textContent = `${patch.latitude.toFixed(5)}, ${patch.longitude.toFixed(5)}`;
    el.totalPoiMetric.textContent = patch.total.toLocaleString();
    el.densityMetric.textContent = `${patch.density_label} · 密度第 ${patch.density_percentile.toFixed(1)} 百分位`;
    el.topCategoryBadge.textContent = `最多：${patch.top_category}`;
    drawPoiMap();
    drawPoiStats(patch.counts, patch.counts, -1);
    drawLatent();
}

function renderSimulation() {
    const result = state.simulation;
    const metrics = result.metrics;
    const total = result.updated_counts.reduce((sum, value) => sum + value, 0);
    el.totalPoiMetric.textContent = total.toLocaleString();
    el.densityMetric.textContent = `${state.patch.density_label} · 原始 ${state.patch.total} + 本次 ${result.amount}`;
    el.shiftMetric.textContent = metrics.standardized_shift.toFixed(2);
    el.saturationMetric.textContent = `${metrics.saturation_index.toFixed(1)}%`;
    el.percentileMetric.textContent = `${metrics.latent_percentile.toFixed(1)}%`;
    el.curveCategoryBadge.textContent = `${result.category_name} +${result.amount}`;
    const color = state.bootstrap.categories[result.category].color;
    el.curveCategoryBadge.style.color = color;
    el.curveCategoryBadge.style.backgroundColor = `${color}1c`;
    el.interpretation.textContent = result.interpretation;
    drawPoiStats(result.base_counts, result.updated_counts, result.category);
    drawLatent();
    drawCurve();
}

function drawPoiStats(baseCounts, updatedCounts, selectedCategory) {
    if (!state.bootstrap) return;
    const max = Math.max(...updatedCounts, 1);
    el.poiStats.replaceChildren(...state.bootstrap.categories.map((category, index) => {
        const row = document.createElement("div");
        row.className = "stat-row";
        const name = document.createElement("span");
        name.className = "stat-name";
        name.textContent = category.name;
        const track = document.createElement("span");
        track.className = "stat-track";
        const base = document.createElement("i");
        base.className = "stat-base";
        base.style.width = `${100 * baseCounts[index] / max}%`;
        base.style.backgroundColor = category.color;
        const added = document.createElement("i");
        added.className = "stat-added";
        added.style.left = `${100 * baseCounts[index] / max}%`;
        added.style.width = `${100 * (updatedCounts[index] - baseCounts[index]) / max}%`;
        added.style.backgroundColor = COLORS.coral;
        track.append(base, added);
        const value = document.createElement("span");
        value.className = "stat-value";
        const increase = updatedCounts[index] - baseCounts[index];
        value.textContent = increase > 0 ? `${baseCounts[index]} +${increase}` : String(baseCounts[index]);
        if (index === selectedCategory) name.style.color = COLORS.coral;
        row.append(name, track, value);
        return row;
    }));
}

function drawLatent() {
    const canvas = el.latentCanvas;
    if (!canvas || !state.embedding) return;
    const { ctx, width, height } = prepareCanvas(canvas);
    const points = state.embedding.points;
    const path = state.simulation?.path ?? [];
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    let xMin = quantile(xs, .01), xMax = quantile(xs, .99);
    let yMin = quantile(ys, .01), yMax = quantile(ys, .99);
    for (const point of path) {
        xMin = Math.min(xMin, point.x); xMax = Math.max(xMax, point.x);
        yMin = Math.min(yMin, point.y); yMax = Math.max(yMax, point.y);
    }
    const xPad = Math.max((xMax - xMin) * .09, .01);
    const yPad = Math.max((yMax - yMin) * .09, .01);
    xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
    const margin = { left: 40, right: 18, top: 18, bottom: 34 };
    const sx = (x) => margin.left + (x - xMin) / (xMax - xMin) * (width - margin.left - margin.right);
    const sy = (y) => height - margin.bottom - (y - yMin) / (yMax - yMin) * (height - margin.top - margin.bottom);

    drawGrid(ctx, width, height, margin, "z₁", "z₂");
    state.latentScreenPoints = points.map((point, index) => ({ x: sx(point[0]), y: sy(point[1]), index }));
    ctx.fillStyle = "rgba(102, 122, 118, .32)";
    for (const point of state.latentScreenPoints) {
        if (point.x < margin.left || point.x > width - margin.right || point.y < margin.top || point.y > height - margin.bottom) continue;
        ctx.beginPath(); ctx.arc(point.x, point.y, 1.65, 0, Math.PI * 2); ctx.fill();
    }

    if (path.length) {
        ctx.strokeStyle = COLORS.coral;
        ctx.lineWidth = 2.2;
        ctx.lineJoin = "round";
        ctx.beginPath();
        path.forEach((point, index) => {
            const x = sx(point.x), y = sy(point.y);
            if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        path.forEach((point, index) => {
            if (index !== 0 && index !== path.length - 1 && index % Math.max(1, Math.floor(path.length / 8)) !== 0) return;
            ctx.fillStyle = index === 0 ? COLORS.ink : COLORS.coral;
            ctx.beginPath(); ctx.arc(sx(point.x), sy(point.y), index === path.length - 1 ? 6 : 3.2, 0, Math.PI * 2); ctx.fill();
        });
    } else if (points[state.patchId]) {
        ctx.fillStyle = COLORS.ink;
        ctx.beginPath(); ctx.arc(sx(points[state.patchId][0]), sy(points[state.patchId][1]), 5, 0, Math.PI * 2); ctx.fill();
    }
}

function drawCurve() {
    const canvas = el.curveCanvas;
    if (!canvas) return;
    const { ctx, width, height } = prepareCanvas(canvas);
    const path = state.simulation?.path ?? [];
    const margin = { left: 46, right: 16, top: 18, bottom: 38 };
    drawGrid(ctx, width, height, margin, "加入數量", "位移");
    if (!path.length) return;
    const xMax = Math.max(path[path.length - 1].amount, 1);
    const yMax = Math.max(...path.map((point) => point.shift), .01) * 1.12;
    const sx = (x) => margin.left + x / xMax * (width - margin.left - margin.right);
    const sy = (y) => height - margin.bottom - y / yMax * (height - margin.top - margin.bottom);

    const gradient = ctx.createLinearGradient(0, margin.top, 0, height - margin.bottom);
    gradient.addColorStop(0, "rgba(224,90,71,.28)");
    gradient.addColorStop(1, "rgba(224,90,71,.015)");
    ctx.beginPath();
    path.forEach((point, index) => index ? ctx.lineTo(sx(point.amount), sy(point.shift)) : ctx.moveTo(sx(point.amount), sy(point.shift)));
    ctx.lineTo(sx(path[path.length - 1].amount), height - margin.bottom);
    ctx.lineTo(sx(0), height - margin.bottom);
    ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

    ctx.beginPath();
    path.forEach((point, index) => index ? ctx.lineTo(sx(point.amount), sy(point.shift)) : ctx.moveTo(sx(point.amount), sy(point.shift)));
    ctx.strokeStyle = COLORS.coral; ctx.lineWidth = 2.5; ctx.stroke();
    const final = path[path.length - 1];
    ctx.fillStyle = COLORS.coral; ctx.beginPath(); ctx.arc(sx(final.amount), sy(final.shift), 5, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = COLORS.muted; ctx.font = "10px sans-serif";
    ctx.textAlign = "left"; ctx.fillText("0", margin.left, height - 18);
    ctx.textAlign = "right"; ctx.fillText(String(xMax), width - margin.right, height - 18);
    ctx.fillText(yMax.toFixed(2), margin.left - 7, margin.top + 4);
}

function drawPoiMap() {
    const canvas = el.poiMapCanvas;
    if (!canvas || !state.patch) return;
    const { ctx, width, height } = prepareCanvas(canvas);
    const margin = { left: 38, right: 18, top: 16, bottom: 34 };
    drawGrid(ctx, width, height, margin, "東西向 (m)", "南北向");
    const size = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    const x0 = margin.left + (width - margin.left - margin.right - size) / 2;
    const y0 = margin.top + (height - margin.top - margin.bottom - size) / 2;
    const sx = (x) => x0 + (x + 50) / 100 * size;
    const sy = (y) => y0 + size - (y + 50) / 100 * size;
    ctx.strokeStyle = "#c8c4b9"; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, size, size);
    state.patch.points.forEach((point) => {
        const category = state.bootstrap.categories[point.category];
        ctx.fillStyle = category.color;
        ctx.globalAlpha = .82;
        ctx.beginPath(); ctx.arc(sx(point.x), sy(point.y), 4, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
}

function drawGrid(ctx, width, height, margin, xLabel, yLabel) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.paper; ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const x = margin.left + i / 4 * (width - margin.left - margin.right);
        const y = margin.top + i / 4 * (height - margin.top - margin.bottom);
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, height - margin.bottom); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    }
    ctx.fillStyle = COLORS.muted; ctx.font = "10px sans-serif";
    ctx.textAlign = "right"; ctx.fillText(xLabel, width - margin.right, height - 12);
    ctx.save(); ctx.translate(13, margin.top + 8); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right"; ctx.fillText(yLabel, 0, 0); ctx.restore();
}

function onLatentClick(event) {
    if (!state.latentScreenPoints.length) return;
    const rect = el.latentCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let nearest = null;
    let best = 14 * 14;
    for (const point of state.latentScreenPoints) {
        const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
        if (distance < best) { best = distance; nearest = point.index; }
    }
    if (nearest !== null) selectPatch(nearest);
}

function prepareCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `要求失敗（${response.status}）`);
    return payload;
}

function quantile(values, q) {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const base = Math.floor(position);
    const rest = position - base;
    return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

function setLoading(loading, includeCanvas = true) {
    if (includeCanvas) el.latentLoading.hidden = !loading;
    el.modelStatus.textContent = loading ? "模型計算中⋯" : `${state.embedding?.label ?? "模型"} 已就緒`;
}

function setSimulationBusy(busy) {
    el.simulateButton.disabled = busy;
    el.simulateButton.firstElementChild.textContent = busy ? "計算中⋯" : "執行模擬";
}

function showError(message) {
    el.errorBanner.textContent = message;
    el.errorBanner.hidden = false;
}

function clearError() {
    el.errorBanner.hidden = true;
    el.errorBanner.textContent = "";
}

function debounce(fn, wait) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), wait);
    };
}
