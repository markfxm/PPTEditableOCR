const apiBase = window.PPTTOEDIT_API_BASE || (location.port && location.port !== "80" && location.port !== "443" ? "http://127.0.0.1:8000" : "/api");
const state = {
  jobId: null,
  pollTimer: null,
  slides: [],
  currentSlide: null,
  selectedIndex: -1,
  image: null,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  drag: null,
  saveTimer: null,
  uploadName: "",
};

const $ = (id) => document.getElementById(id);
const canvas = $("slideCanvas");
const ctx = canvas.getContext("2d");

function setStatus(text, progress = null) {
  $("statusText").textContent = text;
  if (progress !== null) $("progressBar").style.width = `${Math.max(0, Math.min(100, progress))}%`;
}

function log(lines) {
  $("logBox").textContent = (lines || []).slice(-120).join("\n");
  $("logBox").scrollTop = $("logBox").scrollHeight;
}

function setControlsEnabled(enabled) {
  $("exportBtn").disabled = !enabled;
  $("addBoxBtn").disabled = !enabled || !state.currentSlide;
  $("deleteBoxBtn").disabled = !enabled || state.selectedIndex < 0;
  $("repadAllBtn").disabled = !enabled || !state.currentSlide;
  $("repadSelectedBtn").disabled = !enabled || state.selectedIndex < 0;
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // keep response status text
    }
    throw new Error(detail);
  }
  return response.json();
}

function renderSlideList(summarySlides = []) {
  const root = $("slideList");
  root.innerHTML = "";
  const source = state.slides.length
    ? state.slides.map((slide) => ({ index: slide.index, label: `第${slide.index}页`, boxCount: slide.boxes.length }))
    : summarySlides;

  source.forEach((slide) => {
    const item = document.createElement("div");
    item.className = `slide-item ${state.currentSlide?.index === slide.index ? "active" : ""}`;
    item.innerHTML = `<span>${slide.label || `第${slide.index}页`}</span><span>${slide.boxCount ?? 0} 框</span>`;
    item.addEventListener("click", () => selectSlide(slide.index));
    root.appendChild(item);
  });
}

function renderPptList() {
  const root = $("pptList");
  root.innerHTML = "";
  if (!state.uploadName) {
    root.textContent = "本次打开或转换生成的 PPT 会显示在这里。";
    return;
  }
  const item = document.createElement("div");
  item.className = "ppt-item";
  item.title = state.uploadName;
  item.textContent = state.uploadName;
  root.appendChild(item);
}

async function loadSlides() {
  const payload = await request(`/jobs/${state.jobId}/slides`);
  state.slides = payload.slides.map((slide) => ({
    ...slide,
    imageUrl: `${apiBase}${slide.imageUrl.replace(/^\/api/, "")}`,
    boxes: slide.boxes.map(normalizeBox),
  }));
  renderSlideList();
  if (!state.currentSlide && state.slides.length) await selectSlide(state.slides[0].index);
}

function normalizeBox(box) {
  return {
    text: box.text || "",
    score: Number(box.score ?? 1),
    bbox: box.bbox.map(Number),
    erase_rect: (box.erase_rect || box.bbox).map(Number),
    enabled: box.enabled !== false,
    manual: Boolean(box.manual),
    edited: Boolean(box.edited),
    rotation: Number(box.rotation || 0),
  };
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await request(`/jobs/${state.jobId}`);
    setStatus(`${job.phase || job.status} (${job.progress || 0}%)`, job.progress || 0);
    log(job.messages || []);
    renderSlideList(job.slides || []);

    if ((job.status === "ready" || job.status === "done") && state.slides.length === 0) {
      await loadSlides();
    }
    if (job.status === "ready") {
      setControlsEnabled(true);
    }
    if (job.status === "done") {
      $("downloadBtn").href = `${apiBase}/jobs/${state.jobId}/download`;
      $("downloadBtn").classList.remove("disabled");
      setControlsEnabled(true);
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (job.status === "failed") {
      setStatus(`失败：${job.error || "未知错误"}`, job.progress || 0);
      setControlsEnabled(state.slides.length > 0);
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (["running", "queued", "queued_export", "exporting"].includes(job.status)) {
      setControlsEnabled(false);
    }
  } catch (error) {
    setStatus(`连接失败：${error.message}`);
  }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  resetUi();
  state.uploadName = file.name;
  renderPptList();
  setStatus("正在上传文件", 0);
  const response = await fetch(`${apiBase}/jobs`, { method: "POST", body: form });
  if (!response.ok) throw new Error((await response.json()).detail || "上传失败");
  const data = await response.json();
  state.jobId = data.id;
  state.pollTimer = setInterval(pollJob, 1500);
  await pollJob();
}

function resetUi() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  Object.assign(state, {
    jobId: null,
    pollTimer: null,
    slides: [],
    currentSlide: null,
    selectedIndex: -1,
    image: null,
    drag: null,
    saveTimer: null,
    uploadName: "",
  });
  $("slideList").innerHTML = "";
  $("logBox").textContent = "";
  $("downloadBtn").href = "#";
  $("downloadBtn").classList.add("disabled");
  $("emptyHint").style.display = "grid";
  canvas.width = 0;
  canvas.height = 0;
  canvas.style.display = "none";
  setControlsEnabled(false);
  updateInspector();
  renderPptList();
  draw();
}

async function selectSlide(index) {
  const slide = state.slides.find((item) => item.index === index);
  if (!slide) return;
  await saveCurrentSlide();
  state.currentSlide = slide;
  state.selectedIndex = slide.boxes.length ? 0 : -1;
  state.image = new Image();
  state.image.onload = () => {
    $("emptyHint").style.display = "none";
    fitCanvas();
    draw();
    updateInspector();
    renderSlideList();
  };
  state.image.onerror = () => {
    $("emptyHint").style.display = "grid";
    setStatus("页面预览加载失败，请检查后端服务是否仍在运行");
  };
  state.image.src = slide.imageUrl;
}

function fitCanvas() {
  if (!state.currentSlide) return;
  const wrap = document.querySelector(".canvas-wrap");
  const maxWidth = Math.max(320, wrap.clientWidth - 48);
  state.scale = Math.min(1, maxWidth / state.currentSlide.imageWidth);
  canvas.width = Math.round(state.currentSlide.imageWidth * state.scale);
  canvas.height = Math.round(state.currentSlide.imageHeight * state.scale);
  canvas.style.display = "block";
}

function rectToCanvas(rect) {
  return rect.map((value) => Math.round(value * state.scale));
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const hasPreview = Boolean(state.image && state.currentSlide && state.image.complete && state.image.naturalWidth > 0);
  $("emptyHint").style.display = hasPreview ? "none" : "grid";
  canvas.style.display = hasPreview ? "block" : "none";
  if (!hasPreview) return;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  state.currentSlide.boxes.forEach((box, index) => {
    const [x, y, w, h] = rectToCanvas(box.bbox);
    ctx.save();
    ctx.lineWidth = index === state.selectedIndex ? 3 : 2;
    ctx.strokeStyle = box.enabled ? (index === state.selectedIndex ? "#dc2626" : "#0f766e") : "#94a3b8";
    ctx.fillStyle = index === state.selectedIndex ? "rgba(220, 38, 38, 0.08)" : "rgba(15, 118, 110, 0.06)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    if (index === state.selectedIndex) drawHandles(x, y, w, h);
    ctx.restore();
  });
}

function drawHandles(x, y, w, h) {
  const points = handlePoints(x, y, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 2;
  points.forEach((point) => {
    ctx.beginPath();
    ctx.rect(point.x - 4, point.y - 4, 8, 8);
    ctx.fill();
    ctx.stroke();
  });
}

function handlePoints(x, y, w, h) {
  return [
    { name: "nw", x, y },
    { name: "n", x: x + w / 2, y },
    { name: "ne", x: x + w, y },
    { name: "e", x: x + w, y: y + h / 2 },
    { name: "se", x: x + w, y: y + h },
    { name: "s", x: x + w / 2, y: y + h },
    { name: "sw", x, y: y + h },
    { name: "w", x, y: y + h / 2 },
  ];
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) / state.scale,
    y: (event.clientY - rect.top) / state.scale,
  };
}

function hitTest(point) {
  if (!state.currentSlide) return null;
  for (let i = state.currentSlide.boxes.length - 1; i >= 0; i -= 1) {
    const [x, y, w, h] = state.currentSlide.boxes[i].bbox;
    const handles = handlePoints(x, y, w, h);
    const handle = handles.find((item) => Math.abs(point.x - item.x) <= 8 / state.scale && Math.abs(point.y - item.y) <= 8 / state.scale);
    if (handle) return { index: i, mode: "resize", handle: handle.name };
    if (point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h) return { index: i, mode: "move" };
  }
  return null;
}

function clampBox(box) {
  const slide = state.currentSlide;
  if (!slide) return;
  let [x, y, w, h] = box.bbox;
  w = Math.max(8, Math.min(w, slide.imageWidth));
  h = Math.max(8, Math.min(h, slide.imageHeight));
  x = Math.max(0, Math.min(x, slide.imageWidth - w));
  y = Math.max(0, Math.min(y, slide.imageHeight - h));
  box.bbox = [Math.round(x), Math.round(y), Math.round(w), Math.round(h)];
  box.erase_rect = [box.bbox[0], box.bbox[1], box.bbox[0] + box.bbox[2], box.bbox[1] + box.bbox[3]];
  box.edited = true;
}

function expandedRectFromBbox(box) {
  const slide = state.currentSlide;
  if (!slide) return box.erase_rect;
  const [x, y, w, h] = box.bbox;
  const padX = Number($("padXInput").value || 0);
  const padY = Number($("padYInput").value || 0);
  return [
    Math.max(0, Math.round(x - padX)),
    Math.max(0, Math.round(y - padY)),
    Math.min(slide.imageWidth - 1, Math.round(x + w + padX)),
    Math.min(slide.imageHeight - 1, Math.round(y + h + padY)),
  ];
}

function repadBox(box) {
  box.erase_rect = expandedRectFromBbox(box);
  box.edited = true;
}

canvas.addEventListener("mousedown", (event) => {
  const point = canvasPoint(event);
  const hit = hitTest(point);
  if (!hit) {
    state.selectedIndex = -1;
    updateInspector();
    draw();
    return;
  }
  state.selectedIndex = hit.index;
  const box = state.currentSlide.boxes[hit.index];
  state.drag = { ...hit, start: point, original: [...box.bbox] };
  updateInspector();
  draw();
});

window.addEventListener("mousemove", (event) => {
  if (!state.drag || !state.currentSlide) return;
  const point = canvasPoint(event);
  const dx = point.x - state.drag.start.x;
  const dy = point.y - state.drag.start.y;
  const box = state.currentSlide.boxes[state.drag.index];
  let [x, y, w, h] = state.drag.original;
  if (state.drag.mode === "move") {
    box.bbox = [x + dx, y + dy, w, h];
  } else {
    const handle = state.drag.handle;
    if (handle.includes("w")) {
      x += dx;
      w -= dx;
    }
    if (handle.includes("e")) w += dx;
    if (handle.includes("n")) {
      y += dy;
      h -= dy;
    }
    if (handle.includes("s")) h += dy;
    box.bbox = [x, y, w, h];
  }
  clampBox(box);
  draw();
  updateInspector(false);
  scheduleSave();
});

window.addEventListener("mouseup", () => {
  if (state.drag) {
    state.drag = null;
    saveCurrentSlide();
  }
});

function updateInspector(updateText = true) {
  const box = state.currentSlide?.boxes[state.selectedIndex];
  const hasBox = Boolean(box);
  $("deleteBoxBtn").disabled = !hasBox;
  $("repadSelectedBtn").disabled = !hasBox;
  $("repadAllBtn").disabled = !state.currentSlide;
  $("padXInput").disabled = !state.currentSlide;
  $("padYInput").disabled = !state.currentSlide;
  $("textInput").disabled = !hasBox;
  $("enabledInput").disabled = !hasBox;
  $("watermarkInput").disabled = !state.currentSlide;
  $("rotationInput").disabled = !hasBox;
  if (updateText) $("textInput").value = box?.text || "";
  $("enabledInput").checked = box?.enabled !== false;
  $("watermarkInput").checked = state.currentSlide?.removeWatermark !== false;
  $("rotationInput").value = String(box?.rotation || 0);
  $("selectionLabel").textContent = hasBox ? `第${state.currentSlide.index}页 - 第${state.selectedIndex + 1}个框` : "未选择任何框";
  setControlsEnabled(Boolean(state.currentSlide));
}

function scheduleSave() {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveCurrentSlide, 500);
}

async function saveCurrentSlide() {
  if (!state.jobId || !state.currentSlide) return;
  const slide = state.currentSlide;
  await request(`/jobs/${state.jobId}/slides/${slide.index}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      boxes: slide.boxes,
      remove_watermark: slide.removeWatermark,
      watermark_rect: slide.watermarkRect,
    }),
  });
}

$("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await uploadFile(file);
  } catch (error) {
    setStatus(`上传失败：${error.message}`);
  } finally {
    event.target.value = "";
  }
});

$("textInput").addEventListener("input", () => {
  const box = state.currentSlide?.boxes[state.selectedIndex];
  if (!box) return;
  box.text = $("textInput").value;
  box.edited = true;
  scheduleSave();
});

$("enabledInput").addEventListener("change", () => {
  const box = state.currentSlide?.boxes[state.selectedIndex];
  if (!box) return;
  box.enabled = $("enabledInput").checked;
  box.edited = true;
  draw();
  scheduleSave();
});

$("watermarkInput").addEventListener("change", () => {
  if (!state.currentSlide) return;
  state.currentSlide.removeWatermark = $("watermarkInput").checked;
  scheduleSave();
});

$("repadSelectedBtn").addEventListener("click", () => {
  const box = state.currentSlide?.boxes[state.selectedIndex];
  if (!box) return;
  repadBox(box);
  scheduleSave();
});

$("repadAllBtn").addEventListener("click", () => {
  if (!state.currentSlide) return;
  state.currentSlide.boxes.forEach(repadBox);
  scheduleSave();
});

$("rotationInput").addEventListener("change", () => {
  const box = state.currentSlide?.boxes[state.selectedIndex];
  if (!box) return;
  box.rotation = Number($("rotationInput").value);
  box.edited = true;
  scheduleSave();
});

$("addBoxBtn").addEventListener("click", () => {
  const slide = state.currentSlide;
  if (!slide) return;
  const width = Math.min(280, slide.imageWidth - 40);
  const height = 60;
  const left = Math.max(20, slide.imageWidth - width - 40);
  const top = Math.min(Math.max(20, 40), Math.max(20, slide.imageHeight - height - 20));
  slide.boxes.push({
    text: "",
    score: 1,
    bbox: [left, top, width, height],
    erase_rect: [left, top, left + width, top + height],
    enabled: true,
    manual: true,
    edited: true,
    rotation: 0,
  });
  state.selectedIndex = slide.boxes.length - 1;
  updateInspector();
  draw();
  renderSlideList();
  scheduleSave();
});

$("deleteBoxBtn").addEventListener("click", () => {
  if (!state.currentSlide || state.selectedIndex < 0) return;
  state.currentSlide.boxes.splice(state.selectedIndex, 1);
  state.selectedIndex = Math.min(state.selectedIndex, state.currentSlide.boxes.length - 1);
  updateInspector();
  draw();
  renderSlideList();
  scheduleSave();
});

$("exportBtn").addEventListener("click", async () => {
  if (!state.jobId) return;
  clearTimeout(state.saveTimer);
  state.saveTimer = null;
  await saveCurrentSlide();
  $("downloadBtn").classList.add("disabled");
  await request(`/jobs/${state.jobId}/export`, { method: "POST" });
  setControlsEnabled(false);
  if (!state.pollTimer) state.pollTimer = setInterval(pollJob, 1500);
  await pollJob();
});

$("manualBtn").addEventListener("click", () => {
  $("manualOverlay").hidden = false;
  $("manualPanel").classList.add("open");
  $("manualPanel").setAttribute("aria-hidden", "false");
});

function closeManual() {
  $("manualPanel").classList.remove("open");
  $("manualPanel").setAttribute("aria-hidden", "true");
  $("manualOverlay").hidden = true;
}

$("manualOverlay").addEventListener("click", closeManual);

window.addEventListener("resize", () => {
  if (!state.currentSlide) return;
  fitCanvas();
  draw();
});

resetUi();
