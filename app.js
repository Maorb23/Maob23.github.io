const codeLines = [
  "@triton.jit",
  "def matvec_kernel(a_ptr, x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):",
  "    row = tl.program_id(axis=0)",
  "    cols = tl.arange(0, BLOCK_SIZE)",
  "    mask = cols < N",
  "    a_offsets = row * N + cols",
  "    x_offsets = cols",
  "    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)",
  "    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)",
  "    products = a * x",
  "    acc = tl.sum(products, axis=0)",
  "    tl.store(output_ptr + row, acc, mask=row < M)",
];

const steps = [
  {
    line: 2,
    title: "Launch program",
    note: (s) =>
      `The grid is (${s.M},), so Triton launches one program for each output row. This program sees row = ${s.row}.`,
    phase: "program",
  },
  {
    line: 3,
    title: "Build lanes",
    note: (s) =>
      `tl.arange creates ${s.BLOCK_SIZE} lanes inside the program. They are column candidates, not Python loop iterations.`,
    phase: "lanes",
  },
  {
    line: 4,
    title: "Create mask",
    note: (s) =>
      `${s.N} real columns exist, so lanes with cols >= ${s.N} are masked off and skipped.`,
    phase: "mask",
  },
  {
    line: 5,
    title: "A pointer offsets",
    note: (s) =>
      `For row-major A, row ${s.row} starts at row * N = ${s.row * s.N}. Add cols to walk across that row.`,
    phase: "aOffsets",
  },
  {
    line: 6,
    title: "x pointer offsets",
    note: () =>
      "The vector x is shared by every row program, so its offsets are just cols.",
    phase: "xOffsets",
  },
  {
    line: 7,
    title: "Load A row",
    note: () =>
      "tl.load gathers A values for the active lanes. Masked lanes receive other=0.0 in this visualization.",
    phase: "loadA",
  },
  {
    line: 8,
    title: "Load x",
    note: () =>
      "The same lane columns load matching x values, so each lane now owns one A[row, col] and one x[col].",
    phase: "loadX",
  },
  {
    line: 9,
    title: "Multiply lanes",
    note: () =>
      "Each active lane multiplies one matrix value by one vector value. This is the SIMD-shaped part.",
    phase: "multiply",
  },
  {
    line: 10,
    title: "Reduce",
    note: () =>
      "tl.sum reduces the block of lane products into the single scalar output for this row.",
    phase: "reduce",
  },
  {
    line: 11,
    title: "Store output",
    note: (s) =>
      `The scalar result is stored at output_ptr + row, so only output[${s.row}] changes for this program.`,
    phase: "store",
  },
];

const initial = {
  M: 6,
  N: 10,
  BLOCK_SIZE: 16,
  row: 0,
  step: 0,
  running: false,
};

let state = { ...initial };
let timer = null;

const els = {
  mInput: document.querySelector("#mInput"),
  nInput: document.querySelector("#nInput"),
  mValue: document.querySelector("#mValue"),
  nValue: document.querySelector("#nValue"),
  rowSelect: document.querySelector("#rowSelect"),
  gridLabel: document.querySelector("#gridLabel"),
  programLabel: document.querySelector("#programLabel"),
  stepLabel: document.querySelector("#stepLabel"),
  codeProgramLabel: document.querySelector("#codeProgramLabel"),
  codeBlock: document.querySelector("#codeBlock"),
  stepNote: document.querySelector("#stepNote"),
  rowPrograms: document.querySelector("#rowPrograms"),
  laneGrid: document.querySelector("#laneGrid"),
  laneCaption: document.querySelector("#laneCaption"),
  matrixGrid: document.querySelector("#matrixGrid"),
  xGrid: document.querySelector("#xGrid"),
  outGrid: document.querySelector("#outGrid"),
  multiplyGrid: document.querySelector("#multiplyGrid"),
  sumLine: document.querySelector("#sumLine"),
  computeTitle: document.querySelector("#computeTitle"),
  rowFormula: document.querySelector("#rowFormula"),
  aOffsetLabel: document.querySelector("#aOffsetLabel"),
  resetBtn: document.querySelector("#resetBtn"),
  prevBtn: document.querySelector("#prevBtn"),
  nextBtn: document.querySelector("#nextBtn"),
  runBtn: document.querySelector("#runBtn"),
  blockButtons: document.querySelectorAll("[data-block]"),
};

function valueA(row, col) {
  const raw = ((row + 2) * (col + 3) + row * 7 - col * 2) % 17;
  return Number(((raw - 8) / 4).toFixed(2));
}

function valueX(col) {
  const raw = ((col + 5) * 7) % 13;
  return Number(((raw - 6) / 3).toFixed(2));
}

function rowProducts(row = state.row) {
  return Array.from({ length: state.N }, (_, col) => valueA(row, col) * valueX(col));
}

function rowSum(row = state.row) {
  return rowProducts(row).reduce((acc, value) => acc + value, 0);
}

function isPhaseAtLeast(name) {
  const currentIndex = steps.findIndex((step) => step.phase === steps[state.step].phase);
  const targetIndex = steps.findIndex((step) => step.phase === name);
  return currentIndex >= targetIndex;
}

function format(value) {
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  return normalized.toFixed(2);
}

function syncInputs() {
  els.mInput.value = state.M;
  els.nInput.value = state.N;
  els.mValue.textContent = state.M;
  els.nValue.textContent = state.N;
  els.gridLabel.textContent = `(${state.M},) programs`;
  els.programLabel.textContent = `row ${state.row}`;
  els.stepLabel.textContent = `${state.step + 1} / ${steps.length}`;
  els.codeProgramLabel.textContent = `program ${state.row} / ${state.M - 1}`;
  els.rowFormula.textContent = `row = tl.program_id(0) = ${state.row}`;
  els.aOffsetLabel.textContent = `a_ptr + ${state.row} * ${state.N} + cols`;
  els.blockButtons.forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.block) === state.BLOCK_SIZE);
  });
}

function renderRowSelect() {
  els.rowSelect.innerHTML = "";
  for (let row = 0; row < state.M; row += 1) {
    const option = document.createElement("option");
    option.value = String(row);
    option.textContent = `row ${row}`;
    els.rowSelect.append(option);
  }
  state.row = Math.min(state.row, state.M - 1);
  els.rowSelect.value = String(state.row);
}

function renderCode() {
  const activeLine = steps[state.step].line;
  els.codeBlock.innerHTML = "";
  codeLines.forEach((line, index) => {
    const span = document.createElement("span");
    span.className = "code-line";
    if (index === activeLine) span.classList.add("active");
    if (index < activeLine) span.classList.add("done");
    span.textContent = line;
    els.codeBlock.append(span);
  });
  const current = steps[state.step];
  els.stepNote.textContent = `${current.title}: ${current.note(state)}`;
}

function renderPrograms() {
  els.rowPrograms.innerHTML = "";
  for (let row = 0; row < state.M; row += 1) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "program-row";
    if (row === state.row) item.classList.add("active");
    item.style.setProperty("--progress", `${((row + 1) / state.M) * 100}%`);
    item.innerHTML = `<span>program ${row}</span><span class="row-bar"></span><span>output[${row}]</span>`;
    item.addEventListener("click", () => {
      state.row = row;
      state.step = 0;
      stopRun();
      render();
    });
    els.rowPrograms.append(item);
  }
}

function renderLanes() {
  els.laneCaption.textContent = `${state.BLOCK_SIZE} lanes touch columns 0..${state.BLOCK_SIZE - 1}`;
  els.laneGrid.style.gridTemplateColumns = `repeat(${Math.min(state.BLOCK_SIZE, 16)}, minmax(44px, 1fr))`;
  els.laneGrid.innerHTML = "";
  for (let col = 0; col < state.BLOCK_SIZE; col += 1) {
    const lane = document.createElement("div");
    const inMask = col < state.N;
    lane.className = "lane";
    if (isPhaseAtLeast("mask")) lane.classList.add(inMask ? "in-mask" : "out-mask");
    lane.innerHTML = `<small>lane</small>${col}`;
    lane.title = inMask ? `cols[${col}] = ${col}, mask=True` : `cols[${col}] = ${col}, mask=False`;
    els.laneGrid.append(lane);
  }
}

function renderMatrix() {
  els.matrixGrid.innerHTML = "";
  els.matrixGrid.style.gridTemplateColumns = "1fr";
  for (let row = 0; row < state.M; row += 1) {
    const rowEl = document.createElement("div");
    rowEl.className = "matrix-row";
    rowEl.style.gridTemplateColumns = `repeat(${state.N}, minmax(48px, 1fr))`;
    for (let col = 0; col < state.N; col += 1) {
      const cell = document.createElement("div");
      const active = row === state.row && col < state.BLOCK_SIZE && isPhaseAtLeast("aOffsets");
      cell.className = "cell";
      if (active) cell.classList.add("active");
      cell.innerHTML = `<small>${row * state.N + col}</small>${format(valueA(row, col))}`;
      cell.title = `A[${row}, ${col}], flat offset ${row * state.N + col}`;
      rowEl.append(cell);
    }
    els.matrixGrid.append(rowEl);
  }
}

function renderVector() {
  els.xGrid.innerHTML = "";
  els.outGrid.innerHTML = "";
  for (let col = 0; col < state.N; col += 1) {
    const cell = document.createElement("div");
    const active = col < state.BLOCK_SIZE && isPhaseAtLeast("xOffsets");
    cell.className = "vector-cell";
    if (active) cell.classList.add("active");
    cell.innerHTML = `<small>${col}</small>${format(valueX(col))}`;
    els.xGrid.append(cell);
  }
  for (let row = 0; row < state.M; row += 1) {
    const cell = document.createElement("div");
    const stored = row < state.row || (row === state.row && isPhaseAtLeast("store"));
    cell.className = "vector-cell";
    if (stored) cell.classList.add("active");
    cell.innerHTML = `<small>${row}</small>${stored ? format(rowSum(row)) : "0.00"}`;
    cell.title = stored ? `stored output[${row}]` : `output[${row}] not written in this highlighted program`;
    els.outGrid.append(cell);
  }
}

function renderMultiply() {
  els.computeTitle.textContent = `Computation for row ${state.row}`;
  els.multiplyGrid.innerHTML = "";
  els.multiplyGrid.style.gridTemplateColumns = `repeat(${Math.min(state.BLOCK_SIZE, 8)}, minmax(84px, 1fr))`;
  const products = [];
  for (let col = 0; col < state.BLOCK_SIZE; col += 1) {
    const inMask = col < state.N;
    const cell = document.createElement("div");
    cell.className = "multiply-cell";
    if (isPhaseAtLeast("multiply")) cell.classList.add(inMask ? "active" : "masked");
    const a = inMask ? valueA(state.row, col) : 0;
    const x = inMask ? valueX(col) : 0;
    const product = a * x;
    if (inMask) products.push(product);
    const body = inMask
      ? `A[${state.row},${col}]<br><strong>${format(a)}</strong> x <strong>${format(x)}</strong><br>= ${format(product)}`
      : `col ${col}<br><strong>masked</strong><br>= 0.00`;
    cell.innerHTML = body;
    els.multiplyGrid.append(cell);
  }
  if (isPhaseAtLeast("reduce")) {
    const shown = products.map(format).join(" + ");
    els.sumLine.textContent = `tl.sum(products) = ${shown} = ${format(rowSum())}`;
  } else {
    els.sumLine.textContent = "tl.sum waits until the lane products exist.";
  }
}

function render() {
  syncInputs();
  renderRowSelect();
  renderCode();
  renderPrograms();
  renderLanes();
  renderMatrix();
  renderVector();
  renderMultiply();
}

function nextStep() {
  if (state.step < steps.length - 1) {
    state.step += 1;
  } else if (state.row < state.M - 1) {
    state.row += 1;
    state.step = 0;
  } else {
    stopRun();
  }
  render();
}

function previousStep() {
  if (state.step > 0) {
    state.step -= 1;
  } else if (state.row > 0) {
    state.row -= 1;
    state.step = steps.length - 1;
  }
  stopRun();
  render();
}

function stopRun() {
  state.running = false;
  clearInterval(timer);
  timer = null;
  els.runBtn.textContent = "Run";
}

function toggleRun() {
  if (state.running) {
    stopRun();
    return;
  }
  state.running = true;
  els.runBtn.textContent = "Pause";
  timer = setInterval(nextStep, 900);
}

els.mInput.addEventListener("input", (event) => {
  state.M = Number(event.target.value);
  state.row = Math.min(state.row, state.M - 1);
  state.step = 0;
  stopRun();
  render();
});

els.nInput.addEventListener("input", (event) => {
  state.N = Number(event.target.value);
  state.step = 0;
  stopRun();
  render();
});

els.rowSelect.addEventListener("change", (event) => {
  state.row = Number(event.target.value);
  state.step = 0;
  stopRun();
  render();
});

els.blockButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.BLOCK_SIZE = Number(button.dataset.block);
    state.step = 0;
    stopRun();
    render();
  });
});

els.resetBtn.addEventListener("click", () => {
  state = { ...initial };
  stopRun();
  render();
});

els.prevBtn.addEventListener("click", previousStep);
els.nextBtn.addEventListener("click", () => {
  stopRun();
  nextStep();
});
els.runBtn.addEventListener("click", toggleRun);

document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    event.preventDefault();
    stopRun();
    nextStep();
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    previousStep();
  }
  if (event.key === " ") {
    event.preventDefault();
    toggleRun();
  }
});

render();
