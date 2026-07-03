const kernels = {
  matvec: {
    title: "Matvec kernel visualizer",
    file: "matvec_kernel.py",
    programTitle: "One program per row",
    mLabel: "M rows",
    nLabel: "N columns",
    showTemperature: false,
    codeLines: [
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
    ],
    steps: [
      {
        line: 2,
        title: "Launch program",
        phase: "program",
        note: (s) =>
          `The grid is (${s.M},), so Triton launches one program for each output row. This program sees row = ${s.row}.`,
      },
      {
        line: 3,
        title: "Build lanes",
        phase: "lanes",
        note: (s) =>
          `tl.arange creates ${s.BLOCK_SIZE} lanes inside the program. They are column candidates, not Python loop iterations.`,
      },
      {
        line: 4,
        title: "Create mask",
        phase: "mask",
        note: (s) =>
          `${s.N} real columns exist, so lanes with cols >= ${s.N} are masked off and skipped.`,
      },
      {
        line: 5,
        title: "A pointer offsets",
        phase: "aOffsets",
        note: (s) =>
          `For row-major A, row ${s.row} starts at row * N = ${s.row * s.N}. Add cols to walk across that row.`,
      },
      {
        line: 6,
        title: "x pointer offsets",
        phase: "xOffsets",
        note: () =>
          "The vector x is shared by every row program, so its offsets are just cols.",
      },
      {
        line: 7,
        title: "Load A row",
        phase: "loadA",
        note: () =>
          "tl.load gathers A values for the active lanes. Masked lanes receive other=0.0 in this visualization.",
      },
      {
        line: 8,
        title: "Load x",
        phase: "loadX",
        note: () =>
          "The same lane columns load matching x values, so each lane now owns one A[row, col] and one x[col].",
      },
      {
        line: 9,
        title: "Multiply lanes",
        phase: "multiply",
        note: () =>
          "Each active lane multiplies one matrix value by one vector value. This is the SIMD-shaped part.",
      },
      {
        line: 10,
        title: "Reduce",
        phase: "reduce",
        note: () =>
          "tl.sum reduces the block of lane products into the single scalar output for this row.",
      },
      {
        line: 11,
        title: "Store output",
        phase: "store",
        note: (s) =>
          `The scalar result is stored at output_ptr + row, so only output[${s.row}] changes for this program.`,
      },
    ],
    concepts: [
      ["Program id", "<code>tl.program_id(0)</code> chooses the output row."],
      ["Column lanes", "<code>tl.arange(0, BLOCK_SIZE)</code> creates SIMD-like lane offsets."],
      ["A offsets", "<code>row * N + cols</code> walks across one matrix row in row-major memory."],
      ["Mask", "<code>cols &lt; N</code> disables lanes past the real row width."],
    ],
  },
  softmax: {
    title: "Fused softmax visualizer",
    file: "scaled_softmax_kernel.py",
    programTitle: "One program per logits row",
    mLabel: "Rows",
    nLabel: "Logits per row",
    showTemperature: true,
    codeLines: [
      "import triton",
      "import triton.language as tl",
      "import numpy as np",
      "",
      "@triton.jit",
      "def scaled_softmax_kernel(input_ptr, output_ptr, temperature, n_cols, BLOCK_SIZE: tl.constexpr):",
      "    row_idx = tl.program_id(0)",
      "    col_offsets = tl.arange(0, BLOCK_SIZE)",
      "    mask = col_offsets < n_cols",
      "    row_start = row_idx * n_cols",
      "    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))",
      "    x = x / temperature",
      "    x_max = tl.max(x, axis=0)",
      "    x_exp = tl.exp(x - x_max)",
      "    x_sum = tl.sum(x_exp, axis=0)",
      "    result = x_exp / x_sum",
      "    tl.store(output_ptr + row_start + col_offsets, result, mask=mask)",
    ],
    steps: [
      {
        line: 6,
        title: "Launch row program",
        phase: "program",
        note: (s) =>
          `The grid is (${s.M},), so program ${s.row} owns one logits row and computes its full softmax.`,
      },
      {
        line: 7,
        title: "Build column lanes",
        phase: "lanes",
        note: (s) =>
          `tl.arange creates ${s.BLOCK_SIZE} lanes for candidate columns 0..${s.BLOCK_SIZE - 1}.`,
      },
      {
        line: 8,
        title: "Create mask",
        phase: "mask",
        note: (s) =>
          `Only ${s.N} logits are real in this row. Masked lanes load -inf so they vanish after exp.`,
      },
      {
        line: 9,
        title: "Find row start",
        phase: "rowStart",
        note: (s) =>
          `row_start = row_idx * n_cols = ${s.row} * ${s.N} = ${s.row * s.N}, the flat offset for this row.`,
      },
      {
        line: 10,
        title: "Load logits",
        phase: "load",
        note: () =>
          "tl.load gathers one row of logits. Out-of-row lanes are filled with -inf instead of real memory.",
      },
      {
        line: 11,
        title: "Apply temperature",
        phase: "scale",
        note: (s) =>
          `Each lane divides by temperature ${format(s.temperature)}. Lower values sharpen the distribution; higher values flatten it.`,
      },
      {
        line: 12,
        title: "Reduce max",
        phase: "max",
        note: () =>
          "tl.max finds the row maximum inside the block, preparing a numerically stable softmax.",
      },
      {
        line: 13,
        title: "Exponentiate shifted logits",
        phase: "exp",
        note: () =>
          "Subtracting x_max before tl.exp keeps the largest exponent at 1.0 and avoids overflow.",
      },
      {
        line: 14,
        title: "Reduce sum",
        phase: "sum",
        note: () =>
          "tl.sum adds the exponentials across active lanes to produce the softmax denominator.",
      },
      {
        line: 15,
        title: "Normalize",
        phase: "normalize",
        note: () =>
          "Each lane divides by the shared sum, producing probabilities that add up to 1 for the row.",
      },
      {
        line: 16,
        title: "Store probabilities",
        phase: "store",
        note: (s) =>
          `Active lanes write output_ptr + ${s.row * s.N} + col_offsets, so the whole row ${s.row} is stored at once.`,
      },
    ],
    concepts: [
      ["Temperature", "<code>x / temperature</code> changes the entropy before softmax."],
      ["Stability", "<code>x - x_max</code> keeps exponentials bounded without changing probabilities."],
      ["Block reductions", "<code>tl.max</code> and <code>tl.sum</code> reduce values across lanes in one program."],
      ["Masked lanes", "<code>other=-inf</code> makes padded lanes contribute zero after exponentiation."],
    ],
  },
};

const initial = {
  kernel: "matvec",
  M: 6,
  N: 10,
  BLOCK_SIZE: 16,
  row: 0,
  step: 0,
  temperature: 1,
  running: false,
};

let state = { ...initial };
let timer = null;

const els = {
  kernelTitle: document.querySelector("#kernelTitle"),
  kernelButtons: document.querySelectorAll("[data-kernel]"),
  mLabel: document.querySelector("#mLabel"),
  nLabel: document.querySelector("#nLabel"),
  mInput: document.querySelector("#mInput"),
  nInput: document.querySelector("#nInput"),
  mValue: document.querySelector("#mValue"),
  nValue: document.querySelector("#nValue"),
  temperatureControl: document.querySelector("#temperatureControl"),
  temperatureInput: document.querySelector("#temperatureInput"),
  temperatureValue: document.querySelector("#temperatureValue"),
  rowSelect: document.querySelector("#rowSelect"),
  gridLabel: document.querySelector("#gridLabel"),
  programLabel: document.querySelector("#programLabel"),
  stepLabel: document.querySelector("#stepLabel"),
  codeFileLabel: document.querySelector("#codeFileLabel"),
  codeProgramLabel: document.querySelector("#codeProgramLabel"),
  codeBlock: document.querySelector("#codeBlock"),
  stepNote: document.querySelector("#stepNote"),
  programPanelTitle: document.querySelector("#programPanelTitle"),
  rowPrograms: document.querySelector("#rowPrograms"),
  laneGrid: document.querySelector("#laneGrid"),
  laneCaption: document.querySelector("#laneCaption"),
  matrixTitle: document.querySelector("#matrixTitle"),
  matrixGrid: document.querySelector("#matrixGrid"),
  xTitle: document.querySelector("#xTitle"),
  xOffsetLabel: document.querySelector("#xOffsetLabel"),
  xGrid: document.querySelector("#xGrid"),
  outTitle: document.querySelector("#outTitle"),
  outOffsetLabel: document.querySelector("#outOffsetLabel"),
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
  conceptTitles: [
    document.querySelector("#conceptOneTitle"),
    document.querySelector("#conceptTwoTitle"),
    document.querySelector("#conceptThreeTitle"),
    document.querySelector("#conceptFourTitle"),
  ],
  conceptBodies: [
    document.querySelector("#conceptOneBody"),
    document.querySelector("#conceptTwoBody"),
    document.querySelector("#conceptThreeBody"),
    document.querySelector("#conceptFourBody"),
  ],
};

function activeKernel() {
  return kernels[state.kernel];
}

function activeSteps() {
  return activeKernel().steps;
}

function valueA(row, col) {
  const raw = ((row + 2) * (col + 3) + row * 7 - col * 2) % 17;
  return Number(((raw - 8) / 4).toFixed(2));
}

function valueX(col) {
  const raw = ((col + 5) * 7) % 13;
  return Number(((raw - 6) / 3).toFixed(2));
}

function valueLogit(row, col) {
  const wave = Math.sin((row + 1) * (col + 2) * 0.72) * 2.1;
  const trend = (row - col * 0.18) + ((col % 3) - 1) * 0.45;
  return Number((wave + trend).toFixed(2));
}

function rowProducts(row = state.row) {
  return Array.from({ length: state.N }, (_, col) => valueA(row, col) * valueX(col));
}

function rowSum(row = state.row) {
  return rowProducts(row).reduce((acc, value) => acc + value, 0);
}

function softmaxData(row = state.row) {
  const logits = Array.from({ length: state.N }, (_, col) => valueLogit(row, col));
  const scaled = logits.map((value) => value / state.temperature);
  const max = Math.max(...scaled);
  const shifted = scaled.map((value) => value - max);
  const exp = shifted.map((value) => Math.exp(value));
  const sum = exp.reduce((acc, value) => acc + value, 0);
  const result = exp.map((value) => value / sum);
  return { logits, scaled, max, shifted, exp, sum, result };
}

function isPhaseAtLeast(name) {
  const steps = activeSteps();
  const currentIndex = steps.findIndex((step) => step.phase === steps[state.step].phase);
  const targetIndex = steps.findIndex((step) => step.phase === name);
  return currentIndex >= targetIndex;
}

function format(value, digits = 2) {
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  return normalized.toFixed(digits);
}

function syncInputs() {
  const kernel = activeKernel();
  els.kernelTitle.textContent = kernel.title;
  els.mLabel.textContent = kernel.mLabel;
  els.nLabel.textContent = kernel.nLabel;
  els.mInput.value = state.M;
  els.nInput.value = state.N;
  els.mValue.textContent = state.M;
  els.nValue.textContent = state.N;
  els.temperatureInput.value = Math.round(state.temperature * 10);
  els.temperatureValue.textContent = format(state.temperature, 1);
  els.temperatureControl.classList.toggle("hidden", !kernel.showTemperature);
  els.gridLabel.textContent = `(${state.M},) programs`;
  els.programLabel.textContent = `row ${state.row}`;
  els.stepLabel.textContent = `${state.step + 1} / ${activeSteps().length}`;
  els.codeFileLabel.textContent = kernel.file;
  els.codeProgramLabel.textContent = `program ${state.row} / ${state.M - 1}`;
  els.programPanelTitle.textContent = kernel.programTitle;
  els.rowFormula.textContent = state.kernel === "matvec"
    ? `row = tl.program_id(0) = ${state.row}`
    : `row_idx = tl.program_id(0) = ${state.row}`;
  els.aOffsetLabel.textContent = state.kernel === "matvec"
    ? `a_ptr + ${state.row} * ${state.N} + cols`
    : `input_ptr + ${state.row * state.N} + col_offsets`;
  els.blockButtons.forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.block) === state.BLOCK_SIZE);
  });
  els.kernelButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.kernel === state.kernel);
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
  const kernel = activeKernel();
  const activeLine = activeSteps()[state.step].line;
  els.codeBlock.innerHTML = "";
  kernel.codeLines.forEach((line, index) => {
    const span = document.createElement("span");
    span.className = "code-line";
    if (index === activeLine) span.classList.add("active");
    if (index < activeLine) span.classList.add("done");
    span.textContent = line;
    els.codeBlock.append(span);
  });
  const current = activeSteps()[state.step];
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
    const outputLabel = state.kernel === "matvec" ? `output[${row}]` : `output row ${row}`;
    item.innerHTML = `<span>program ${row}</span><span class="row-bar"></span><span>${outputLabel}</span>`;
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
  const laneName = state.kernel === "matvec" ? "cols" : "col_offsets";
  els.laneCaption.textContent = `${state.BLOCK_SIZE} lanes touch ${laneName} 0..${state.BLOCK_SIZE - 1}`;
  els.laneGrid.style.gridTemplateColumns = `repeat(${Math.min(state.BLOCK_SIZE, 16)}, minmax(44px, 1fr))`;
  els.laneGrid.innerHTML = "";
  for (let col = 0; col < state.BLOCK_SIZE; col += 1) {
    const lane = document.createElement("div");
    const inMask = col < state.N;
    lane.className = "lane";
    if (isPhaseAtLeast("mask")) lane.classList.add(inMask ? "in-mask" : "out-mask");
    lane.innerHTML = `<small>lane</small>${col}`;
    lane.title = inMask ? `${laneName}[${col}] = ${col}, mask=True` : `${laneName}[${col}] = ${col}, mask=False`;
    els.laneGrid.append(lane);
  }
}

function renderMatrix() {
  els.matrixGrid.innerHTML = "";
  els.matrixGrid.style.gridTemplateColumns = "1fr";
  els.matrixTitle.textContent = state.kernel === "matvec"
    ? "A memory, flattened row-major"
    : "input logits, flattened row-major";

  for (let row = 0; row < state.M; row += 1) {
    const rowEl = document.createElement("div");
    rowEl.className = "matrix-row";
    rowEl.style.gridTemplateColumns = `repeat(${state.N}, minmax(48px, 1fr))`;
    for (let col = 0; col < state.N; col += 1) {
      const cell = document.createElement("div");
      const offsetPhase = state.kernel === "matvec" ? "aOffsets" : "rowStart";
      const active = row === state.row && col < state.BLOCK_SIZE && isPhaseAtLeast(offsetPhase);
      cell.className = "cell";
      if (active) cell.classList.add("active");
      const value = state.kernel === "matvec" ? valueA(row, col) : valueLogit(row, col);
      cell.innerHTML = `<small>${row * state.N + col}</small>${format(value)}`;
      cell.title = `${state.kernel === "matvec" ? "A" : "input"}[${row}, ${col}], flat offset ${row * state.N + col}`;
      rowEl.append(cell);
    }
    els.matrixGrid.append(rowEl);
  }
}

function renderMatvecVector() {
  els.xTitle.textContent = "x vector";
  els.xOffsetLabel.textContent = "x_ptr + cols";
  els.outTitle.textContent = "output vector";
  els.outOffsetLabel.textContent = "output_ptr + row";
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

function renderSoftmaxVector() {
  const data = softmaxData();
  els.xTitle.textContent = "scaled logits";
  els.xOffsetLabel.textContent = "x / temperature";
  els.outTitle.textContent = "output row probabilities";
  els.outOffsetLabel.textContent = "output_ptr + row_start + col_offsets";
  els.xGrid.innerHTML = "";
  els.outGrid.innerHTML = "";

  for (let col = 0; col < state.N; col += 1) {
    const cell = document.createElement("div");
    const active = col < state.BLOCK_SIZE && isPhaseAtLeast("scale");
    cell.className = "vector-cell";
    if (active) cell.classList.add("active");
    cell.innerHTML = `<small>${col}</small>${active ? format(data.scaled[col]) : "-"}`;
    cell.title = `input[${state.row}, ${col}] / ${format(state.temperature, 1)}`;
    els.xGrid.append(cell);
  }
  for (let col = 0; col < state.N; col += 1) {
    const cell = document.createElement("div");
    const stored = col < state.BLOCK_SIZE && isPhaseAtLeast("store");
    cell.className = "vector-cell";
    if (stored) cell.classList.add("active");
    cell.innerHTML = `<small>${col}</small>${stored ? format(data.result[col], 3) : "0.000"}`;
    cell.title = stored ? `stored probability output[${state.row}, ${col}]` : `output[${state.row}, ${col}] not stored yet`;
    els.outGrid.append(cell);
  }
}

function renderVector() {
  if (state.kernel === "matvec") {
    renderMatvecVector();
  } else {
    renderSoftmaxVector();
  }
}

function renderMatvecCompute() {
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
    cell.innerHTML = inMask
      ? `A[${state.row},${col}]<br><strong>${format(a)}</strong> x <strong>${format(x)}</strong><br>= ${format(product)}`
      : `col ${col}<br><strong>masked</strong><br>= 0.00`;
    els.multiplyGrid.append(cell);
  }
  els.sumLine.textContent = isPhaseAtLeast("reduce")
    ? `tl.sum(products) = ${products.map(format).join(" + ")} = ${format(rowSum())}`
    : "tl.sum waits until the lane products exist.";
}

function renderSoftmaxCompute() {
  const data = softmaxData();
  els.computeTitle.textContent = `Fused softmax for row ${state.row}`;
  els.multiplyGrid.innerHTML = "";
  els.multiplyGrid.style.gridTemplateColumns = `repeat(${Math.min(state.BLOCK_SIZE, 8)}, minmax(98px, 1fr))`;

  for (let col = 0; col < state.BLOCK_SIZE; col += 1) {
    const inMask = col < state.N;
    const cell = document.createElement("div");
    cell.className = "multiply-cell";
    if (isPhaseAtLeast("load")) cell.classList.add(inMask ? "active" : "masked");
    if (!inMask) {
      cell.innerHTML = `col ${col}<br><strong>masked</strong><br>-inf`;
    } else if (isPhaseAtLeast("normalize")) {
      cell.innerHTML = `p[${col}]<br><strong class="cool">${format(data.result[col], 3)}</strong><br>exp/sum`;
    } else if (isPhaseAtLeast("exp")) {
      cell.innerHTML = `exp lane ${col}<br><strong>${format(data.exp[col], 3)}</strong><br>x-max ${format(data.shifted[col])}`;
    } else if (isPhaseAtLeast("scale")) {
      cell.innerHTML = `x[${col}] / T<br><strong>${format(data.scaled[col])}</strong><br>raw ${format(data.logits[col])}`;
    } else {
      cell.innerHTML = `input[${state.row},${col}]<br><strong>${format(data.logits[col])}</strong><br>offset ${state.row * state.N + col}`;
    }
    els.multiplyGrid.append(cell);
  }

  if (isPhaseAtLeast("sum")) {
    els.sumLine.textContent = `x_max = ${format(data.max)}, x_sum = ${format(data.sum, 4)}, probabilities sum = ${format(data.result.reduce((acc, value) => acc + value, 0), 4)}`;
  } else if (isPhaseAtLeast("max")) {
    els.sumLine.textContent = `tl.max(x) = ${format(data.max)}. Each active lane will exponentiate x - x_max.`;
  } else {
    els.sumLine.textContent = "The row is loaded once, then all softmax math stays fused inside this program.";
  }
}

function renderCompute() {
  if (state.kernel === "matvec") {
    renderMatvecCompute();
  } else {
    renderSoftmaxCompute();
  }
}

function renderConcepts() {
  activeKernel().concepts.forEach(([title, body], index) => {
    els.conceptTitles[index].textContent = title;
    els.conceptBodies[index].innerHTML = body;
  });
}

function render() {
  syncInputs();
  renderRowSelect();
  renderCode();
  renderPrograms();
  renderLanes();
  renderMatrix();
  renderVector();
  renderCompute();
  renderConcepts();
}

function nextStep() {
  const steps = activeSteps();
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
  const steps = activeSteps();
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

function switchKernel(kernelName) {
  state.kernel = kernelName;
  state.step = 0;
  state.row = Math.min(state.row, state.M - 1);
  stopRun();
  render();
}

els.kernelButtons.forEach((button) => {
  button.addEventListener("click", () => switchKernel(button.dataset.kernel));
});

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

els.temperatureInput.addEventListener("input", (event) => {
  state.temperature = Number(event.target.value) / 10;
  state.step = Math.max(state.step, activeSteps().findIndex((step) => step.phase === "scale"));
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
  const kernel = state.kernel;
  state = { ...initial, kernel };
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
