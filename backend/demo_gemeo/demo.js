const get = (id) => document.getElementById(id);
const play = get("play");
const step = get("step");
const reset = get("reset");
const scoreText = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const metricText = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });
const clock = new Intl.DateTimeFormat("pt-BR", { timeStyle: "medium", timeZone: "UTC" });
const fields = [
  ["type", "Tipo do produto", ""],
  ["air_temperature_k", "Temperatura do ar", " K"],
  ["process_temperature_k", "Temperatura do processo", " K"],
  ["rotational_speed_rpm", "Rotação", " rpm"],
  ["torque_nm", "Torque", " Nm"],
  ["tool_wear_min", "Desgaste da ferramenta", " min"],
];

let replay = [];
let history = [];
let playing = false;
let busy = false;
let timer;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : "A API não conseguiu processar a leitura.");
  }
  return response.json();
}

function showError(error) {
  get("error").hidden = false;
  get("error").textContent = error.message;
}

function updateControls() {
  const atEnd = history.length >= replay.length;
  play.textContent = playing ? "Pausar" : "Reproduzir";
  play.disabled = atEnd || (!playing && busy);
  step.disabled = busy || playing || atEnd;
  reset.disabled = busy || !replay.length;
  get("progress").textContent = replay.length
    ? `${history.length} de ${replay.length} leituras${atEnd ? " · fim da reprodução" : ""}`
    : "Fonte de leituras indisponível";
}

function graph() {
  const recent = history.slice(-30);
  const x = (index) => 42 + 530 * index / Math.max(1, recent.length - 1);
  const y = (value) => 15 + 115 * (1 - value);
  const edge = recent.at(-1)?.result.threshold ?? 0.5;
  const axes = [0, 0.5, 1].map((value) =>
    `<line x1="42" x2="572" y1="${y(value)}" y2="${y(value)}" stroke="#d8e4df"/>
     <text x="32" y="${y(value) + 4}" text-anchor="end" fill="#526965">${String(value).replace(".", ",")}</text>`
  ).join("");
  const path = recent.map((row, i) => `${x(i)},${y(row.result.model_score)}`).join(" ");
  const points = recent.map((row, i) => {
    const alert = row.result.inspection_recommended;
    return `<circle cx="${x(i)}" cy="${y(row.result.model_score)}" r="${alert ? 4.8 : 3}" fill="${alert ? "#973d1d" : "#176657"}"><title>Leitura ${row.udi}, score ${scoreText.format(row.result.model_score)}</title></circle>`;
  }).join("");
  get("chart").innerHTML = `<svg viewBox="0 0 590 155" aria-hidden="true" font-size="13">
    ${axes}<line x1="42" x2="572" y1="${y(edge)}" y2="${y(edge)}" stroke="#825b41" stroke-dasharray="5 4"/>
    <polyline fill="none" stroke="#176657" stroke-width="2.4" points="${path}"/>${points}
    ${recent.length ? `<text x="42" y="149" fill="#526965">UDI ${recent[0].udi}</text>` : ""}
    ${recent.length > 1 ? `<text x="572" y="149" fill="#526965" text-anchor="end">UDI ${recent.at(-1).udi}</text>` : ""}
    </svg>`;
}

function render() {
  const last = history.at(-1);
  const { result, measurements } = last;
  get("status").dataset.alert = String(result.inspection_recommended);
  get("decision").textContent = result.inspection_recommended ? "Inspeção indicada" : "Sem alerta nesta leitura";
  get("stamp").textContent = `UDI ${last.udi} · ${clock.format(new Date(last.timestamp))} (horário simulado)`;
  get("score").textContent = scoreText.format(result.model_score);
  get("threshold").textContent = `Limiar: ${scoreText.format(result.threshold)}`;
  const readings = get("readings");
  readings.replaceChildren();
  for (const [key, label, unit] of fields) {
    const row = document.createElement("div");
    const name = document.createElement("dt");
    const value = document.createElement("dd");
    name.textContent = label;
    value.textContent = unit ? metricText.format(measurements[key]) + unit : measurements[key];
    row.append(name, value);
    readings.append(row);
  }
  const range = get("range");
  range.hidden = !result.out_of_training_range.length;
  range.textContent = result.out_of_training_range.length
    ? "Fora das faixas do treino: " + fields.filter(([key]) => result.out_of_training_range.includes(key))
      .map(([, label]) => label).join(", ") + "."
    : "";
  const count = history.filter((entry) => entry.result.inspection_recommended).length;
  get("alerts").textContent = count === 1 ? "1 alerta" : `${count} alertas`;

  const row = document.createElement("tr");
  row.dataset.alert = String(result.inspection_recommended);
  const values = [
    clock.format(new Date(last.timestamp)), last.udi,
    metricText.format(measurements.rotational_speed_rpm) + " rpm",
    metricText.format(measurements.tool_wear_min) + " min",
    scoreText.format(result.model_score), result.inspection_recommended ? "Inspeção indicada" : "Sem alerta",
  ];
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  get("history").prepend(row);
  graph();
}

function pause() {
  playing = false;
  clearTimeout(timer);
  updateControls();
}

async function next() {
  if (busy || history.length >= replay.length) return;
  busy = true;
  updateControls();
  get("error").hidden = true;
  try {
    const reading = replay[history.length];
    const result = await request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reading.measurements),
    });
    history.push({ ...reading, result });
    render();
  } catch (error) {
    pause();
    showError(error);
  } finally {
    busy = false;
    if (history.length >= replay.length) pause();
    updateControls();
  }
  if (playing) timer = setTimeout(next, 1100);
}

play.addEventListener("click", () => {
  if (playing) {
    pause();
  } else {
    playing = true;
    next();
  }
});
step.addEventListener("click", next);
reset.addEventListener("click", () => {
  pause();
  history = [];
  get("history").replaceChildren();
  next();
});

request("/demo/leituras").then(({ readings }) => {
  replay = readings;
  next();
}).catch((error) => {
  showError(error);
  updateControls();
});
