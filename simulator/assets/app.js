/*
 * LogicTune replay app: loads the recorded evaluation (replay.json) and drives
 * two simulation lanes plus the results charts. No data is generated here;
 * every response, rule, score, and verdict comes from the exported run.
 */
(function () {
  "use strict";
  const { Lane, hasTrafficLight, outcomeText, FAIR_GREEN_TICK } = window.LogicTuneSim;
  const Charts = window.LogicTuneCharts;
  const ReplayState = window.LogicTuneState;
  const pct = Charts.pct;
  const SPEC_LABEL = (id) => "φ" + id.split("_")[1];

  const state = { index: null, prefix: "data/", run: null, data: null, scenario: 1, sample: 0, playing: false, speed: 1.5, lanes: [], models: [null, null] };

  // ------------------------------------------------------------ data loading
  // results/latest/index.json lists the exported runs; each run folder holds
  // replay.json and the CSV files. The deployed site serves them under data/.
  async function fetchJson(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }
  async function loadIndex() {
    for (const prefix of ReplayState.dataPrefixes(location.pathname)) {
      try { return { prefix, index: await fetchJson(prefix + "index.json") }; } catch (e) { /* try next */ }
    }
    throw new Error("index.json not found");
  }
  const runDir = () => state.prefix + state.run.path;

  function cssColors() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    return {
      ground: v("--sim-ground"), road: v("--sim-road"), marking: v("--sim-marking"), crosswalk: v("--sim-crosswalk"),
      ego: v("--sim-ego"), other: v("--sim-other"), egoGlass: v("--sim-glass"), pedestrian: v("--sim-ped"),
      signalBox: v("--sim-box"), signalLabel: v("--sim-label"), lampOff: v("--sim-off"), red: "#e34948", green: "#22c55e", critical: v("--critical"),
    };
  }
  let colors = null;
  const getColors = () => colors || (colors = cssColors());

  // --------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const scenarioById = (id) => state.data.scenarios.find((s) => s.id === id);
  const modelById = (id) => state.data.models.find((m) => m.id === id);
  const samplesFor = (modelId, scenarioId) => ReplayState.samplesFor(state.data, modelId, scenarioId);

  // --------------------------------------------------------------- headline
  function renderTiles() {
    const d = state.data, h = d.headline, rounds = d.rounds;
    const base = rounds[0], fin = rounds[rounds.length - 1];
    const ci = h.change_ci95_pp ? `95% CI ${h.change_ci95_pp[0].toFixed(1)} to ${h.change_ci95_pp[1].toFixed(1)} pts` : "";
    const p = h.p_value !== undefined ? ` · p ${h.p_value < 0.001 ? "< 0.001" : "= " + h.p_value.toFixed(3)}` : "";
    const tiles = [
      { label: "Base model satisfaction", value: pct(base.satisfaction_rate), delta: "before DPO fine-tuning", key: "base" },
      { label: "Fine-tuned satisfaction", value: pct(fin.satisfaction_rate), delta: `after ${h.final_round} DPO round${h.final_round === 1 ? "" : "s"}`, key: "ft" },
      { label: "Change", value: `${h.change_pp >= 0 ? "+" : "−"}${Math.abs(h.change_pp).toFixed(1)} pts`, delta: ci + p },
      { label: "All 15 specs satisfied", value: `${pct(base.full_pass_rate)} → ${pct(fin.full_pass_rate)}`, delta: "share of responses, base → fine-tuned" },
    ];
    $("#tiles").innerHTML = tiles.map((t) => `<div class="tile"><div class="label">${t.key ? `<i class="key ${t.key}"></i>` : ""}${esc(t.label)}</div><div class="value">${esc(t.value)}</div><div class="delta">${esc(t.delta)}</div></div>`).join("");
    const pv = d.provenance || {};
    const parts = [`Model: ${d.headline.base_model}`, `run ${pv.run_name || "?"}`];
    if (pv.gpu) parts.push(pv.gpu);
    if (pv.source_commit) parts.push(`source ${pv.source_commit.slice(0, 7)}`);
    if (pv.exported_at) parts.push(`exported ${pv.exported_at.slice(0, 10)}`);
    $("#provenance").textContent = `${parts.join(" · ")}. Evaluation: ${d.samples[d.models[0].id].length} recorded responses per model (17 scenarios × 2 unseen prompt templates × 10 samples).`;
    $("#foot-commit").textContent = pv.source_commit ? `source commit ${pv.source_commit.slice(0, 7)}` : "";
  }

  // ------------------------------------------------------------- scenarios
  function renderScenarioSelect() {
    const sel = $("#scenario");
    sel.innerHTML = state.data.scenarios.map((s) => `<option value="${s.id}">${s.id}. ${esc(s.text)}${s.heldout ? " (held out)" : ""}</option>`).join("");
    sel.value = state.scenario;
  }

  function renderStrip() {
    const strip = $("#strip");
    strip.innerHTML = "";
    state.models.forEach((mid, laneIdx) => {
      const m = modelById(mid);
      const who = document.createElement("div");
      who.className = "who";
      who.innerHTML = `<i class="key ${laneIdx === 0 ? "base" : "ft"}" style="width:10px;height:10px;border-radius:3px;display:inline-block"></i>${esc(m.label)}`;
      const cells = document.createElement("div");
      cells.className = "cells";
      samplesFor(mid, state.scenario).forEach((x, i) => {
        const b = document.createElement("button");
        b.type = "button";
        // Background ramps from --surface-2 (0) to --score-hi (15). Light text
        // from 13 up keeps every score at 4.5:1 or more in both themes.
        b.className = "cell" + (i === state.sample ? " sel" : "") + (x.score >= 13 ? " hi" : "");
        b.style.setProperty("--share", `${Math.round((x.score / 15) * 100)}%`);
        b.textContent = x.score;
        b.title = `Prompt template ${x.template + 1}, sample ${x.sample_idx + 1}: ${x.score}/15`;
        b.setAttribute("aria-label", b.title);
        b.setAttribute("aria-pressed", String(i === state.sample));
        b.addEventListener("click", () => { state.sample = i; refresh(); });
        cells.appendChild(b);
      });
      strip.append(who, cells);
    });
  }

  // ------------------------------------------------------------------ lanes
  function buildLanes() {
    const tpl = $("#lane-tpl");
    document.querySelectorAll(".lane").forEach((node, idx) => {
      node.appendChild(tpl.content.cloneNode(true));
      const sel = $(".model", node);
      sel.addEventListener("change", () => { state.models[idx] = sel.value; refresh(); });
      const lane = new Lane($("canvas", node), { colors: getColors, onTick: () => {} });
      state.lanes.push({ node, lane, lastRule: null, lastViolations: -1 });
    });
  }

  function populateModelSelects() {
    state.lanes.forEach((L, idx) => {
      const sel = $(".model", L.node);
      sel.innerHTML = state.data.models.map((m) => `<option value="${m.id}">${esc(m.label)} · ${pct(m.satisfaction_rate)}</option>`).join("");
      sel.value = state.models[idx];
    });
  }

  function renderLane(idx) {
    const L = state.lanes[idx], node = L.node;
    const samples = samplesFor(state.models[idx], state.scenario);
    const sample = samples[Math.min(state.sample, samples.length - 1)];
    const scen = scenarioById(state.scenario);
    L.sample = sample;
    L.lane.load(scen, sample.rules, scen.id);
    const pill = $(".score-pill", node);
    pill.textContent = `${sample.score}/15 specs`;
    // verdict chips: focusable list items with the formula in a tooltip
    const verdicts = $(".verdicts", node);
    verdicts.innerHTML = "";
    state.data.specs.forEach((s, j) => {
      const ok = sample.specs[j] === 1, name = `${SPEC_LABEL(s.id)} ${ok ? "satisfied" : "violated"}`;
      const chip = document.createElement("span");
      chip.className = `chip ${ok ? "pass" : "fail"}`;
      chip.setAttribute("role", "listitem");
      chip.setAttribute("aria-label", `${name}: ${s.description}`);
      chip.innerHTML = `<span class="ic" aria-hidden="true">${ok ? "✓" : "✗"}</span>${SPEC_LABEL(s.id)}`;
      Charts.bindTip(chip, `<b>${name}</b><br>${esc(s.formula)}<br>${esc(s.description)}`);
      verdicts.appendChild(chip);
    });
    // rules
    const rules = sample.rules;
    $(".rules", node).innerHTML = rules.length
      ? rules.map((r) => `<li><span class="guard">if ${esc(r.guard_text)}</span> → <span class="act">${esc(r.action)}</span></li>`).join("")
      : `<li class="muted">No rule could be parsed. The action is undefined in every state.</li>`;
    const free = Object.entries(sample.free_atoms || {});
    $(".free", node).textContent = free.length
      ? `Free atoms (unknown phrases): ${free.map(([k, v]) => `${k} = "${v}"`).join(", ")}. The simulation treats them as false; the model checker explores both values.`
      : "";
    $(".prompt", node).textContent = `Prompt: ${sample.prompt}`;
    $("pre", node).textContent = sample.response;
    L.lastRule = null; L.lastViolations = -1;
    updateLaneHud(idx);
  }

  // Scroll the rules list (not the page) so the firing rule is visible.
  function revealRule(ol, idx) {
    const li = idx >= 0 ? ol.children[idx] : null;
    if (!li) return;
    if (li.offsetTop < ol.scrollTop || li.offsetTop + li.offsetHeight > ol.scrollTop + ol.clientHeight) {
      ol.scrollTop = li.offsetTop - ol.clientHeight / 3;
    }
  }

  function updateLaneHud(idx) {
    const L = state.lanes[idx], lane = L.lane, node = L.node;
    const d = lane.decision;
    const act = $(".hud-action", node);
    act.textContent = d.action ? d.action : "NO RULE";
    act.classList.toggle("undef", !d.action);
    const t = `t = ${(lane.t / 60).toFixed(1)} s`;
    $(".hud-rule", node).textContent = lane.done ? `last decision · ${t}` : d.rule >= 0 ? `rule ${d.rule + 1} · ${t}` : `undefined · ${t}`;
    if (L.lastRule !== d.rule) {
      node.querySelectorAll(".rules li").forEach((li, i) => li.classList.toggle("hit", i === d.rule));
      revealRule($(".rules", node), d.rule);
      L.lastRule = d.rule;
    }
    if (L.lastViolations !== lane.violations.length) {
      $(".violations", node).innerHTML = lane.violations.length
        ? lane.violations.map((v) => `<li><span class="bad">${SPEC_LABEL(v.id)} violated</span> at ${v.t.toFixed(1)} s: ${esc(v.why)}</li>`).join("")
        : `<li class="muted">No violation yet.</li>`;
      L.lastViolations = lane.violations.length;
    }
    const out = $(".outcome", node);
    if (lane.done) {
      out.hidden = false;
      out.textContent = outcomeText(lane.outcome, lane.doneAt);
    } else out.hidden = true;
  }

  // -------------------------------------------------------------- playback
  let raf = null;
  function frame() {
    // Every lane steps until both cars are done and the shared timeline has
    // played out, so a lane whose car has left keeps showing the same lights,
    // other road users and time as the other lane.
    state.lanes.forEach((L) => L.lane.step(state.speed));
    state.lanes.forEach((L, i) => { L.lane.draw(); updateLaneHud(i); });
    const running = state.lanes.some((L) => !L.lane.finished());
    if (running && state.playing) raf = requestAnimationFrame(frame);
    else { state.playing = false; $("#play").textContent = "Play"; }
  }
  function play() {
    if (state.lanes.every((L) => L.lane.finished())) state.lanes.forEach((L) => L.lane.reset());
    state.playing = !state.playing;
    $("#play").textContent = state.playing ? "Pause" : "Play";
    if (state.playing) raf = requestAnimationFrame(frame);
    else cancelAnimationFrame(raf);
  }
  function resetAll() {
    state.playing = false; cancelAnimationFrame(raf); $("#play").textContent = "Play";
    state.lanes.forEach((L, i) => { L.lane.reset(); renderLane(i); });
  }

  function refresh() {
    renderScenarioSelect();
    state.lanes.forEach((L, i) => { $(".model", L.node).value = state.models[i]; });
    const scen = scenarioById(state.scenario);
    const noLight = hasTrafficLight(scen.id) ? "" : `<span class="note">No traffic light: the controller reads <code>green_traffic_light</code> as false (not green) until ${FAIR_GREEN_TICK / 60} s, when it turns true as the model checker's fairness assumption requires.</span>`;
    $("#scenario-text").innerHTML = `${esc(scen.text)} <span class="tag">intent: ${esc(scen.intent)}</span>${scen.heldout ? '<span class="tag">held out of training</span>' : ""}${noLight}`;
    renderStrip();
    resetAll();
    writeHash();
  }

  // ----------------------------------------------------------------- results
  function renderResults() {
    const d = state.data, rounds = d.rounds, base = rounds[0], fin = rounds[rounds.length - 1];
    const h = d.headline;
    $("#results-hint").textContent = `Evaluation on prompts never used in training (two unseen instruction templates). Scenarios ${d.scenarios.filter((s) => s.heldout).map((s) => s.id).join(", ")} were also held out of training entirely. Held-out scenarios: ${pct(base.heldout_satisfaction)} → ${pct(fin.heldout_satisfaction)}.`;
    Charts.roundsLine($("#chart-rounds"), rounds, "satisfaction_rate", "satisfaction_ci95", "Satisfaction");
    Charts.roundsLine($("#chart-fullpass"), rounds, "full_pass_rate", null, "Full pass");
    Charts.pairedBars($("#chart-specs"), d.specs.map((s) => ({ name: SPEC_LABEL(s.id), title: `${SPEC_LABEL(s.id)}: ${esc(s.formula)}`, base: base.per_spec[s.id], ft: fin.per_spec[s.id] })),
      { label: "Per-specification satisfaction", baseLabel: base.label, ftLabel: fin.label, labelWidth: 44 });
    const baseScen = d.samples[d.models[0].id], finScen = d.samples[d.models[d.models.length - 1].id];
    const mean = (arr) => arr.reduce((a, x) => a + x.score / 15, 0) / Math.max(1, arr.length);
    Charts.dumbbell($("#chart-scen"), d.scenarios.map((s) => ({
      id: s.id, text: s.text, heldout: s.heldout,
      base: mean(baseScen.filter((x) => x.scenario_id === s.id)), ft: mean(finScen.filter((x) => x.scenario_id === s.id)),
    })), { label: "Per-scenario satisfaction", baseLabel: base.label, ftLabel: fin.label });
    $("#rounds-table").innerHTML = `<div class="table-scroll"><table><thead><tr><th>Round</th><th class="num">Satisfaction</th><th class="num">95% CI</th><th class="num">Full pass</th><th class="num">Parse rate</th><th class="num">Seen scen.</th><th class="num">Held-out scen.</th></tr></thead><tbody>${rounds.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${pct(r.satisfaction_rate)}</td><td class="num">${pct(r.satisfaction_ci95[0])} – ${pct(r.satisfaction_ci95[1])}</td><td class="num">${pct(r.full_pass_rate)}</td><td class="num">${pct(r.parse_rate)}</td><td class="num">${pct(r.seen_satisfaction)}</td><td class="num">${pct(r.heldout_satisfaction)}</td></tr>`).join("")}</tbody></table></div>`;
    $("#spec-table tbody").innerHTML = d.specs.map((s) => `<tr><td>${SPEC_LABEL(s.id)}</td><td>${esc(s.formula)}</td><td>${esc(s.description)}</td></tr>`).join("");
  }

  // ---------------------------------------------------------- url + theme
  function writeHash() {
    history.replaceState(null, "", ReplayState.formatHash({ run: state.run.id, scenario: state.scenario, sample: state.sample, models: state.models }));
  }
  // Apply the hash state; invalid values fall back to scenario 1, sample 0 and
  // the run's first and last model (refresh() then rewrites a valid hash).
  function readHash() {
    const ms = state.data.models;
    const defaults = { scenario: state.data.scenarios[0].id, models: [ms[0].id, ms[ms.length - 1].id] };
    Object.assign(state, ReplayState.resolveHash(ReplayState.parseHash(location.hash), state.data, defaults));
  }
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("lt-theme"); } catch (e) { /* storage blocked */ }
    if (saved) document.documentElement.dataset.theme = saved;
    $("#theme-btn").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("lt-theme", next); } catch (e) { /* ignore */ }
      colors = null; renderResults(); state.lanes.forEach((L) => L.lane.draw());
    });
  }

  // The URL hash holds the replay state, so the header links scroll to their
  // section without replacing it.
  function scrollToSection(id) {
    const target = document.getElementById(id);
    if (!target) return false;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
    if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
    return true;
  }
  function initSectionLinks() {
    document.querySelectorAll('.links a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
      if (scrollToSection(a.getAttribute("href").slice(1))) e.preventDefault();
    }));
  }

  // Typing a new hash, or Back/Forward between hashes, re-applies it. A plain
  // section anchor scrolls to its section and the replay state is restored.
  function onHashChange() {
    const p = ReplayState.parseHash(location.hash);
    if (!p) {
      const id = ReplayState.sectionId(location.hash);
      if (id) scrollToSection(id);
      writeHash();
    } else if (p.r && p.r !== state.run.id && state.index.runs.some((r) => r.id === p.r)) {
      selectRun(p.r, true);
    } else {
      readHash();
      refresh();
    }
  }

  // ------------------------------------------------------------------- init
  let loadToken = 0;
  async function selectRun(runId, fromHash) {
    state.run = state.index.runs.find((r) => r.id === runId) || state.index.runs[0];
    $("#run").value = state.run.id;
    const token = ++loadToken;
    const data = await fetchJson(runDir() + "replay.json");
    if (token !== loadToken) return; // a newer selectRun superseded this load
    state.data = data;
    const ms = state.data.models;
    state.models = [ms[0].id, ms[ms.length - 1].id];
    if (fromHash) readHash();
    else { state.scenario = state.data.scenarios[0].id; state.sample = 0; }
    document.querySelectorAll("[data-file]").forEach((a) => { a.href = runDir() + a.dataset.file; });
    renderTiles();
    populateModelSelects();
    renderResults();
    refresh();
  }

  async function init() {
    initTheme();
    initSectionLinks();
    try {
      const { prefix, index } = await loadIndex();
      state.prefix = prefix; state.index = index;
    } catch (e) {
      $("#provenance").textContent = "Could not load results. Run `python main.py export --run-dir <run>` and serve the repository over HTTP.";
      return;
    }
    const runSel = $("#run");
    runSel.innerHTML = state.index.runs.map((r) => `<option value="${r.id}">${esc(r.base_model)} · ${pct(r.base_satisfaction)} → ${pct(r.final_satisfaction)}</option>`).join("");
    runSel.addEventListener("change", () => selectRun(runSel.value, false));
    const hashRun = (ReplayState.parseHash(location.hash) || {}).r;
    const scenSel = $("#scenario");
    buildLanes();
    $("#play").addEventListener("click", play);
    $("#reset").addEventListener("click", resetAll);
    const sp = $("#speed");
    sp.value = state.speed;
    sp.addEventListener("input", () => { state.speed = +sp.value; $("#speed-out").textContent = `${state.speed}×`; });
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { renderResults(); state.lanes.forEach((L) => L.lane.draw()); }, 150); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { colors = null; state.lanes.forEach((L) => L.lane.draw()); });
    scenSel.addEventListener("change", () => { state.scenario = +scenSel.value; state.sample = 0; refresh(); });
    await selectRun(hashRun || state.index.default, true);
    window.addEventListener("hashchange", onHashChange);
  }
  init();
})();
