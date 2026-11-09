/*
 * Replay state helpers (pure, no DOM): sample lookup and the shareable URL
 * hash (#r=run&s=scenario&i=sample&a=modelA&b=modelB). Values read from a
 * hash are validated against the loaded replay.json before they are used.
 */
(function (global) {
  "use strict";

  const STATE_KEYS = ["r", "s", "i", "a", "b"];

  // Recorded samples of one model for one scenario, in strip order.
  function samplesFor(data, modelId, scenarioId) {
    return (data.samples[modelId] || [])
      .filter((x) => x.scenario_id === scenarioId)
      .sort((a, b) => a.template - b.template || a.sample_idx - b.sample_idx);
  }

  // Raw state values from a hash, or null when it holds none (a section anchor).
  function parseHash(hash) {
    const p = new URLSearchParams(String(hash || "").replace(/^#/, ""));
    if (!STATE_KEYS.some((k) => p.has(k))) return null;
    return Object.fromEntries(STATE_KEYS.map((k) => [k, p.get(k)]));
  }

  const nonNegInt = (raw) => (raw != null && /^\d+$/.test(raw) ? Number(raw) : null);

  // Validated { scenario, sample, models }; anything invalid falls back to
  // defaults.scenario / defaults.models, and to sample 0.
  function resolveHash(params, data, defaults) {
    const out = { scenario: defaults.scenario, sample: 0, models: [...defaults.models] };
    if (!params) return out;
    const isModel = (id) => data.models.some((m) => m.id === id);
    if (isModel(params.a)) out.models[0] = params.a;
    if (isModel(params.b)) out.models[1] = params.b;
    const s = nonNegInt(params.s);
    if (s !== null && data.scenarios.some((x) => x.id === s)) out.scenario = s;
    const count = Math.min(...out.models.map((m) => samplesFor(data, m, out.scenario).length));
    const i = nonNegInt(params.i);
    out.sample = i !== null && i < count ? i : 0;
    return out;
  }

  function formatHash(st) {
    return `#r=${st.run}&s=${st.scenario}&i=${st.sample}&a=${st.models[0]}&b=${st.models[1]}`;
  }

  const api = { samplesFor, parseHash, resolveHash, formatHash };
  global.LogicTuneState = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api; // Node tests
})(typeof window !== "undefined" ? window : globalThis);
