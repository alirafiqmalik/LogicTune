// Shared helpers for the simulator tests (node --test, no dependencies).
"use strict";
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..", "..");
const sim = require("../assets/sim.js");

function loadReplay(run = "qwen2.5-3b") {
  return JSON.parse(fs.readFileSync(path.join(ROOT, "results", "latest", run, "replay.json"), "utf8"));
}

const data = loadReplay();
const scenario = (id) => data.scenarios.find((s) => s.id === id);

// Guard AST helpers, in the same shape as replay.json rules.
const v = (name) => ({ op: "var", name });
const not = (arg) => ({ op: "not", arg });
const and = (...args) => ({ op: "and", args });
const T = { op: "const", value: true };
const rule = (guard, action) => ({ guard, action });

// A headless lane loaded with a scenario from the recorded data.
function lane(scenarioId, rules) {
  const l = new sim.Lane(null, {});
  l.load(scenario(scenarioId), rules, scenarioId);
  return l;
}

module.exports = { sim, data, scenario, loadReplay, lane, v, not, and, T, rule, ROOT };
