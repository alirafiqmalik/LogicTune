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

// A headless lane replaying one recorded sample of replay data d.
function replayLane(x, d = data) {
  const l = new sim.Lane(null, {});
  l.load(d.scenarios.find((s) => s.id === x.scenario_id), x.rules, x.scenario_id);
  return l;
}

// A headless lane replaying recorded sample i (the strip order: template, then sample index).
function recorded(model, scenarioId, i, d = data) {
  return replayLane(d.samples[model].filter((s) => s.scenario_id === scenarioId)
    .sort((a, b) => a.template - b.template || a.sample_idx - b.sample_idx)[i], d);
}

// Play a lane to the end of the episode with a given playback speed (ticks per frame).
function playOut(l, speed = 1) {
  while (l.t < sim.EPISODE_TICKS) l.step(speed);
  return l;
}

module.exports = { sim, data, scenario, loadReplay, lane, replayLane, recorded, playOut, v, not, and, T, rule, ROOT };
