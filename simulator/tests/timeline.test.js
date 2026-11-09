"use strict";
// The simulated environment must agree with the model checker's environment
// (src/logictune/environment.py, exported into replay.json).
const test = require("node:test");
const assert = require("node:assert/strict");
const { sim, data, loadReplay, lane, rule, T } = require("./helpers.js");

test("the simulator observes the same propositions as the model checker", () => {
  assert.deepEqual(sim.PROPS, data.environment.propositions);
});

// fairness.smv is a conjunction like "green_traffic_light & !opposite_car & ...".
const fairness = data.environment.fairness.smv.split("&").map((t) => t.trim())
  .map((t) => (t.startsWith("!") ? [t.slice(1), false] : [t, true]));

test("every scenario reaches the fair state (green light, intersection clear) for a car waiting at the line", () => {
  for (const s of data.scenarios) {
    const l = lane(s.id, [rule(T, "stop")]);
    for (let i = 0; i < 420; i++) l.step(1);
    for (const [prop, value] of fairness) {
      assert.equal(l.vars[prop], value, `scenario ${s.id}: ${prop} should be ${value} by 7 s`);
    }
  }
});

test("controllers that satisfy all 15 specifications never end the episode waiting at the line", () => {
  for (const run of ["qwen2.5-3b", "tinyllama-1.1b"]) {
    const d = loadReplay(run);
    for (const [model, samples] of Object.entries(d.samples)) {
      for (const x of samples.filter((x) => x.score === 15)) {
        const l = new sim.Lane(null, {});
        l.load(d.scenarios.find((s) => s.id === x.scenario_id), x.rules, x.scenario_id);
        while (l.t < sim.EPISODE_TICKS) l.step(1);
        assert.notEqual(l.outcome, "waiting", `${run} ${model} s${x.scenario_id} template ${x.template} sample ${x.sample_idx}`);
      }
    }
  }
});

test("only the stop-sign and uncontrolled scenarios (10, 11, 12, 16) have no traffic light", () => {
  const noLight = data.scenarios.filter((s) => !sim.hasTrafficLight(s.id)).map((s) => s.id);
  assert.deepEqual(noLight, [10, 11, 12, 16]);
  for (const id of noLight) {
    assert.ok(!data.scenarios.find((s) => s.id === id).init.green_traffic_light, `scenario ${id} starts not green`);
  }
});

test("without a traffic light, green_traffic_light turns true at the fairness time", () => {
  const l = lane(16, [rule(T, "stop")]);
  while (l.t < sim.FAIR_GREEN_TICK - 1) l.step(1);
  assert.equal(l.vars.green_traffic_light, false);
  l.step(1);
  assert.equal(l.vars.green_traffic_light, true);
});
