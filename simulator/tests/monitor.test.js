"use strict";
// Runtime monitor: state-action invariants checked on every tick of the car's run.
const test = require("node:test");
const assert = require("node:assert/strict");
const { lane, rule, v } = require("./helpers.js");

test("an undefined action after the car has turned is reported as a φ6 violation", () => {
  // Turns left while the pedestrian is there; no rule applies once the pedestrian clears at 4 s.
  const l = lane(3, [rule(v("pedestrian_at_left"), "turn_left")]);
  for (let i = 0; i < 239; i++) l.step(1); // the pedestrian clears on tick 240
  assert.equal(l.car.mode, "exit", "the turn is complete before the pedestrian clears");
  assert.deepEqual(l.violations, []);
  l.step(1);
  assert.equal(l.decision.action, null);
  assert.deepEqual(l.violations.map((x) => [x.id, x.t.toFixed(1)]), [["phi_6", "4.0"]]);
});

test("a violation is reported once, at its first tick", () => {
  const l = lane(1, [rule(v("pedestrian_in_front"), "go_straight")]); // φ14 from the first tick
  for (let i = 0; i < 30; i++) l.step(1);
  assert.deepEqual(l.violations.map((x) => [x.id, x.t]), [["phi_14", 1 / 60]]);
});
