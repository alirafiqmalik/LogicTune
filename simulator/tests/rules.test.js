"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { sim, v, not, and, T, rule } = require("./helpers.js");

test("the first rule whose guard holds sets the action", () => {
  const rules = [rule(v("pedestrian_in_front"), "stop"), rule(T, "go_straight")];
  assert.deepEqual(sim.decide(rules, { pedestrian_in_front: true }), { action: "stop", rule: 0 });
  assert.deepEqual(sim.decide(rules, { pedestrian_in_front: false }), { action: "go_straight", rule: 1 });
});

test("no matching rule leaves the action undefined", () => {
  const rules = [rule(and(v("green_traffic_light"), not(v("opposite_car"))), "turn_left")];
  assert.deepEqual(sim.decide(rules, { green_traffic_light: false }), { action: null, rule: -1 });
});

test("free atoms (unknown propositions) evaluate to false", () => {
  assert.equal(sim.evalGuard(v("ambulance_nearby"), { ambulance_nearby: true }), false);
});
