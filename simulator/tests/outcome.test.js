"use strict";
// How an episode ends, and the label shown for it.
const test = require("node:test");
const assert = require("node:assert/strict");
const { sim, recorded, playOut } = require("./helpers.js");

const car = (x, y, mode) => ({ x, y, mode });

test("the outcome is classified by where the car is", () => {
  assert.equal(sim.classifyOutcome(car(330, 434, "approach")), "waiting");
  assert.equal(sim.classifyOutcome(car(300, 300, "turning")), "in intersection");
  assert.equal(sim.classifyOutcome(car(330, 250, "exit")), "in intersection");
  assert.equal(sim.classifyOutcome(car(76, 270, "exit")), "exit road");
  assert.equal(sim.classifyOutcome(car(330, 150, "exit")), "exit road");
});

test("a car whose body has left the canvas has crossed", () => {
  assert.equal(sim.classifyOutcome(car(-32, 270, "exit")), "crossed");
  assert.equal(sim.classifyOutcome(car(625, 330, "exit")), "crossed");
  assert.equal(sim.classifyOutcome(car(330, -21, "exit")), "crossed");
  assert.equal(sim.classifyOutcome(car(-15, 270, "exit")), "exit road", "still partly visible");
});

test("each outcome has a label that matches it", () => {
  assert.equal(sim.outcomeText("crossed", 4.13), "Crossed the intersection at 4.1 s");
  assert.equal(sim.outcomeText("waiting"), "Still waiting at the stop line");
  assert.equal(sim.outcomeText("in intersection"), "Stopped inside the intersection");
  assert.equal(sim.outcomeText("exit road"), "Stopped after the intersection");
});

test("recorded controllers that stop after the turn are not labelled 'inside the intersection'", () => {
  // qwen round0, scenario 3 sample 5: turns left, then no rule applies and it stops on the west road.
  const l = playOut(recorded("round0", 3, 4));
  assert.equal(l.outcome, "exit road");
  assert.ok(l.car.x < 240 - 20, `x = ${l.car.x}`);
});

test("the outcome does not depend on playback speed", () => {
  for (const [model, s, i] of [["round0", 2, 12], ["round0", 12, 2], ["round0", 12, 12], ["round0", 12, 13], ["round0", 5, 1], ["round3", 4, 0]]) {
    const slow = playOut(recorded(model, s, i), 0.25), fast = playOut(recorded(model, s, i), 4), mid = playOut(recorded(model, s, i), 1.5);
    assert.equal(slow.outcome, fast.outcome, `${model} s${s} i${i}`);
    assert.equal(mid.outcome, fast.outcome, `${model} s${s} i${i}`);
    assert.deepEqual(slow.car, fast.car, `${model} s${s} i${i}`);
    assert.deepEqual(slow.violations, mid.violations, `${model} s${s} i${i}`);
  }
});
