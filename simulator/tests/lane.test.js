"use strict";
// One lane's episode: shared timeline, controller, car, outcome.
const test = require("node:test");
const assert = require("node:assert/strict");
const { sim, lane, rule, T } = require("./helpers.js");

const run = (l, ticks) => { for (let i = 0; i < ticks; i++) l.step(1); return l; };
const untilDone = (l) => { while (!l.done) l.step(1); return l; };

test("a lane whose car has crossed keeps showing the shared environment timeline", () => {
  const gone = untilDone(lane(4, [rule(T, "go_straight")])); // drives straight through at once
  const waiting = lane(4, [rule(T, "stop")]);
  assert.equal(gone.outcome, "crossed");
  assert.ok(gone.t < 300, "crossed before the oncoming car clears");
  const crossedAt = gone.t;
  run(waiting, crossedAt);
  run(gone, 360 - crossedAt); run(waiting, 360 - crossedAt);
  assert.equal(gone.t, 360);
  assert.equal(gone.vars.opposite_car, false, "the timeline event still fires after the car has left");
  assert.deepEqual(gone.vars, waiting.vars);
  assert.deepEqual(gone.actors(), waiting.actors());
});

test("after crossing, the car, decision and monitor are frozen and the crossing time is kept", () => {
  const l = untilDone(lane(4, [rule(T, "go_straight")]));
  const car = { ...l.car }, decision = l.decision, crossedAt = l.doneAt;
  assert.ok(crossedAt <= l.t / 60 && crossedAt > (l.t - 1) / 60, "set during the last step");
  run(l, 200);
  assert.deepEqual(l.car, car);
  assert.equal(l.decision, decision);
  assert.equal(l.doneAt, crossedAt);
});

test("time stops at the end of the episode", () => {
  const l = run(lane(13, [rule(T, "stop")]), sim.EPISODE_TICKS + 50);
  assert.equal(l.t, sim.EPISODE_TICKS);
  assert.equal(l.outcome, "waiting");
});
