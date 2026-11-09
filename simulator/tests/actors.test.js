"use strict";
// Other road users: where they are drawn over the shared scenario timeline.
const test = require("node:test");
const assert = require("node:assert/strict");
const { sim, data, scenario } = require("./helpers.js");

const W = 600, BOX = [240, 360]; // canvas size and intersection box (logical units)
const actor = (sid, prop, t) => sim.actorsAt(sid, scenario(sid).init, t).find((a) => a.prop === prop);
const onCanvas = (a) => a.x > -22 && a.x < W + 22 && a.y > -22 && a.y < W + 22; // some of the body is visible

test("the oncoming car in scenario 4 waits above the intersection in the southbound lane", () => {
  const a = actor(4, "opposite_car", 0);
  assert.equal(a.kind, "car");
  assert.equal(a.x, 270);
  assert.ok(a.y < BOX[0] - 20, `y=${a.y} should be above the box`);
  assert.equal(a.heading, Math.PI / 2); // facing south
});

test("the oncoming car drives straight through and is past the box when opposite_car turns false", () => {
  const before = actor(4, "opposite_car", 150), atClear = actor(4, "opposite_car", 300);
  assert.equal(before.x, 270);
  assert.equal(atClear.x, 270);
  assert.ok(atClear.y - 20 > BOX[1], `body (y=${atClear.y}) must have left the box at t = 300`);
  assert.ok(onCanvas(atClear), "it is still visible, driving away, not vanished");
});

test("cross traffic drives through in its own lane: from the left eastbound, from the right westbound", () => {
  const l0 = actor(5, "car_from_left", 0), l1 = actor(5, "car_from_left", 300);
  assert.equal(l0.y, 330); assert.equal(l1.y, 330);
  assert.equal(l0.heading, 0);
  assert.ok(l0.x < BOX[0] - 20 && l1.x - 20 > BOX[1], `${l0.x} -> ${l1.x}`);
  const r0 = actor(16, "car_from_right", 0), r1 = actor(16, "car_from_right", 300);
  assert.equal(r0.y, 270); assert.equal(r1.y, 270);
  assert.equal(r0.heading, Math.PI);
  assert.ok(r0.x > BOX[1] + 20 && r1.x + 20 < BOX[0], `${r0.x} -> ${r1.x}`);
});

test("pedestrians walk along their crosswalk and are off the road when they clear", () => {
  const f0 = actor(1, "pedestrian_in_front", 0), f1 = actor(1, "pedestrian_in_front", 240);
  assert.equal(f0.kind, "person");
  assert.equal(f1.y, f0.y);
  assert.ok(f1.x - 8 > BOX[1], `in-front pedestrian at x=${f1.x} has left the ego's lane`);
  const r0 = actor(2, "pedestrian_at_right", 0), r1 = actor(2, "pedestrian_at_right", 240);
  assert.equal(r1.x, r0.x);
  assert.ok(r1.y - 8 > BOX[1], `right pedestrian at y=${r1.y} has left the east road`);
  const l0 = actor(3, "pedestrian_at_left", 0), l1 = actor(3, "pedestrian_at_left", 240);
  assert.equal(l1.x, l0.x);
  assert.ok(l1.y + 8 < BOX[0], `left pedestrian at y=${l1.y} has left the west road`);
});

test("no actor vanishes on the canvas: each moves continuously until it is off-screen", () => {
  for (const s of data.scenarios) {
    for (const prop of Object.keys(s.init).filter((p) => s.init[p] && sim.actorsAt(s.id, s.init, 0).some((a) => a.prop === p))) {
      let prev = actor(s.id, prop, 0);
      for (let t = 1; t <= sim.EPISODE_TICKS; t++) {
        const cur = actor(s.id, prop, t);
        if (!cur) { assert.ok(!onCanvas(prev), `s${s.id} ${prop} vanished at t=${t} on canvas (${prev.x}, ${prev.y})`); break; }
        assert.ok(Math.hypot(cur.x - prev.x, cur.y - prev.y) <= 3, `s${s.id} ${prop} jumped at t=${t}`);
        prev = cur;
      }
    }
  }
});

test("scenarios without other road users draw none", () => {
  assert.deepEqual(sim.actorsAt(7, scenario(7).init, 0), []);
});
