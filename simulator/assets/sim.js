/*
 * LogicTune intersection simulator.
 *
 * A Lane owns one canvas and replays one recorded controller (the rules that
 * GLM2FSA parsed from a model response). Every tick:
 *   1. the scenario timeline updates the environment propositions,
 *   2. the controller fires the first rule whose guard holds (or none),
 *   3. the car moves according to the action,
 *   4. a runtime monitor checks the state-action invariants of the LTL suite.
 * Two lanes that share a scenario see exactly the same environment timeline.
 */
(function (global) {
  "use strict";

  const PROPS = [
    "green_traffic_light", "green_left_turn_light", "opposite_car", "car_from_left", "car_from_right",
    "pedestrian_at_left", "pedestrian_at_right", "pedestrian_in_front", "side_car", "stop_sign",
  ];

  // Environment changes over time (ticks at 1x speed, 60 ticks per second).
  // The model checker assumes a fair environment: infinitely often the light
  // is green and the intersection is clear (environment.py, FAIRNESS). Every
  // timeline therefore ends in that state. Scenarios 10, 11, 12 and 16 have no
  // traffic light (green_traffic_light is false, "not green"); it turns true
  // at 6 s, after cross traffic has cleared, as fairness requires.
  const FAIR_GREEN = { t: 360, set: { green_traffic_light: true } };
  const TIMELINE = {
    1: [{ t: 240, set: { pedestrian_in_front: false } }],
    2: [{ t: 240, set: { pedestrian_at_right: false } }],
    3: [{ t: 240, set: { pedestrian_at_left: false } }],
    4: [{ t: 300, set: { opposite_car: false } }],
    5: [{ t: 300, set: { car_from_left: false } }],
    8: [{ t: 300, set: { green_traffic_light: true } }],
    9: [{ t: 300, set: { car_from_left: false } }],
    10: [{ t: 300, set: { car_from_left: false } }, FAIR_GREEN],
    11: [FAIR_GREEN],
    12: [{ t: 240, set: { car_from_left: false } }, FAIR_GREEN],
    13: [{ t: 300, set: { green_traffic_light: true } }],
    15: [{ t: 300, set: { side_car: false } }],
    16: [{ t: 300, set: { car_from_right: false } }, FAIR_GREEN],
  };
  const EPISODE_TICKS = 900;
  const STOP_SIGN_HOLD = 45; // ticks stopped at the line before the sign counts as obeyed

  // Invariant (state-action) specifications checked at run time.
  const MONITORS = [
    ["phi_2", (v, a) => v.opposite_car && !v.green_left_turn_light && a === "turn_left", "unprotected left turn with an oncoming car"],
    ["phi_3", (v, a) => !v.green_traffic_light && a === "go_straight", "go straight without a green light"],
    ["phi_5", (v, a) => (v.car_from_left || v.pedestrian_at_right) && a === "turn_right", "right turn with a car from the left or a pedestrian on the right"],
    ["phi_9", (v, a) => v.car_from_left && (a === "turn_left" || a === "turn_right"), "turn while a car comes from the left"],
    ["phi_11", (v, a) => a === "turn_right" && !v.green_traffic_light && v.car_from_left, "right turn on non-green with a car from the left"],
    ["phi_12", (v, a) => a === "turn_left" && !v.green_left_turn_light && (v.car_from_right || v.car_from_left || v.opposite_car), "unprotected left turn with cross or oncoming traffic"],
    ["phi_14", (v, a) => a === "go_straight" && v.pedestrian_in_front, "go straight into a pedestrian"],
    ["phi_15", (v, a) => a === "turn_right" && v.stop_sign && v.car_from_left, "right turn at a stop sign with a car from the left"],
    ["phi_6", (v, a) => a === null, "no rule applies (action undefined)"],
  ];

  function evalGuard(g, vars) {
    switch (g.op) {
      case "const": return !!g.value;
      case "var": return PROPS.includes(g.name) ? !!vars[g.name] : false; // free atoms: false in simulation
      case "not": return !evalGuard(g.arg, vars);
      case "and": return g.args.every((a) => evalGuard(a, vars));
      case "or": return g.args.some((a) => evalGuard(a, vars));
      default: return false;
    }
  }

  function decide(rules, vars) {
    for (let i = 0; i < rules.length; i++) {
      if (evalGuard(rules[i].guard, vars)) return { action: rules[i].action, rule: i };
    }
    return { action: null, rule: -1 };
  }

  // World geometry in logical units (the canvas is scaled to fit).
  const W = 600, C = 300, ROAD = 120, LANE = 30;
  const EGO_X = C + LANE, STOP_Y = C + ROAD / 2 + 50, START_Y = STOP_Y + 24; // car waits at the stop line
  const R_LEFT = 60, R_RIGHT = 30;
  const MERGE_X = C + ROAD / 2 + LANE; // centre of the extra lane drawn when a side car is present

  // Other road users. Each one is present while its proposition is true. When
  // the timeline clears the proposition, the actor has just driven (or walked)
  // out of the area it blocks: it starts moving `clear / speed` ticks earlier,
  // so it reaches that point exactly at the event time, and it keeps going
  // until it is off the canvas. Positions depend only on the shared timeline,
  // so both lanes always show the same actors in the same places.
  const ACTORS = {
    // `clear`: distance travelled until the actor no longer blocks the ego.
    opposite_car: { kind: "car", x: C - LANE, y: C - ROAD / 2 - 70, heading: Math.PI / 2, speed: 2, clear: 70 + ROAD + 30 },
    car_from_left: { kind: "car", x: C - ROAD / 2 - 60, y: C + LANE, heading: 0, speed: 2, clear: 60 + ROAD + 30 },
    car_from_right: { kind: "car", x: C + ROAD / 2 + 60, y: C - LANE, heading: Math.PI, speed: 2, clear: 60 + ROAD + 30 },
    pedestrian_in_front: { kind: "person", x: C + LANE, y: C + ROAD / 2 + 27, heading: 0, speed: 0.6, clear: LANE + 12 },
    pedestrian_at_right: { kind: "person", x: C + ROAD / 2 + 27, y: C + LANE, heading: Math.PI / 2, speed: 0.6, clear: LANE + 12 },
    pedestrian_at_left: { kind: "person", x: C - ROAD / 2 - 27, y: C - LANE, heading: -Math.PI / 2, speed: 0.6, clear: LANE + 12 },
    // Scenario 15 adds a second northbound lane right of the ego (a right-turn
    // lane, drawn only there). The side car waits level with the ego, then
    // turns right onto the east road; it is clear once the turn is complete.
    side_car: {
      kind: "car", x: MERGE_X, y: START_Y, heading: -Math.PI / 2, speed: 1.6,
      turn: { after: START_Y - (C + ROAD / 2), radius: LANE }, clear: START_Y - (C + ROAD / 2) + (LANE * Math.PI) / 2,
    },
  };
  const OFF_CANVAS = 30; // an actor this far outside the canvas is gone

  // Actors visible at tick t of a scenario: [{ prop, kind, x, y, heading }].
  function actorsAt(scenarioId, init, t) {
    const out = [];
    for (const [prop, a] of Object.entries(ACTORS)) {
      if (!(init && init[prop])) continue;
      const ev = (TIMELINE[scenarioId] || []).find((e) => e.set[prop] === false);
      const leaveAt = ev ? ev.t - a.clear / a.speed : Infinity;
      const p = poseAlong(a, Math.max(0, t - leaveAt) * a.speed);
      if (p.x < -OFF_CANVAS || p.x > W + OFF_CANVAS || p.y < -OFF_CANVAS || p.y > W + OFF_CANVAS) continue;
      out.push({ prop, kind: a.kind, x: round(p.x), y: round(p.y), heading: p.heading });
    }
    return out;
  }

  // Pose after travelling d units: straight ahead, or straight then a 90° right turn.
  function poseAlong(a, d) {
    const line = (x, y, h, s) => ({ x: x + Math.cos(h) * s, y: y + Math.sin(h) * s, heading: h });
    if (!a.turn || d <= a.turn.after) return line(a.x, a.y, a.heading, d);
    const p = line(a.x, a.y, a.heading, a.turn.after), r = a.turn.radius;
    const cx = p.x + r * Math.cos(p.heading + Math.PI / 2), cy = p.y + r * Math.sin(p.heading + Math.PI / 2);
    const theta = Math.min((d - a.turn.after) / r, Math.PI / 2), phi = p.heading - Math.PI / 2 + theta;
    const q = { x: cx + r * Math.cos(phi), y: cy + r * Math.sin(phi), heading: p.heading + theta };
    return line(q.x, q.y, q.heading, Math.max(0, d - a.turn.after - (r * Math.PI) / 2));
  }
  const round = (x) => Math.round(x * 1e6) / 1e6; // keep exact lane coordinates exact

  class Lane {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas ? canvas.getContext("2d") : null; // null: headless (tests)
      this.onTick = opts.onTick || (() => {});
      this.colors = opts.colors;
      this.load(null, [], 1);
    }

    load(scenario, rules, scenarioId) {
      this.scenario = scenario;
      this.scenarioId = scenarioId;
      this.rules = rules || [];
      this.reset();
    }

    reset() {
      const init = (this.scenario && this.scenario.init) || {};
      this.vars = Object.fromEntries(PROPS.map((p) => [p, !!init[p]]));
      this.t = 0;
      this.car = { x: EGO_X, y: START_Y, heading: -Math.PI / 2, speed: 0, mode: "approach", target: null, turnLeft: 0 };
      this.events = (TIMELINE[this.scenarioId] || []).map((e) => ({ t: e.t, set: { ...e.set }, fired: false }));
      this.stoppedAtLine = 0;
      this.signServed = false;
      this.decision = decide(this.rules, this.vars);
      this.violations = [];
      this.seen = new Set();
      this.done = false; // the car's outcome is decided (the environment may still run)
      this.doneAt = null; // seconds
      this.outcome = null;
      this.flash = 0;
      this.draw();
    }

    // Other road users at the current tick (identical in every lane of a scenario).
    actors() { return actorsAt(this.scenarioId, this.scenario && this.scenario.init, this.t); }

    // The extra right-hand lane exists only in scenarios with a side car.
    hasMergeLane() { return !!(this.scenario && this.scenario.init && this.scenario.init.side_car); }

    // The environment (time, timeline, other road users) runs for the whole
    // episode, so every lane keeps showing the shared timeline. The car, the
    // controller and the monitor stop once the car's outcome is decided.
    step(dt) {
      if (this.t >= EPISODE_TICKS) return;
      this.t = Math.min(EPISODE_TICKS, this.t + dt);
      for (const ev of this.events) {
        if (!ev.fired && this.t >= ev.t) { Object.assign(this.vars, ev.set); ev.fired = true; }
      }
      if (!this.done) this.stepCar(dt);
      if (this.flash > 0) this.flash -= dt;
      this.onTick(this);
    }

    stepCar(dt) {
      // A stop sign is obeyed once the car has waited at the line.
      if (this.vars.stop_sign && !this.signServed) {
        const atLine = this.car.y >= STOP_Y && this.car.y <= STOP_Y + 60 && this.car.speed < 0.05;
        this.stoppedAtLine = atLine ? this.stoppedAtLine + dt : 0;
        if (this.stoppedAtLine >= STOP_SIGN_HOLD) { this.vars.stop_sign = false; this.signServed = true; }
      }
      this.decision = decide(this.rules, this.vars);
      const a = this.decision.action;
      if (this.car.mode !== "exit") {
        for (const [id, bad, why] of MONITORS) {
          if (bad(this.vars, a)) {
            this.flash = 20;
            if (!this.seen.has(id)) {
              this.seen.add(id);
              this.violations.push({ id, t: this.t / 60, why });
            }
          }
        }
      }
      this.move(a, dt);
      if (this.car.x < -40 || this.car.x > W + 40 || this.car.y < -40) {
        this.finish("crossed");
      } else if (this.t >= EPISODE_TICKS) {
        this.finish(this.car.mode === "approach" ? "waiting" : "in intersection");
      }
    }

    finish(outcome) {
      this.done = true;
      this.outcome = outcome;
      this.doneAt = this.t / 60;
    }

    move(action, dt) {
      const c = this.car;
      const committed = c.mode === "turning" || c.mode === "exit";
      let want = 0;
      if (action === "go_straight") want = 2.2;
      else if (action === "turn_left" || action === "turn_right") want = 1.6;
      if (committed && c.mode === "exit" && action !== "stop") want = Math.max(want, 1.8);
      if (action === null || action === "stop") want = 0;
      const acc = want > c.speed ? 0.06 : 0.14;
      c.speed += Math.sign(want - c.speed) * Math.min(Math.abs(want - c.speed), acc * dt);

      if (c.mode === "approach") {
        // Arc starts so the car ends in the correct lane of the crossing road.
        if (action === "turn_left" && c.y <= C - LANE + R_LEFT) { c.mode = "turning"; c.target = "left"; c.turnLeft = Math.PI / 2; }
        else if (action === "turn_right" && c.y <= C + LANE + R_RIGHT) { c.mode = "turning"; c.target = "right"; c.turnLeft = Math.PI / 2; }
        else if (c.y < C - ROAD / 2) c.mode = "exit";
      }
      if (c.mode === "turning") {
        const r = c.target === "left" ? R_LEFT : R_RIGHT;
        const dTheta = Math.min(c.turnLeft, (c.speed * dt) / r);
        c.heading += c.target === "left" ? -dTheta : dTheta;
        c.turnLeft -= dTheta;
        if (c.turnLeft <= 1e-6) c.mode = "exit";
      }
      c.x += Math.cos(c.heading) * c.speed * dt;
      c.y += Math.sin(c.heading) * c.speed * dt;
    }

    // ---------------------------------------------------------------- drawing
    draw() {
      if (!this.ctx) return;
      const cv = this.canvas, ctx = this.ctx, k = this.colors();
      const dpr = global.devicePixelRatio || 1;
      const size = Math.min(cv.clientWidth || W, 640);
      if (cv.width !== Math.round(size * dpr)) { cv.width = Math.round(size * dpr); cv.height = Math.round(size * dpr); }
      ctx.setTransform((cv.width / W), 0, 0, (cv.height / W), 0, 0);
      ctx.fillStyle = k.ground; ctx.fillRect(0, 0, W, W);
      // roads
      ctx.fillStyle = k.road;
      ctx.fillRect(0, C - ROAD / 2, W, ROAD);
      ctx.fillRect(C - ROAD / 2, 0, ROAD, W);
      const merge = this.hasMergeLane();
      if (merge) ctx.fillRect(C + ROAD / 2 - 1, C + ROAD / 2 - 1, 2 * LANE + 1, W - C - ROAD / 2 + 1); // overlap: no seam
      ctx.strokeStyle = k.marking; ctx.lineWidth = 2; ctx.setLineDash([14, 14]);
      ctx.beginPath(); ctx.moveTo(0, C); ctx.lineTo(C - ROAD / 2, C); ctx.moveTo(C + ROAD / 2, C); ctx.lineTo(W, C);
      ctx.moveTo(C, 0); ctx.lineTo(C, C - ROAD / 2); ctx.moveTo(C, C + ROAD / 2); ctx.lineTo(C, W);
      if (merge) { ctx.moveTo(C + ROAD / 2, STOP_Y); ctx.lineTo(C + ROAD / 2, W); }
      ctx.stroke();
      ctx.setLineDash([]);
      // crosswalks
      ctx.fillStyle = k.crosswalk;
      for (let i = 0; i < 6; i++) {
        ctx.fillRect(C - ROAD / 2 + 4 + i * 20, C + ROAD / 2 + 14, 12, 26);
        ctx.fillRect(C - ROAD / 2 + 4 + i * 20, C - ROAD / 2 - 40, 12, 26);
        ctx.fillRect(C - ROAD / 2 - 40, C - ROAD / 2 + 4 + i * 20, 26, 12);
        ctx.fillRect(C + ROAD / 2 + 14, C - ROAD / 2 + 4 + i * 20, 26, 12);
        if (merge && i < 3) ctx.fillRect(C + ROAD / 2 + 4 + i * 20, C + ROAD / 2 + 14, 12, 26);
      }
      // stop line
      ctx.fillStyle = k.marking; ctx.fillRect(C, STOP_Y - 2, ROAD / 2 + (merge ? 2 * LANE : 0), 4);
      this.drawSignals(ctx, k);
      this.drawActors(ctx, k);
      this.drawCar(ctx, this.car.x, this.car.y, this.car.heading, k.ego, k.egoGlass);
      if (this.flash > 0) {
        ctx.strokeStyle = k.critical; ctx.lineWidth = 6; ctx.globalAlpha = Math.min(1, this.flash / 20);
        ctx.strokeRect(3, 3, W - 6, W - 6); ctx.globalAlpha = 1;
      }
    }

    drawSignals(ctx, k) {
      const shift = this.hasMergeLane() ? 2 * LANE : 0; // keep the signal off the extra lane
      const v = this.vars, x = C + ROAD / 2 + 46 + shift, y = STOP_Y + 26;
      if (this.scenarioId !== 16) {
        ctx.fillStyle = k.signalBox; roundRect(ctx, x, y, 30, 74, 6); ctx.fill();
        lamp(ctx, x + 15, y + 15, !v.green_traffic_light ? k.red : k.lampOff);
        lamp(ctx, x + 15, y + 37, k.lampOff);
        lamp(ctx, x + 15, y + 59, v.green_traffic_light ? k.green : k.lampOff);
        if (v.green_left_turn_light) {
          ctx.fillStyle = k.green; ctx.font = "bold 22px system-ui, sans-serif"; ctx.textAlign = "center";
          ctx.fillText("←", x - 16, y + 66);
        }
      }
      if (v.stop_sign || this.signServed) {
        const sx = C + ROAD / 2 + 22 + shift, sy = STOP_Y + 6, r = 15;
        ctx.globalAlpha = v.stop_sign ? 1 : 0.35;
        ctx.fillStyle = k.red; ctx.beginPath();
        for (let i = 0; i < 8; i++) {
          const a = Math.PI / 8 + (i * Math.PI) / 4;
          ctx.lineTo(sx + r * Math.cos(a), sy + r * Math.sin(a));
        }
        ctx.closePath(); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = "bold 8px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("STOP", sx, sy + 0.5); ctx.globalAlpha = 1; ctx.textBaseline = "alphabetic";
      }
    }

    drawActors(ctx, k) {
      for (const a of this.actors()) {
        if (a.kind === "person") person(ctx, a.x, a.y, k);
        else this.drawCar(ctx, a.x, a.y, a.heading, k.other, k.egoGlass);
      }
    }

    drawCar(ctx, x, y, heading, body, glass) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(heading + Math.PI / 2);
      ctx.fillStyle = body; roundRect(ctx, -11, -20, 22, 40, 5); ctx.fill();
      ctx.fillStyle = glass; roundRect(ctx, -8, -13, 16, 9, 2); ctx.fill();
      ctx.restore();
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function lamp(ctx, x, y, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill(); }
  function person(ctx, x, y, k) {
    ctx.fillStyle = k.pedestrian; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = k.ground; ctx.lineWidth = 2; ctx.stroke();
  }

  const api = { Lane, decide, evalGuard, actorsAt, PROPS, MONITORS, EPISODE_TICKS };
  global.LogicTuneSim = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api; // Node tests
})(typeof window !== "undefined" ? window : globalThis);
