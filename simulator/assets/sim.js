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
  const TIMELINE = {
    1: [{ t: 240, set: { pedestrian_in_front: false } }],
    2: [{ t: 240, set: { pedestrian_at_right: false } }],
    3: [{ t: 240, set: { pedestrian_at_left: false } }],
    4: [{ t: 300, set: { opposite_car: false } }],
    5: [{ t: 300, set: { car_from_left: false } }],
    8: [{ t: 300, set: { green_traffic_light: true } }],
    9: [{ t: 300, set: { car_from_left: false } }],
    10: [{ t: 300, set: { car_from_left: false } }],
    12: [{ t: 240, set: { car_from_left: false } }],
    13: [{ t: 300, set: { green_traffic_light: true } }],
    15: [{ t: 300, set: { side_car: false } }],
    16: [{ t: 300, set: { car_from_right: false } }],
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

  class Lane {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
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
      this.done = false;
      this.outcome = null;
      this.flash = 0;
      this.draw();
    }

    step(dt) {
      if (this.done) return;
      this.t += dt;
      for (const ev of this.events) {
        if (!ev.fired && this.t >= ev.t) { Object.assign(this.vars, ev.set); ev.fired = true; }
      }
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
      if (this.flash > 0) this.flash -= dt;
      if (this.car.x < -40 || this.car.x > W + 40 || this.car.y < -40) {
        this.done = true;
        this.outcome = "crossed";
      } else if (this.t >= EPISODE_TICKS) {
        this.done = true;
        this.outcome = this.car.mode === "approach" ? "waiting" : "in intersection";
      }
      this.onTick(this);
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
      ctx.strokeStyle = k.marking; ctx.lineWidth = 2; ctx.setLineDash([14, 14]);
      ctx.beginPath(); ctx.moveTo(0, C); ctx.lineTo(C - ROAD / 2, C); ctx.moveTo(C + ROAD / 2, C); ctx.lineTo(W, C);
      ctx.moveTo(C, 0); ctx.lineTo(C, C - ROAD / 2); ctx.moveTo(C, C + ROAD / 2); ctx.lineTo(C, W); ctx.stroke();
      ctx.setLineDash([]);
      // crosswalks
      ctx.fillStyle = k.crosswalk;
      for (let i = 0; i < 6; i++) {
        ctx.fillRect(C - ROAD / 2 + 4 + i * 20, C + ROAD / 2 + 14, 12, 26);
        ctx.fillRect(C - ROAD / 2 + 4 + i * 20, C - ROAD / 2 - 40, 12, 26);
        ctx.fillRect(C - ROAD / 2 - 40, C - ROAD / 2 + 4 + i * 20, 26, 12);
        ctx.fillRect(C + ROAD / 2 + 14, C - ROAD / 2 + 4 + i * 20, 26, 12);
      }
      // stop line
      ctx.fillStyle = k.marking; ctx.fillRect(C, STOP_Y - 2, ROAD / 2, 4);
      this.drawSignals(ctx, k);
      this.drawActors(ctx, k);
      this.drawCar(ctx, this.car.x, this.car.y, this.car.heading, k.ego, k.egoGlass);
      if (this.flash > 0) {
        ctx.strokeStyle = k.critical; ctx.lineWidth = 6; ctx.globalAlpha = Math.min(1, this.flash / 20);
        ctx.strokeRect(3, 3, W - 6, W - 6); ctx.globalAlpha = 1;
      }
    }

    drawSignals(ctx, k) {
      const v = this.vars, x = C + ROAD / 2 + 46, y = STOP_Y + 26;
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
        const sx = C + ROAD / 2 + 22, sy = STOP_Y + 6, r = 15;
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
      const v = this.vars;
      if (v.pedestrian_in_front) person(ctx, C + LANE, C + ROAD / 2 + 27, k);
      if (v.pedestrian_at_right) person(ctx, C + ROAD / 2 + 27, C + LANE, k);
      if (v.pedestrian_at_left) person(ctx, C - ROAD / 2 - 27, C - LANE, k);
      if (v.opposite_car) this.drawCar(ctx, C - LANE, C - ROAD / 2 - 70, Math.PI / 2, k.other, k.egoGlass);
      if (v.car_from_left) this.drawCar(ctx, C - ROAD / 2 - 60, C + LANE, 0, k.other, k.egoGlass);
      if (v.car_from_right) this.drawCar(ctx, C + ROAD / 2 + 60, C - LANE, Math.PI, k.other, k.egoGlass);
      if (v.side_car) this.drawCar(ctx, C + ROAD / 2 + 16, this.car.y, -Math.PI / 2, k.other, k.egoGlass);
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

  global.LogicTuneSim = { Lane, decide, evalGuard, PROPS, MONITORS, EPISODE_TICKS };
})(window);
