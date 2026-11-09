/*
 * Small SVG charts for the results section (no dependencies).
 * Colors come from CSS custom properties, so light and dark themes both work.
 * Every mark has a hover tooltip; every chart has a table view in the page.
 */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const pct = (x) => `${(100 * x).toFixed(1)}%`;

  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }

  let tip;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "chart-tip";
      tip.setAttribute("role", "status");
      document.body.appendChild(tip);
    }
    return tip;
  }
  function bindTip(node, html) {
    const show = (ev) => {
      const t = tooltip();
      t.innerHTML = html;
      t.style.display = "block";
      const r = t.getBoundingClientRect();
      let x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
      if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
      t.style.left = `${x}px`; t.style.top = `${y}px`;
    };
    node.addEventListener("pointermove", show);
    node.addEventListener("pointerenter", show);
    node.addEventListener("pointerleave", () => { tooltip().style.display = "none"; });
    node.setAttribute("tabindex", "0");
    node.addEventListener("focus", () => {
      const b = node.getBoundingClientRect();
      show({ clientX: b.right, clientY: b.top });
    });
    node.addEventListener("blur", () => { tooltip().style.display = "none"; });
  }

  function frame(container, height, margin) {
    container.innerHTML = "";
    const width = Math.max(280, container.clientWidth || 600);
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, role: "img" }, container);
    return { svg, width, height, m: margin, iw: width - margin.l - margin.r, ih: height - margin.t - margin.b };
  }

  function yAxis(f, ticks, fmt) {
    const g = el("g", { class: "axis" }, f.svg);
    for (const v of ticks) {
      const y = f.m.t + f.ih * (1 - v);
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y, y2: y, class: "grid" }, g);
      const t = el("text", { x: f.m.l - 8, y: y + 4, "text-anchor": "end", class: "tick" }, g);
      t.textContent = fmt(v);
    }
  }

  // Satisfaction rate per round: line + 95% CI band + markers.
  function roundsLine(container, rounds, key, ciKey, label) {
    const f = frame(container, 260, { t: 16, r: 24, b: 40, l: 48 });
    f.svg.setAttribute("aria-label", label);
    yAxis(f, [0, 0.25, 0.5, 0.75, 1], (v) => `${Math.round(v * 100)}%`);
    const n = rounds.length;
    const x = (i) => f.m.l + (n === 1 ? f.iw / 2 : (f.iw * i) / (n - 1));
    const y = (v) => f.m.t + f.ih * (1 - v);
    if (ciKey) {
      const top = rounds.map((r, i) => `${x(i)},${y(r[ciKey][1])}`);
      const bot = rounds.map((r, i) => `${x(i)},${y(r[ciKey][0])}`).reverse();
      el("polygon", { points: top.concat(bot).join(" "), class: "band" }, f.svg);
    }
    el("polyline", { points: rounds.map((r, i) => `${x(i)},${y(r[key])}`).join(" "), class: "line" }, f.svg);
    rounds.forEach((r, i) => {
      const cls = r.round === 0 ? "dot base" : i === n - 1 ? "dot ft" : "dot mid";
      const d = el("circle", { cx: x(i), cy: y(r[key]), r: 5, class: cls }, f.svg);
      const ci = ciKey ? `<br>95% CI ${pct(r[ciKey][0])} – ${pct(r[ciKey][1])}` : "";
      bindTip(d, `<b>${r.label}</b><br>${label}: ${pct(r[key])}${ci}`);
      const tx = el("text", { x: x(i), y: f.m.t + f.ih + 24, "text-anchor": "middle", class: "tick" }, f.svg);
      tx.textContent = r.round === 0 ? "Base" : `Round ${r.round}`;
    });
    const last = rounds[n - 1];
    const lab = el("text", { x: x(n - 1) - 8, y: y(last[key]) - 12, "text-anchor": "end", class: "value" }, f.svg);
    lab.textContent = pct(last[key]);
    const first = el("text", { x: x(0) + 8, y: y(rounds[0][key]) - 12, "text-anchor": "start", class: "value" }, f.svg);
    first.textContent = pct(rounds[0][key]);
  }

  // Horizontal paired bars: one row per item, base vs fine-tuned.
  function pairedBars(container, rows, opts) {
    const rowH = 30;
    const f = frame(container, rows.length * rowH + 40, { t: 8, r: 52, b: 28, l: opts.labelWidth || 64 });
    f.svg.setAttribute("aria-label", opts.label);
    const x = (v) => f.m.l + f.iw * v;
    for (const v of [0, 0.5, 1]) {
      el("line", { x1: x(v), x2: x(v), y1: f.m.t, y2: f.m.t + rows.length * rowH, class: "grid" }, f.svg);
      const t = el("text", { x: x(v), y: f.m.t + rows.length * rowH + 18, "text-anchor": "middle", class: "tick" }, f.svg);
      t.textContent = `${v * 100}%`;
    }
    rows.forEach((r, i) => {
      const y0 = f.m.t + i * rowH + 5;
      const name = el("text", { x: f.m.l - 8, y: y0 + 13, "text-anchor": "end", class: "tick" }, f.svg);
      name.textContent = r.name;
      [["base", r.base, 0], ["ft", r.ft, 1]].forEach(([cls, v, j]) => {
        const w = Math.max(0, x(v) - x(0));
        const h = 9, yy = y0 + j * (h + 2);
        const path = w < 4 ? `M${x(0)},${yy}h${w}v${h}h${-w}z`
          : `M${x(0)},${yy}h${w - 4}a4,4 0 0 1 4,4v${h - 8}a4,4 0 0 1 -4,4h${-(w - 4)}z`;
        const bar = el("path", { d: path, class: `bar ${cls}` }, f.svg);
        bindTip(bar, `<b>${r.title || r.name}</b><br>${cls === "base" ? opts.baseLabel : opts.ftLabel}: ${pct(v)}`);
      });
      const d = el("text", { x: x(Math.max(r.base, r.ft)) + 6, y: y0 + 14, class: "delta" }, f.svg);
      const diff = 100 * (r.ft - r.base);
      d.textContent = `${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(0)}`;
    });
  }

  // Dumbbell: base -> fine-tuned per scenario.
  function dumbbell(container, rows, opts) {
    const rowH = 24;
    const f = frame(container, rows.length * rowH + 40, { t: 8, r: 20, b: 28, l: 64 });
    f.svg.setAttribute("aria-label", opts.label);
    const x = (v) => f.m.l + f.iw * v;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: x(v), x2: x(v), y1: f.m.t, y2: f.m.t + rows.length * rowH, class: "grid" }, f.svg);
      const t = el("text", { x: x(v), y: f.m.t + rows.length * rowH + 18, "text-anchor": "middle", class: "tick" }, f.svg);
      t.textContent = `${v * 100}%`;
    }
    rows.forEach((r, i) => {
      const y = f.m.t + i * rowH + rowH / 2;
      const name = el("text", { x: f.m.l - 8, y: y + 4, "text-anchor": "end", class: r.heldout ? "tick held" : "tick" }, f.svg);
      name.textContent = r.heldout ? `S${r.id} ◆` : `S${r.id}`;
      el("line", { x1: x(r.base), x2: x(r.ft), y1: y, y2: y, class: "link" }, f.svg);
      const b = el("circle", { cx: x(r.base), cy: y, r: 5, class: "dot base" }, f.svg);
      const t = el("circle", { cx: x(r.ft), cy: y, r: 5, class: "dot ft" }, f.svg);
      const tipHtml = `<b>Scenario ${r.id}${r.heldout ? " (held out)" : ""}</b><br>${r.text}<br>${opts.baseLabel}: ${pct(r.base)}<br>${opts.ftLabel}: ${pct(r.ft)}`;
      bindTip(b, tipHtml); bindTip(t, tipHtml);
    });
  }

  global.LogicTuneCharts = { roundsLine, pairedBars, dumbbell, bindTip, pct };
})(window);
