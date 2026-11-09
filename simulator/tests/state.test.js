"use strict";
// Shareable replay state in the URL hash (#r=…&s=…&i=…&a=…&b=…).
const test = require("node:test");
const assert = require("node:assert/strict");
const { data } = require("./helpers.js");
const st = require("../assets/replaystate.js");

const DEFAULT = { scenario: 1, models: ["round0", "round3"] };
const resolve = (hash) => st.resolveHash(st.parseHash(hash), data, DEFAULT);

test("a valid hash selects that scenario, sample and pair of models", () => {
  assert.deepEqual(resolve("#r=qwen2.5-3b&s=4&i=3&a=round1&b=round2"), { scenario: 4, sample: 3, models: ["round1", "round2"] });
  assert.equal(st.parseHash("#r=qwen2.5-3b&s=4&i=3&a=round1&b=round2").r, "qwen2.5-3b");
});

test("out-of-range or malformed scenario and sample values fall back to the defaults", () => {
  for (const s of ["99", "0", "abc", "1.5", "-3", ""]) {
    assert.equal(resolve(`#r=qwen2.5-3b&s=${s}&i=0`).scenario, 1, `s=${s}`);
  }
  for (const i of ["25", "20", "-1", "1.5", "x"]) {
    assert.equal(resolve(`#r=qwen2.5-3b&s=4&i=${i}`).sample, 0, `i=${i}`);
  }
  assert.equal(resolve("#r=qwen2.5-3b&s=4&i=19").sample, 19, "the last sample is valid");
});

test("unknown models fall back to the defaults, one lane at a time", () => {
  assert.deepEqual(resolve("#s=2&a=round9&b=round1").models, ["round0", "round1"]);
});

test("a hash without state keys (a section anchor) is not replay state", () => {
  assert.equal(st.parseHash("#results"), null);
  assert.equal(st.parseHash(""), null);
});

test("the written hash reads back as the same state", () => {
  const s = { run: "tinyllama-1.1b", scenario: 7, sample: 3, models: ["round1", "round2"] };
  const h = st.formatHash(s);
  assert.equal(h, "#r=tinyllama-1.1b&s=7&i=3&a=round1&b=round2");
  assert.deepEqual(resolve(h), { scenario: 7, sample: 3, models: ["round1", "round2"] });
});

test("samples are listed in strip order: prompt template, then sample index", () => {
  const xs = st.samplesFor(data, "round0", 4);
  assert.equal(xs.length, 20);
  assert.ok(xs.every((x) => x.scenario_id === 4));
  assert.deepEqual(xs.slice(9, 11).map((x) => [x.template, x.sample_idx]), [[0, 9], [1, 0]]);
});
