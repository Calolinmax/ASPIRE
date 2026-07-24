/**
 * D3 charts for the ASPIRE benchmark page.
 * All numbers come from the ASPIRE NeurIPS 2026 paper (see _source comments
 * in data/benchmark_data.json).
 */

const C = {
  aspire: "#76b900",        // brand lime, a touch toned so large fills don't glare
  aspireDark: "#5f9c00",    // brighter lime for ASPIRE labels/points (readable on dark)
  text: "#1b1e23",
  muted: "#5c636e",        // light slate — readable for axis ticks + value labels
  grid: "rgba(0, 0, 0, 0.08)",
  axis: "rgba(0, 0, 0, 0.22)",
};

let dataCache = null;
let tooltip = null;

async function getData() {
  if (!dataCache) {
    const res = await fetch("data/benchmark_data.json?v=9");
    dataCache = await res.json();
  }
  return dataCache;
}

function getTooltip() {
  if (!tooltip) {
    tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "chart-tooltip-floating")
      .style("opacity", 0);
  }
  return tooltip;
}

function showTip(html, x, y) {
  getTooltip()
    .html(html)
    .style("left", x + 14 + "px")
    .style("top", y - 10 + "px")
    .style("opacity", 1);
}

function hideTip() {
  if (tooltip) tooltip.style("opacity", 0);
}

function panelWidth(el, fallback = 880) {
  const cs = window.getComputedStyle(el);
  const padX = parseFloat(cs.paddingLeft || 0) + parseFloat(cs.paddingRight || 0);
  const w = (el.clientWidth || fallback) - padX;
  return Math.max(320, w);
}

function styleAxis(g) {
  g.selectAll("path").attr("stroke", C.axis);
  g.selectAll("line").attr("stroke", C.axis);
  g.selectAll("text").attr("fill", C.muted).attr("font-size", "11px");
}

function bindBarHover(selection, tipFn) {
  selection
    .on("mouseenter", function (event, d) {
      d3.select(this)
        .attr("fill-opacity", 1)
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.75);
      const t = tipFn(d);
      if (t) showTip(t, event.pageX, event.pageY);
    })
    .on("mousemove", function (event, d) {
      const t = tipFn(d);
      if (t) showTip(t, event.pageX, event.pageY);
    })
    .on("mouseleave", function () {
      d3.select(this).attr("fill-opacity", 0.85).attr("stroke", "none");
      hideTip();
    });
}

function addYAxisTitle(svg, x, y, label) {
  svg
    .append("text")
    .attr("transform", `translate(${x},${y}) rotate(-90)`)
    .attr("text-anchor", "middle")
    .attr("fill", C.muted)
    .attr("font-size", "11px")
    .attr("font-weight", 600)
    .text(label);
}

function formatPct(v) {
  return Number.isInteger(v) ? `${v}%` : `${v.toFixed(1)}%`;
}

/* ── Entrance animations (played when a chart scrolls into view) ── */
// Grow every bar from its own base: works for grouped bars (base = axis
// baseline) and stacked segments (base = the segment below it).
function animateGrow(g) {
  const rects = g.selectAll("rect");
  rects.each(function () {
    const r = d3.select(this);
    r.attr("data-fy", r.attr("y")).attr("data-fh", r.attr("height"));
  });
  rects
    .attr("y", function () {
      const r = d3.select(this);
      return +r.attr("data-fy") + +r.attr("data-fh");
    })
    .attr("height", 0)
    .transition()
    .duration(600)
    .delay((_, i) => i * 24)
    .ease(d3.easeCubicOut)
    .attr("y", function () { return +d3.select(this).attr("data-fy"); })
    .attr("height", function () { return +d3.select(this).attr("data-fh"); });
}

// Draw a line on left-to-right, then restore its original dash pattern.
function animateLineDraw(path, dur = 850, delay = 0) {
  const node = path.node();
  if (!node || !node.getTotalLength) return;
  const len = node.getTotalLength();
  const orig = path.attr("stroke-dasharray");
  path
    .attr("stroke-dasharray", `${len} ${len}`)
    .attr("stroke-dashoffset", len)
    .transition()
    .duration(dur)
    .delay(delay)
    .ease(d3.easeCubicInOut)
    .attr("stroke-dashoffset", 0)
    .on("end", function () { d3.select(this).attr("stroke-dasharray", orig); });
}

// Pop data points in after the line has mostly drawn.
function animatePoints(sel, baseR = 5, startDelay = 300) {
  sel
    .attr("r", 0)
    .transition()
    .duration(280)
    .delay((_, i) => startDelay + i * 55)
    .ease(d3.easeBackOut)
    .attr("r", baseR);
}

/* ─────────────────── Portrait phones: horizontal bars ───────────────────
   On a narrow portrait phone, vertical grouped bars get too thin. Each bar
   chart maps its data into this shared renderer, which lays the bars out
   horizontally (value runs left→right, groups stack top→bottom) so every bar
   gets the full panel width. Desktop/landscape keep the vertical charts. */
const isPortrait = () => window.matchMedia("(max-width: 640px) and (orientation: portrait)").matches;

// Wrap a row caption into (up to) two balanced lines so it fits a narrow left
// gutter — e.g. "Object · Pos" → ["Object","Pos"], "Cube Lift" → ["Cube","Lift"].
function wrapLabel(text) {
  if (text.includes(" · ")) return text.split(" · ");
  const words = text.split(" ");
  if (words.length <= 1) return [text];
  let best = 1, bestDiff = Infinity;
  for (let i = 1; i < words.length; i++) {
    const diff = Math.abs(words.slice(0, i).join(" ").length - words.slice(i).join(" ").length);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  }
  return [words.slice(0, best).join(" "), words.slice(best).join(" ")];
}

function drawHBarChart(elId, legendId, model) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = "";

  const rows = model.rows;
  const seriesN = rows.reduce((m, r) => Math.max(m, r.bars.length), 1);
  const barH = 15;     // each bar's thickness
  const barGap = 3;    // gap between bars within a group
  const rowGap = 18;   // gap between groups
  const rowH = seriesN * barH + (seriesN - 1) * barGap;

  const W = panelWidth(el);
  const margin = { top: 14, right: 50, bottom: 26, left: 70 };
  const innerW = Math.max(70, W - margin.left - margin.right);
  const innerH = rows.length * rowH + Math.max(0, rows.length - 1) * rowGap;
  const totalH = innerH + margin.top + margin.bottom;
  const fmt = model.fmt || ((v) => v);

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, model.max]).range([0, innerW]).nice();

  // vertical gridlines
  g.append("g")
    .call(d3.axisTop(x).ticks(4).tickSize(-innerH).tickFormat(""))
    .call((s) => s.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  rows.forEach((row, ri) => {
    const ry = ri * (rowH + rowGap);
    const lines = wrapLabel(row.label);
    const lineH = 11.5;
    const startY = ry + rowH / 2 - ((lines.length - 1) * lineH) / 2;
    lines.forEach((ln, li) => {
      g.append("text")
        .attr("x", -7)
        .attr("y", startY + li * lineH)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "10.5px")
        .attr("font-weight", 600)
        .attr("fill", C.text)
        .text(ln);
    });
    row.bars.forEach((bar, bi) => {
      const by = ry + bi * (barH + barGap);
      const w = Math.max(0, x(bar.value));
      const rect = g
        .append("rect")
        .attr("x", 0)
        .attr("y", by)
        .attr("height", barH)
        .attr("width", w)
        .attr("rx", 1.5)
        .attr("fill", bar.color)
        .attr("fill-opacity", 0.85)
        .style("cursor", "pointer")
        .datum({ label: row.label, series: bar.seriesLabel, value: bar.value });
      bindBarHover(rect, (d) => `<strong>${d.series}</strong><br>${d.label}: <b>${fmt(d.value)}</b>`);
      g.append("text")
        .attr("x", w + 5)
        .attr("y", by + barH / 2)
        .attr("dominant-baseline", "middle")
        .attr("font-size", "10px")
        .attr("font-weight", bar.highlight ? 700 : 600)
        .attr("fill", bar.highlight ? C.aspireDark : C.muted)
        .text(bar.barLabel != null ? bar.barLabel : fmt(bar.value));
    });
  });

  const xAxis = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(4).tickFormat(fmt))
    .call((s) => s.select(".domain").attr("stroke", C.axis));
  styleAxis(xAxis);

  // grow bars from the left
  g.selectAll("rect")
    .each(function () { const r = d3.select(this); r.attr("data-fw", r.attr("width")); })
    .attr("width", 0)
    .transition()
    .duration(550)
    .delay((_, i) => i * 18)
    .ease(d3.easeCubicOut)
    .attr("width", function () { return +d3.select(this).attr("data-fw"); });

  if (legendId) renderLegend(legendId, model.legend);
}

/* ─────────────────────── LIBERO-PRO ─────────────────────── */
async function initLiberoChart() {
  const data = await getData();
  const lp = data.liberoPro;
  const el = document.getElementById("libero-chart");
  if (!el) return;

  if (isPortrait()) {
    const rows = [];
    lp.suites.forEach((suite, si) => lp.axes.forEach((axis) => {
      const key = axis === "Pos" ? "pos" : "task";
      rows.push({
        label: `${suite} · ${axis}`,
        bars: lp.methods.map((m) => ({ value: lp[key][m.id][si], color: m.color, highlight: m.id === "aspire", seriesLabel: m.label })),
      });
    }));
    return drawHBarChart("libero-chart", "libero-legend", {
      rows, max: 100, fmt: (v) => v + "%",
      legend: lp.methods.map((m) => ({ label: m.label, color: m.color })),
    });
  }

  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 12, bottom: 60, left: 60 };
  const innerW = W - margin.left - margin.right;
  const innerH = 340;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // Outer: suite. Inner: axis (Pos, Task). Innermost: method.
  const x0 = d3
    .scaleBand()
    .domain(lp.suites)
    .range([0, innerW])
    .paddingInner(0.18)
    .paddingOuter(0.05);
  const x1 = d3
    .scaleBand()
    .domain(lp.axes)
    .range([0, x0.bandwidth()])
    .paddingInner(0.18);
  const x2 = d3
    .scaleBand()
    .domain(lp.methods.map((m) => m.id))
    .range([0, x1.bandwidth()])
    .paddingInner(0.12);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]).nice();

  // grid
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  // suite groups
  const suiteG = g
    .selectAll(".suite")
    .data(lp.suites)
    .join("g")
    .attr("class", "suite")
    .attr("transform", (s) => `translate(${x0(s)},0)`);

  // axis sub-groups (Pos, Task)
  const axisG = suiteG
    .selectAll(".axis-group")
    .data((suite) => lp.axes.map((axis) => ({ suite, axis })))
    .join("g")
    .attr("class", "axis-group")
    .attr("transform", (d) => `translate(${x1(d.axis)},0)`);

  // bars per method inside each (suite, axis)
  axisG.each(function ({ suite, axis }) {
    const si = lp.suites.indexOf(suite);
    const axisKey = axis === "Pos" ? "pos" : "task";
    const records = lp.methods.map((m) => ({
      suite,
      axis,
      method: m.label,
      methodId: m.id,
      color: m.color,
      value: lp[axisKey][m.id][si],
    }));
    const sub = d3.select(this);
    const bars = sub
      .selectAll("rect")
      .data(records)
      .join("rect")
      .attr("x", (d) => x2(d.methodId))
      .attr("width", x2.bandwidth())
      .attr("y", (d) => y(d.value))
      .attr("height", (d) => innerH - y(d.value))
      .attr("rx", 1.5)
      .attr("fill", (d) => d.color)
      .attr("fill-opacity", 0.85)
      .style("cursor", "pointer");
    bindBarHover(
      bars,
      (d) =>
        `<strong>${d.method}</strong><br>${d.suite} · ${d.axis}: <b>${d.value}%</b>`
    );

    // value labels above every bar (zero values rendered as a small "0" baseline marker)
    sub
      .selectAll("text.bar-value")
      .data(records)
      .join("text")
      .attr("class", "bar-value")
      .attr("x", (d) => x2(d.methodId) + x2.bandwidth() / 2)
      .attr("y", (d) => y(d.value) - 4)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("font-weight", (d) => (d.methodId === "aspire" ? 700 : 600))
      .attr("fill", (d) => (d.methodId === "aspire" ? C.aspireDark : C.muted))
      .text((d) => d.value);
  });

  // axis (Pos / Task) labels
  suiteG.each(function (suite) {
    const node = d3.select(this);
    lp.axes.forEach((axis) => {
      node
        .append("text")
        .attr("x", x1(axis) + x1.bandwidth() / 2)
        .attr("y", innerH + 16)
        .attr("text-anchor", "middle")
        .attr("font-size", "10.5px")
        .attr("fill", C.muted)
        .text(axis);
    });
  });

  // suite labels (below axis labels)
  suiteG
    .append("text")
    .attr("x", x0.bandwidth() / 2)
    .attr("y", innerH + 38)
    .attr("text-anchor", "middle")
    .attr("font-size", "12.5px")
    .attr("font-weight", 600)
    .attr("fill", C.text)
    .text((s) => `libero-${s.toLowerCase()}`);

  // y axis
  const yAxis = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"))
    .call((sel) => sel.select(".domain").attr("stroke", C.axis));
  styleAxis(yAxis);

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "Success rate (%)");

  animateGrow(g);

  renderLegend(
    "libero-legend",
    lp.methods.map((m) => ({ label: m.label, color: m.color }))
  );
}

/* ─────────────────────── Robosuite (vertical bars) ─────────────────────── */
async function initRobosuiteChart() {
  const data = await getData();
  const rs = data.robosuite;
  const el = document.getElementById("robosuite-chart");
  if (!el) return;

  if (isPortrait()) {
    return drawHBarChart("robosuite-chart", "robosuite-legend", {
      rows: rs.tasks.map((t, ti) => ({
        label: t,
        bars: rs.methods.map((m) => ({ value: rs.values[m.id][ti], color: m.color, highlight: m.id === "aspire", seriesLabel: m.label })),
      })),
      max: 100, fmt: (v) => v + "%",
      legend: rs.methods.map((m) => ({ label: m.label, color: m.color })),
    });
  }

  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 12, bottom: 56, left: 60 };
  const innerW = W - margin.left - margin.right;
  const innerH = 340;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // Outer band: task. Inner band: method.
  const x0 = d3
    .scaleBand()
    .domain(rs.tasks)
    .range([0, innerW])
    .paddingInner(0.24)
    .paddingOuter(0.06);
  const x1 = d3
    .scaleBand()
    .domain(rs.methods.map((m) => m.id))
    .range([0, x0.bandwidth()])
    .paddingInner(0.16);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]).nice();

  // y grid
  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  // bars
  rs.tasks.forEach((task, ti) => {
    rs.methods.forEach((m) => {
      const val = rs.values[m.id][ti];
      const rect = g
        .append("rect")
        .attr("x", x0(task) + x1(m.id))
        .attr("width", x1.bandwidth())
        .attr("y", y(val))
        .attr("height", innerH - y(val))
        .attr("rx", 1.5)
        .attr("fill", m.color)
        .attr("fill-opacity", 0.85)
        .style("cursor", "pointer")
        .datum({ task, method: m.label, value: val });
      bindBarHover(
        rect,
        (d) => `<strong>${d.method}</strong><br>${d.task}: <b>${d.value}%</b>`
      );

      // value label above each bar
      g.append("text")
        .attr("x", x0(task) + x1(m.id) + x1.bandwidth() / 2)
        .attr("y", y(val) - 5)
        .attr("text-anchor", "middle")
        .attr("font-size", "10.5px")
        .attr("font-weight", m.id === "aspire" ? 700 : 600)
        .attr("fill", m.id === "aspire" ? C.aspireDark : C.muted)
        .text(val);
    });
  });

  // x axis (task labels)
  const xAxis = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x0).tickSize(0));
  xAxis.call((sel) => sel.select(".domain").remove());
  xAxis
    .selectAll("text")
    .attr("fill", C.text)
    .attr("font-size", "11px")
    .attr("font-weight", 600)
    .attr("dy", "1.15em");

  // y axis
  const yAxis = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"))
    .call((sel) => sel.select(".domain").attr("stroke", C.axis));
  styleAxis(yAxis);

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "Success rate (%)");

  animateGrow(g);

  renderLegend(
    "robosuite-legend",
    rs.methods.map((m) => ({ label: m.label, color: m.color }))
  );
}

/* ─────────────────────── BEHAVIOR-1K ─────────────────────── */
async function initBehaviorChart() {
  const data = await getData();
  const bh = data.behavior;
  const el = document.getElementById("behavior-chart");
  if (!el) return;

  if (isPortrait()) {
    const rows = [];
    bh.tasks.forEach((t, ti) => bh.metrics.forEach((metric) => {
      const key = metric === "Nav" ? "nav" : "task";
      rows.push({
        label: `${t.split(" ")[0]} · ${metric}`,
        bars: bh.series.map((s) => ({ value: bh[key][s.id][ti], color: s.color, highlight: s.id === "aspire", seriesLabel: s.label })),
      });
    }));
    return drawHBarChart("behavior-chart", "behavior-legend", {
      rows, max: 100, fmt: (v) => v + "%",
      legend: bh.series.map((s) => ({ label: s.label, color: s.color })),
    });
  }

  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 12, bottom: 64, left: 60 };
  const innerW = W - margin.left - margin.right;
  const innerH = 320;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x0 = d3
    .scaleBand()
    .domain(bh.tasks)
    .range([0, innerW])
    .paddingInner(0.25)
    .paddingOuter(0.08);
  const x1 = d3
    .scaleBand()
    .domain(bh.metrics)
    .range([0, x0.bandwidth()])
    .paddingInner(0.2);
  const x2 = d3
    .scaleBand()
    .domain(bh.series.map((s) => s.id))
    .range([0, x1.bandwidth()])
    .paddingInner(0.1);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]).nice();

  // grid
  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  // y axis
  const yAxisG = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"));
  styleAxis(yAxisG);

  // bars + per-bar labels
  bh.tasks.forEach((task, ti) => {
    const tG = g.append("g").attr("transform", `translate(${x0(task)},0)`);
    bh.metrics.forEach((metric) => {
      const mG = tG.append("g").attr("transform", `translate(${x1(metric)},0)`);
      const key = metric === "Nav" ? "nav" : "task";
      bh.series.forEach((s) => {
        const val = bh[key][s.id][ti];
        const rect = mG
          .append("rect")
          .attr("x", x2(s.id))
          .attr("width", x2.bandwidth())
          .attr("y", y(val))
          .attr("height", innerH - y(val))
          .attr("rx", 1.5)
          .attr("fill", s.color)
          .attr("fill-opacity", 0.85)
          .style("cursor", "pointer")
          .datum({ task, metric, method: s.label, value: val });
        bindBarHover(
          rect,
          (d) =>
            `<strong>${d.method}</strong><br>${d.task} · ${d.metric}: <b>${d.value}%</b>`
        );
        // value label
        mG.append("text")
          .attr("x", x2(s.id) + x2.bandwidth() / 2)
          .attr("y", y(val) - 4)
          .attr("text-anchor", "middle")
          .attr("font-size", "11px")
          .attr("font-weight", 700)
          .attr("fill", s.id === "aspire" ? C.aspireDark : C.text)
          .text(val);
      });
      // metric (Nav/Task) label
      mG.append("text")
        .attr("x", x1.bandwidth() / 2)
        .attr("y", innerH + 16)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("fill", C.muted)
        .text(metric);
    });
    // task name below metrics
    tG.append("text")
      .attr("x", x0.bandwidth() / 2)
      .attr("y", innerH + 42)
      .attr("text-anchor", "middle")
      .attr("font-size", "12.5px")
      .attr("font-weight", 600)
      .attr("fill", C.text)
      .text(task);
  });

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "Success rate (%)");

  animateGrow(g);

  renderLegend(
    "behavior-legend",
    bh.series.map((s) => ({ label: s.label, color: s.color }))
  );
}

/* ─────────────────────── Zero-shot transfer ─────────────────────── */
async function initZeroshotChart() {
  const data = await getData();
  const zs = data.zeroshot;
  const el = document.getElementById("zeroshot-chart");
  if (!el) return;
  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 24, bottom: 48, left: 60 };
  const innerW = W - margin.left - margin.right;
  const innerH = isPortrait() ? 224 : 320;   // shrink the scaling-law plot ~30% on portrait
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 90]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 45]).range([innerH, 0]).nice();

  // grid
  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  // Prior methods have no skill library, so they stay flat near the floor.
  // Mark that ceiling with a single dashed reference line (absorbs the former
  // separate baseline-bar chart).
  const priorMax = d3.max(zs.baselines, (b) => Math.max(b.pos, b.task));
  g.append("line")
    .attr("x1", 0)
    .attr("x2", innerW)
    .attr("y1", y(priorMax))
    .attr("y2", y(priorMax))
    .attr("stroke", C.muted)
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "7,4")
    .attr("stroke-opacity", 0.95);
  g.append("text")
    .attr("x", 8)
    .attr("y", y(priorMax) - 7)
    .attr("font-size", "11px")
    .attr("font-weight", 700)
    .attr("fill", C.muted)
    .text(`Prior methods (no library) ≤ ${formatPct(priorMax)}`);

  // x axis
  const xAxisG = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(
      d3
        .axisBottom(x)
        .tickValues(zs.librarySizes)
        .tickFormat((d) => `N=${d}`)
    );
  styleAxis(xAxisG);

  // y axis
  const yAxisG = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"));
  styleAxis(yAxisG);

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "LIBERO-Long success rate (%)");
  svg
    .append("text")
    .attr("x", margin.left + innerW / 2)
    .attr("y", totalH - 6)
    .attr("text-anchor", "middle")
    .attr("font-size", "11px")
    .attr("fill", C.muted)
    .text("Skill library size (# of LIBERO-90 tasks seeded)");

  // ASPIRE curves
  const linePos = d3
    .line()
    .x((_, i) => x(zs.librarySizes[i]))
    .y((d) => y(d))
    .curve(d3.curveMonotoneX);
  const lineTask = d3
    .line()
    .x((_, i) => x(zs.librarySizes[i]))
    .y((d) => y(d))
    .curve(d3.curveMonotoneX);

  const posPath = g.append("path")
    .datum(zs.aspire.pos)
    .attr("fill", "none")
    .attr("stroke", C.aspireDark)
    .attr("stroke-width", 2.5)
    .attr("stroke-dasharray", "6,4")
    .attr("d", linePos);
  const taskPath = g.append("path")
    .datum(zs.aspire.task)
    .attr("fill", "none")
    .attr("stroke", C.aspire)
    .attr("stroke-width", 3)
    .attr("d", lineTask);
  animateLineDraw(taskPath);
  animateLineDraw(posPath);

  // points
  zs.librarySizes.forEach((n, i) => {
    [
      { kind: "Pos", val: zs.aspire.pos[i], fill: "#ffffff", stroke: C.aspireDark },
      { kind: "Task", val: zs.aspire.task[i], fill: C.aspire, stroke: C.aspireDark },
    ].forEach(({ kind, val, fill, stroke }) => {
      g.append("circle")
        .attr("cx", x(n))
        .attr("cy", y(val))
        .attr("r", 5)
        .attr("fill", fill)
        .attr("stroke", stroke)
        .attr("stroke-width", 2)
        .style("cursor", "pointer")
        .on("mouseenter", (event) => {
          showTip(
            `<strong>ASPIRE · ${kind}</strong><br>N=${n}: <b>${formatPct(val)}</b>`,
            event.pageX,
            event.pageY
          );
          d3.select(event.currentTarget).attr("r", 7);
        })
        .on("mouseleave", (event) => {
          hideTip();
          d3.select(event.currentTarget).attr("r", 5);
        });
    });
  });

  // value labels at N=90 endpoint
  const lastIdx = zs.librarySizes.length - 1;
  g.append("text")
    .attr("x", x(zs.librarySizes[lastIdx]) - 6)
    .attr("y", y(zs.aspire.task[lastIdx]) - 8)
    .attr("text-anchor", "end")
    .attr("font-size", "11px")
    .attr("font-weight", 700)
    .attr("fill", C.aspireDark)
    .text(formatPct(zs.aspire.task[lastIdx]));
  g.append("text")
    .attr("x", x(zs.librarySizes[lastIdx]) - 6)
    .attr("y", y(zs.aspire.pos[lastIdx]) - 8)
    .attr("text-anchor", "end")
    .attr("font-size", "11px")
    .attr("font-weight", 700)
    .attr("fill", C.aspireDark)
    .text(formatPct(zs.aspire.pos[lastIdx]));

  animatePoints(g.selectAll("circle"), 5, 500);

  renderLegend("zeroshot-legend", [
    { label: "ASPIRE · Task perturbation", color: C.aspire, kind: "line" },
    { label: "ASPIRE · Pos perturbation", color: C.aspireDark, kind: "dash" },
    { label: "Prior methods (π<sub>0.5</sub>, CaP-Agent0)", color: C.muted, kind: "dash" },
  ]);
}

/* ─────────────────────── Zero-shot N=90 vs. baselines ─────────────────────── */
async function initZeroshotBaselineChart() {
  const data = await getData();
  const zs = data.zeroshot;
  const el = document.getElementById("zeroshot-baseline-chart");
  if (!el) return;
  el.innerHTML = "";

  // Figure 5(a) compares the three non-zero methods: ASPIRE (N=90), CaP-Agent0, π0.5
  const lastIdx = zs.librarySizes.length - 1;
  const methods = [
    {
      id: "pi05",
      label: "π<sub>0.5</sub>",
      color: "#93a0b2",
      pos: zs.baselines.find((b) => b.label === "π<sub>0.5</sub>").pos,
      task: zs.baselines.find((b) => b.label === "π<sub>0.5</sub>").task,
    },
    {
      id: "cap0",
      label: "CaP-Agent0",
      color: "#6b7686",
      pos: zs.baselines.find((b) => b.label === "CaP-Agent0").pos,
      task: zs.baselines.find((b) => b.label === "CaP-Agent0").task,
    },
    {
      id: "aspire",
      label: "ASPIRE (N=90)",
      color: C.aspire,
      pos: zs.aspire.pos[lastIdx],
      task: zs.aspire.task[lastIdx],
    },
  ];
  const axes = ["Pos (avg.)", "Task (avg.)"];

  const W = panelWidth(el);
  const margin = { top: 18, right: 16, bottom: 44, left: 60 };
  const innerW = W - margin.left - margin.right;
  const innerH = 320;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x0 = d3
    .scaleBand()
    .domain(axes)
    .range([0, innerW])
    .paddingInner(0.4)
    .paddingOuter(0.15);
  const x1 = d3
    .scaleBand()
    .domain(methods.map((m) => m.id))
    .range([0, x0.bandwidth()])
    .paddingInner(0.18);
  const y = d3.scaleLinear().domain([0, 45]).range([innerH, 0]).nice();

  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  const xAxisG = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x0).tickSize(0));
  styleAxis(xAxisG);
  xAxisG
    .selectAll("text")
    .attr("font-size", "12.5px")
    .attr("font-weight", 600)
    .attr("fill", C.text)
    .attr("dy", "1.4em");

  const yAxisG = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"));
  styleAxis(yAxisG);

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "Success rate (%)");

  axes.forEach((axis) => {
    const axisKey = axis === "Pos (avg.)" ? "pos" : "task";
    const aG = g.append("g").attr("transform", `translate(${x0(axis)},0)`);
    methods.forEach((m) => {
      const val = m[axisKey];
      const rect = aG
        .append("rect")
        .attr("x", x1(m.id))
        .attr("width", x1.bandwidth())
        .attr("y", y(val))
        .attr("height", innerH - y(val))
        .attr("rx", 1.5)
        .attr("fill", m.color)
        .attr("fill-opacity", 0.85)
        .style("cursor", "pointer")
        .datum({ axis, method: m.label, value: val });
      bindBarHover(
        rect,
        (d) =>
          `<strong>${d.method}</strong><br>${d.axis}: <b>${formatPct(d.value)}</b>`
      );
      aG.append("text")
        .attr("x", x1(m.id) + x1.bandwidth() / 2)
        .attr("y", y(val) - 5)
        .attr("text-anchor", "middle")
        .attr("font-size", "10.5px")
        .attr("font-weight", m.id === "aspire" ? 700 : 600)
        .attr("fill", m.id === "aspire" ? C.aspireDark : C.muted)
        .text(formatPct(val));
    });
  });

  renderLegend(
    "zeroshot-baseline-legend",
    methods.map((m) => ({ label: m.label, color: m.color }))
  );
}

/* ─────────────────────── Ablations Figure 6(a, b): Pos + Task ─────────────────────── */
async function initAblationChart() {
  const data = await getData();
  const ab = data.ablation;
  const el = document.getElementById("ablation-chart");
  if (!el) return;
  el.innerHTML = "";

  const axes = [
    { label: "Pos perturbation", key: "pos" },
    { label: "Task perturbation", key: "task" },
  ];

  const W = panelWidth(el);
  const margin = { top: 18, right: 16, bottom: 48, left: 56 };
  const innerW = W - margin.left - margin.right;
  const innerH = 300;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3
    .scaleBand()
    .domain(axes.map((a) => a.label))
    .range([0, innerW])
    .paddingInner(0.45)
    .paddingOuter(0.2);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]).nice();

  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  const xAxisG = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x).tickSize(0))
    .call((sel) => sel.select(".domain").remove());
  styleAxis(xAxisG);
  xAxisG
    .selectAll("text")
    .attr("font-size", "12.5px")
    .attr("font-weight", 600)
    .attr("fill", C.text)
    .attr("dy", "1.3em");

  const yAxisG = g
    .append("g")
    .call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"))
    .call((sel) => sel.select(".domain").remove());
  styleAxis(yAxisG);

  // baseline at y=0 sits flush with the bars
  g.append("line")
    .attr("x1", 0)
    .attr("x2", innerW)
    .attr("y1", innerH)
    .attr("y2", innerH)
    .attr("stroke", C.axis)
    .attr("stroke-width", 1);

  addYAxisTitle(svg, 16, margin.top + innerH / 2, "Macro-avg. success rate (%)");

  axes.forEach(({ label, key }) => {
    let cum = 0;
    ab.conditions.forEach((cond) => {
      const inc = ab.increments[key][cond.id];
      const newCum = cum + inc;
      const rect = g
        .append("rect")
        .attr("x", x(label))
        .attr("width", x.bandwidth())
        .attr("y", y(newCum))
        .attr("height", y(cum) - y(newCum))
        .attr("fill", cond.color)
        .attr("fill-opacity", 0.85)
        .attr("stroke", "none")
        .style("cursor", "pointer")
        .datum({ axis: label, condition: cond.label, increment: inc, cumulative: newCum });
      bindBarHover(
        rect,
        (d) =>
          `<strong>${d.condition}</strong><br>${d.axis}<br>+${d.increment}% &middot; total <b>${d.cumulative}%</b>`
      );
      const segH = y(cum) - y(newCum);
      if (segH > 16) {
        g.append("text")
          .attr("x", x(label) + x.bandwidth() / 2)
          .attr("y", y(newCum) + segH / 2 + 4)
          .attr("text-anchor", "middle")
          .attr("font-size", "11px")
          .attr("font-weight", 700)
          .attr("fill", cond.id === "evo" ? "#ffffff" : C.text)
          .text(`+${inc}%`);
      }
      cum = newCum;
    });
    g.append("text")
      .attr("x", x(label) + x.bandwidth() / 2)
      .attr("y", y(cum) - 8)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", 800)
      .attr("fill", C.aspireDark)
      .text(`${cum}%`);
  });

  animateGrow(g);

  renderLegend(
    "ablation-legend",
    ab.conditions.map((c) => ({ label: c.label, color: c.color }))
  );
}

/* ─────────────────────── Ablation Figure 6(c): Evo iterations ─────────────────────── */
async function initEvoIterationChart() {
  const data = await getData();
  const ev = data.evoIteration;
  const el = document.getElementById("evo-iter-chart");
  if (!el) return;
  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 16, bottom: 44, left: 52 };
  const innerW = W - margin.left - margin.right;
  const innerH = 280;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 4]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 100]).range([innerH, 0]).nice();

  g.append("g")
    .call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  const xAxisG = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x).tickValues(ev.iterations).tickFormat((d) => d));
  styleAxis(xAxisG);

  const yAxisG = g.append("g").call(d3.axisLeft(y).ticks(5).tickFormat((d) => d + "%"));
  styleAxis(yAxisG);

  addYAxisTitle(svg, 14, margin.top + innerH / 2, "Mean success rate (%)");
  svg
    .append("text")
    .attr("x", margin.left + innerW / 2)
    .attr("y", totalH - 6)
    .attr("text-anchor", "middle")
    .attr("font-size", "11px")
    .attr("fill", C.muted)
    .text("Evolutionary-search iteration");

  const taskLine = d3
    .line()
    .x((_, i) => x(ev.iterations[i]))
    .y((d) => y(d))
    .curve(d3.curveMonotoneX);

  // Mean line only
  const meanPath = g.append("path")
    .datum(ev.mean)
    .attr("fill", "none")
    .attr("stroke", C.aspire)
    .attr("stroke-width", 3)
    .attr("d", taskLine);
  animateLineDraw(meanPath);

  // Mean markers + value labels
  ev.iterations.forEach((it, i) => {
    const val = ev.mean[i];
    g.append("circle")
      .attr("cx", x(it))
      .attr("cy", y(val))
      .attr("r", 5)
      .attr("fill", C.aspire)
      .attr("stroke", C.aspireDark)
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("mouseenter", (event) => {
        showTip(
          `<strong>Iter ${it}</strong><br>Mean SR: <b>${formatPct(val)}</b>`,
          event.pageX,
          event.pageY
        );
      })
      .on("mouseleave", hideTip);
    g.append("text")
      .attr("x", x(it))
      .attr("y", y(val) - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "10.5px")
      .attr("font-weight", 700)
      .attr("fill", C.aspireDark)
      .text(formatPct(val));
  });

  animatePoints(g.selectAll("circle"), 5, 500);

  renderLegend("evo-iter-legend", [
    { label: "Mean over 8 low-performing tasks", color: C.aspire, kind: "line" },
  ]);
}

/* ─────────────────────── Legend ─────────────────────── */
// Subscript the π model names (so the "0.5" — dot included — sits low),
// while leaving CaP-Agent0 with a normal baseline 0.
function fmtLabel(s) {
  return s
    .replaceAll("π0.5", "π<sub>0.5</sub>")
    .replaceAll("π0", "π<sub>0</sub>");
}

function renderLegend(id, items) {
  const node = document.getElementById(id);
  if (!node) return;
  node.classList.add("legend");
  node.innerHTML = items
    .map((it) => {
      const swatch =
        it.kind === "dash"
          ? `<span class="chart-legend-dash" style="border-color:${it.color}"></span>`
          : it.kind === "line"
          ? `<span class="chart-legend-line" style="background:${it.color}"></span>`
          : `<span class="chart-legend-swatch" style="background:${it.color}"></span>`;
      return `<span class="chart-legend-item">${swatch}<span class="chart-legend-text">${fmtLabel(it.label)}</span></span>`;
    })
    .join("");
}

/* ───── Real-robot cross-embodiment skill transfer (Table 1) ─────
   Grouped bars per task, w/o Skills vs w/ Skills. Two views: debugging-token
   cost (↓) and held-out success (↑), since the two metrics differ in scale. */
function drawTransferBars(elId, legendId, rt, opts) {
  const el = document.getElementById(elId);
  if (!el) return;

  if (isPortrait()) {
    return drawHBarChart(elId, legendId, {
      rows: rt.tasks.map((t, ti) => ({
        label: t,
        bars: rt.series.map((s) => ({
          value: opts.values[s.id][ti],
          color: s.color,
          highlight: !!s.highlight,
          seriesLabel: s.label,
          barLabel: opts.barFmt(opts.values[s.id][ti], ti, s.id),
        })),
      })),
      max: opts.yMax, fmt: opts.tickFmt,
      legend: rt.series.map((s) => ({ label: s.label, color: s.color })),
    });
  }

  el.innerHTML = "";

  const W = panelWidth(el);
  const margin = { top: 18, right: 14, bottom: 56, left: 64 };
  const innerW = W - margin.left - margin.right;
  const innerH = 320;
  const totalH = innerH + margin.top + margin.bottom;

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", W)
    .attr("height", totalH)
    .attr("viewBox", `0 0 ${W} ${totalH}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x0 = d3.scaleBand().domain(rt.tasks).range([0, innerW]).paddingInner(0.34).paddingOuter(0.12);
  const x1 = d3.scaleBand().domain(rt.series.map((s) => s.id)).range([0, x0.bandwidth()]).paddingInner(0.2);
  const y = d3.scaleLinear().domain([0, opts.yMax]).range([innerH, 0]).nice();

  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(opts.yTicks).tickSize(-innerW).tickFormat(""))
    .call((sel) => sel.select(".domain").remove())
    .selectAll("line")
    .attr("stroke", C.grid);

  rt.tasks.forEach((task, ti) => {
    rt.series.forEach((s) => {
      const val = opts.values[s.id][ti];
      const rect = g
        .append("rect")
        .attr("x", x0(task) + x1(s.id))
        .attr("width", x1.bandwidth())
        .attr("y", y(val))
        .attr("height", innerH - y(val))
        .attr("rx", 1.5)
        .attr("fill", s.color)
        .attr("fill-opacity", 0.85)
        .style("cursor", "pointer")
        .datum({ task, label: s.label, idx: ti, sid: s.id });
      bindBarHover(rect, opts.tipFmt);

      g.append("text")
        .attr("x", x0(task) + x1(s.id) + x1.bandwidth() / 2)
        .attr("y", y(val) - 5)
        .attr("text-anchor", "middle")
        .attr("font-size", "10.5px")
        .attr("font-weight", s.highlight ? 700 : 600)
        .attr("fill", s.highlight ? C.aspireDark : C.muted)
        .text(opts.barFmt(val, ti, s.id));
    });
  });

  const xAxis = g
    .append("g")
    .attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x0).tickSize(0));
  xAxis.call((sel) => sel.select(".domain").remove());
  xAxis
    .selectAll("text")
    .attr("fill", C.text)
    .attr("font-size", "11px")
    .attr("font-weight", 600)
    .attr("dy", "1.15em");

  const yAxis = g
    .append("g")
    .call(d3.axisLeft(y).ticks(opts.yTicks).tickFormat(opts.tickFmt))
    .call((sel) => sel.select(".domain").attr("stroke", C.axis));
  styleAxis(yAxis);

  if (opts.axisTitle) addYAxisTitle(svg, 16, margin.top + innerH / 2, opts.axisTitle);

  animateGrow(g);

  renderLegend(legendId, rt.series.map((s) => ({ label: s.label, color: s.color })));
}

async function initTransferOutputChart() {
  const rt = (await getData()).realTransfer;
  drawTransferBars("transfer-output-chart", "transfer-cost-legend", rt, {
    values: rt.outputTokens,
    yMax: 1.5,
    yTicks: 3,
    tickFmt: (d) => d,
    axisTitle: "",
    barFmt: (v) => v,
    tipFmt: (d) => `<strong>${d.label}</strong><br>${d.task}: <b>${rt.outputTokens[d.sid][d.idx]}M output tokens</b>`,
  });
}

async function initTransferTokensChart() {
  const rt = (await getData()).realTransfer;
  drawTransferBars("transfer-tokens-chart", "transfer-cost-legend", rt, {
    values: rt.totalTokens,
    yMax: 350,
    yTicks: 5,
    tickFmt: (d) => d,
    axisTitle: "",
    barFmt: (v) => v,
    tipFmt: (d) => `<strong>${d.label}</strong><br>${d.task}: <b>${rt.totalTokens[d.sid][d.idx]}M tokens</b>`,
  });
}

async function initTransferSuccessChart() {
  const rt = (await getData()).realTransfer;
  drawTransferBars("transfer-success-chart", "transfer-success-legend", rt, {
    values: rt.successPct,
    yMax: 100,
    yTicks: 5,
    tickFmt: (d) => d + "%",
    axisTitle: "",
    barFmt: (v, ti, sid) => rt.successRaw[sid][ti],
    tipFmt: (d) => `<strong>${d.label}</strong><br>${d.task}: <b>${rt.successRaw[d.sid][d.idx]}</b>`,
  });
}

/* ─────────────────────── Init / resize ─────────────────────── */
const CHART_JOBS = [
  ["libero-chart", initLiberoChart],
  ["robosuite-chart", initRobosuiteChart],
  ["behavior-chart", initBehaviorChart],
  ["zeroshot-chart", initZeroshotChart],
  ["transfer-output-chart", initTransferOutputChart],
  ["transfer-tokens-chart", initTransferTokensChart],
  ["transfer-success-chart", initTransferSuccessChart],
  ["ablation-chart", initAblationChart],
];

// Render each chart the first time it scrolls into view, so its entrance
// animation plays where the visitor can see it. Re-runs on resize.
let chartObserver = null;
function initAllCharts() {
  if (chartObserver) { chartObserver.disconnect(); chartObserver = null; }
  const draw = (fn) =>
    Promise.resolve(fn()).catch((err) => console.error("Chart init failed:", err));
  if (!("IntersectionObserver" in window)) {
    CHART_JOBS.forEach(([, fn]) => draw(fn));
    return;
  }
  chartObserver = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const job = CHART_JOBS.find(([id]) => document.getElementById(id) === e.target);
        if (job) draw(job[1]);
        obs.unobserve(e.target);
      });
    },
    { threshold: 0.2 }
  );
  CHART_JOBS.forEach(([id]) => {
    const el = document.getElementById(id);
    if (el) chartObserver.observe(el);
  });
}

let resizeTimer;
let lastChartW = window.innerWidth;
window.addEventListener("resize", () => {
  // Only re-render on a real WIDTH change. On mobile, scrolling shows/hides the
  // browser URL bar, which fires resize with a height-only change — ignore those
  // so the charts don't keep re-rendering (and replaying their animation).
  if (window.innerWidth === lastChartW) return;
  lastChartW = window.innerWidth;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(initAllCharts, 250);
});
