/* =========================================================================
   VBT Studio — shared analysis renderer (window.VBT)

   Renders an analysis object (summary cards, calibration line, per-rep table,
   velocity chart) into ANY container, so both the single-clip page (app.js) and
   the bulk-import per-set review (batch.js) draw identical results.

   The container ("root") is located by id/selector by the caller; inside it the
   renderer finds its slots by [data-role=...]:
     warn | body | summary | calib | repTable (a <table> with <tbody>) | chart (<canvas>)

   This file is a verbatim port of the chart/table code that used to live in
   app.js — behaviour is unchanged, only the DOM lookups are scoped to `root`.
   ========================================================================= */
(function () {
  "use strict";

  const css = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function metricCard(k, v, unit, cls) {
    return (
      `<div class="metric ${cls || ""}">` +
      `<div class="k">${k}</div>` +
      `<div class="v">${v}<span class="u">${unit || ""}</span></div>` +
      `</div>`
    );
  }

  function niceStep(raw) {
    const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  }

  /* Render `a` (analysis dict) into `root`. Returns nothing; safe to re-call. */
  function renderAnalysisInto(root, a) {
    if (!root) return;
    const q = (role) => root.querySelector(`[data-role="${role}"]`);
    const warn = q("warn");
    const body = q("body");

    if (!a || !a.calibrated) {
      if (body) body.classList.add("hidden");
      if (warn) {
        warn.classList.remove("hidden");
        warn.textContent =
          "Not enough confident detections to calibrate — no velocity metrics for this clip.";
      }
      return;
    }
    if (warn) warn.classList.add("hidden");
    if (body) body.classList.remove("hidden");

    // ---- summary metric cards ----
    const s = a.summary || {};
    const styleLabel =
      {
        "down-first": "eccentric → concentric (squat-style)",
        "up-first": "concentric → eccentric (deadlift-style)",
        auto: "auto-detected",
      }[a.rep_style] || "—";

    let cards = metricCard("Reps", a.rep_count, "");
    if (s.avg_mcv != null) {
      cards +=
        metricCard("Avg MCV", s.avg_mcv, " m/s", "accent") +
        metricCard("Best MCV", s.best_mcv, " m/s", "good") +
        metricCard("Velocity loss", s.velocity_loss_pct, "%");
    }
    const summary = q("summary");
    if (summary) summary.innerHTML = cards;

    // ---- calibration line ----
    const calib = q("calib");
    if (calib) {
      const mmPerPx = (a.m_per_px * 1000).toFixed(2);
      calib.innerHTML =
        `Structure <b>${styleLabel}</b>` +
        (s.last_mcv != null ? ` · Last MCV <b>${s.last_mcv} m/s</b>` : "") +
        ` · Calibration: plate <b>${a.plate_diameter_mm} mm</b> ` +
        `&approx; <b>${a.plate_size_px} px</b> &rarr; <b>${mmPerPx} mm/px</b>`;
    }

    // ---- per-rep table ----
    const table = q("repTable");
    if (table) {
      const tbody = table.querySelector("tbody");
      tbody.innerHTML = "";
      const best = Math.max(...a.reps.map((r) => r.mean_concentric_v), -Infinity);
      for (const r of a.reps) {
        const tr = document.createElement("tr");
        if (r.mean_concentric_v === best) tr.className = "best";
        tr.innerHTML =
          `<td>${r.rep}</td><td>${r.duration_s.toFixed(2)}</td>` +
          `<td>${r.rom_m.toFixed(2)}</td><td>${r.mean_concentric_v.toFixed(2)}</td>` +
          `<td>${r.peak_concentric_v.toFixed(2)}</td><td>${r.mean_eccentric_v.toFixed(2)}</td>` +
          `<td>${r.peak_eccentric_v.toFixed(2)}</td>`;
        tbody.appendChild(tr);
      }
      const wrap = table.closest(".table-wrap");
      if (wrap) wrap.classList.toggle("hidden", a.reps.length === 0);
    }

    const chart = q("chart");
    if (chart) drawVelocityChart(chart, a);
  }

  /* --------------------- Velocity chart (canvas) ------------------------ */
  /* Faithful port of the original hand-rolled chart, re-coloured to the
     light palette via CSS variables. Phase bands behind the curve, rep
     markers along the top, dpr/retina aware, axis labels, +up velocity. */
  function drawVelocityChart(canvas, a) {
    const t = a.series.t,
      v = a.series.v,
      phase = a.series.phase,
      rep = a.series.rep;
    if (!t || t.length < 2) return;

    const C = {
      line: css("--accent") || "#2563eb",
      grid: css("--line") || "#e6e8ec",
      zero: css("--muted-2") || "#9aa1ad",
      text: css("--muted") || "#6b7280",
      bandUp: css("--phase-up-band") || "rgba(22,163,74,.10)",
      bandDown: css("--phase-down-band") || "rgba(234,138,12,.12)",
    };

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth,
      H = 260;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const padL = 46,
      padR = 12,
      padT = 24,
      padB = 26;
    const t0 = t[0],
      t1 = t[t.length - 1];
    let vMin = Math.min(0, ...v),
      vMax = Math.max(0, ...v);
    const span = vMax - vMin || 1;
    vMin -= span * 0.08;
    vMax += span * 0.08;
    const X = (tt) => padL + ((tt - t0) / (t1 - t0 || 1)) * (W - padL - padR);
    const Y = (vv) => padT + ((vMax - vv) / (vMax - vMin)) * (H - padT - padB);

    // ---- phase bands ----
    const bandColor = { up: C.bandUp, down: C.bandDown };
    let i = 0;
    while (i < t.length) {
      let j = i;
      while (j + 1 < t.length && phase[j + 1] === phase[i]) j++;
      if (bandColor[phase[i]]) {
        ctx.fillStyle = bandColor[phase[i]];
        ctx.fillRect(
          X(t[i]),
          padT,
          Math.max(X(t[j]) - X(t[i]), 1),
          H - padT - padB
        );
      }
      i = j + 1;
    }

    // ---- rep numbers along the top (at each rep's midpoint) ----
    ctx.fillStyle = C.text;
    ctx.font = "11px " + css("--font");
    ctx.textAlign = "center";
    const seen = new Set();
    for (let k = 0; k < t.length; k++) {
      if (rep[k] > 0 && !seen.has(rep[k])) {
        seen.add(rep[k]);
        let kEnd = k;
        while (kEnd + 1 < t.length && rep[kEnd + 1] === rep[k]) kEnd++;
        ctx.fillText(`R${rep[k]}`, X((t[k] + t[kEnd]) / 2), padT - 9);
      }
    }

    // ---- y grid + ticks ----
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    const step = niceStep((vMax - vMin) / 5);
    for (let vv = Math.ceil(vMin / step) * step; vv <= vMax; vv += step) {
      const y = Y(vv);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.fillText(vv.toFixed(Math.abs(step) < 0.1 ? 2 : 1), padL - 7, y + 3);
    }

    // ---- zero line (dashed) ----
    ctx.strokeStyle = C.zero;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, Y(0));
    ctx.lineTo(W - padR, Y(0));
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- x ticks ----
    ctx.fillStyle = C.text;
    ctx.textAlign = "center";
    const tStep = t1 - t0 > 20 ? 5 : 1;
    for (let tt = Math.ceil(t0); tt <= t1; tt += tStep)
      ctx.fillText(`${tt}s`, X(tt), H - 7);

    // ---- velocity polyline ----
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let k = 0; k < t.length; k++)
      k === 0 ? ctx.moveTo(X(t[k]), Y(v[k])) : ctx.lineTo(X(t[k]), Y(v[k]));
    ctx.stroke();

    // ---- y-axis label ----
    ctx.save();
    ctx.fillStyle = C.text;
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("m/s", 0, 0);
    ctx.restore();
  }

  window.VBT = window.VBT || {};
  window.VBT.renderAnalysisInto = renderAnalysisInto;
  window.VBT.drawVelocityChart = drawVelocityChart;
  window.VBT.metricCard = metricCard;
  window.VBT.niceStep = niceStep;
})();
