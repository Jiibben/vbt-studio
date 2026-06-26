/* =========================================================================
   VBT Studio — client logic
   Vanilla JS, no frameworks/CDNs. Talks to the fixed Flask backend:
     POST /track          (multipart)        -> results + analysis
     GET/POST /plate-config (json)           -> plate diameters
   Sections:
     1. Helpers
     2. Import & trim
     3. Confidence slider
     4. Track submission
     5. Plate-config editor
     6. Results / summary / rep table
     7. Velocity chart (canvas, faithful port)
   ========================================================================= */

(function () {
  "use strict";

  /* ------------------------------- 1. Helpers ------------------------------ */
  const $ = (id) => document.getElementById(id);
  const css = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const fileEl   = $("file");
  const video    = $("preview");
  const player   = $("player");
  const startEl  = $("start");
  const endEl    = $("end");
  const runBtn   = $("run");
  const statusEl = $("status");
  const progress = $("progress");
  const results  = $("results");

  let objectUrl = null;
  let lastAnalysis = null;

  function setStatus(html, kind) {
    statusEl.className = "status" + (kind ? " is-" + kind : "");
    statusEl.innerHTML = html;
  }

  /* --------------------------- 2. Import & trim ---------------------------- */
  fileEl.addEventListener("change", () => {
    const f = fileEl.files[0];
    if (!f) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    video.src = objectUrl;
    player.classList.remove("hidden");
    runBtn.disabled = false;
    setStatus("");
  });

  video.addEventListener("loadedmetadata", () => {
    endEl.value = video.duration.toFixed(2);
    startEl.value = "0";
    startEl.max = endEl.max = video.duration.toFixed(2);
    $("dur").innerHTML = `Duration: <b>${video.duration.toFixed(2)} s</b>`;
  });

  $("setStart").addEventListener("click", () => {
    startEl.value = video.currentTime.toFixed(2);
  });
  $("setEnd").addEventListener("click", () => {
    endEl.value = video.currentTime.toFixed(2);
  });
  $("resetTrim").addEventListener("click", () => {
    startEl.value = "0";
    endEl.value = video.duration ? video.duration.toFixed(2) : "0";
    setStatus("");
  });

  /* ------------------------ 3. Confidence slider --------------------------- */
  const confEl = $("conf");
  const confVal = $("confVal");
  const syncConf = () => { confVal.textContent = Number(confEl.value).toFixed(2); };
  confEl.addEventListener("input", syncConf);
  syncConf();

  /* ------------------------- 4. Track submission --------------------------- */
  runBtn.addEventListener("click", async () => {
    const f = fileEl.files[0];
    if (!f) return;

    const start = parseFloat(startEl.value) || 0;
    const end = parseFloat(endEl.value) || 0;
    if (end > 0 && end <= start) {
      setStatus('<span class="ico">&#10007;</span> End must be greater than start.', "err");
      return;
    }

    const fd = new FormData();
    fd.append("video", f);
    fd.append("start", start);
    fd.append("end", end);
    fd.append("model", $("model").value);
    fd.append("conf", confEl.value);
    fd.append("plate", $("plate").value);
    fd.append("trail_style", $("trailStyle").value);
    fd.append("rep_style", $("repStyle").value);

    runBtn.disabled = true;
    progress.classList.remove("hidden");
    setStatus(
      '<span class="spinner"></span> Trimming &amp; tracking… running the model frame-by-frame, give it a moment.'
    );
    results.classList.add("hidden");

    try {
      const resp = await fetch("/track", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Tracking failed.");

      const bust = "?t=" + Date.now(); // dodge the cache between runs
      $("boxedVid").src = data.boxed_url + bust;
      $("barpathVid").src = data.barpath_url + bust;
      results.classList.remove("hidden");

      $("resultStats").innerHTML =
        `Saved to <b>${data.saved_dir}/</b> — ` +
        `<span class="mono">${data.boxed_name}</span>, ` +
        `<span class="mono">${data.barpath_name}</span>, ` +
        `<span class="mono">${data.csv_name}</span>, ` +
        `<span class="mono">${data.metrics_name}</span>.<br>` +
        `fps <b>${data.fps}</b> · frames <b>${data.n_frames}</b> · ` +
        `box coverage <b>${data.coverage}%</b> · ` +
        `track ids <b>${JSON.stringify(data.track_ids)}</b>`;

      renderAnalysis(data.analysis);
      setStatus('<span class="ico">&#10003;</span> Done.', "ok");
    } catch (e) {
      setStatus(`<span class="ico">&#10007;</span> ${e.message}`, "err");
    } finally {
      runBtn.disabled = false;
      progress.classList.add("hidden");
    }
  });

  /* ------------------------ 5. Plate-config editor ------------------------- */
  $("savePlates").addEventListener("click", async () => {
    const cfg = {};
    document
      .querySelectorAll(".plate-mm")
      .forEach((el) => (cfg[el.dataset.kg] = parseFloat(el.value)));

    const st = $("plateStatus");
    st.className = "status";
    st.innerHTML = '<span class="spinner"></span> Saving…';

    try {
      const resp = await fetch("/plate-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Save failed.");

      // Refresh the calibration dropdown labels with the new diameters.
      document.querySelectorAll("#plate option").forEach((opt) => {
        opt.textContent = `${opt.value} kg — ${Math.round(data[opt.value])} mm`;
      });
      st.className = "status is-ok";
      st.innerHTML = '<span class="ico">&#10003;</span> Saved';
    } catch (e) {
      st.className = "status is-err";
      st.innerHTML = `<span class="ico">&#10007;</span> ${e.message}`;
    }
  });

  /* -------------------- 6. Results / summary / rep table ------------------- */
  function metricCard(k, v, unit, cls) {
    return (
      `<div class="metric ${cls || ""}">` +
      `<div class="k">${k}</div>` +
      `<div class="v">${v}<span class="u">${unit || ""}</span></div>` +
      `</div>`
    );
  }

  function renderAnalysis(a) {
    lastAnalysis = a;
    const panel = $("analysisPanel");
    const warn = $("analysisWarn");
    const body = $("analysisBody");
    panel.classList.remove("hidden");

    if (!a || !a.calibrated) {
      body.classList.add("hidden");
      warn.classList.remove("hidden");
      warn.textContent =
        "Not enough confident detections to calibrate — no velocity metrics for this clip.";
      return;
    }
    warn.classList.add("hidden");
    body.classList.remove("hidden");

    // ---- summary metric cards ----
    const s = a.summary || {};
    const styleLabel =
      {
        "down-first": "eccentric → concentric (squat-style)",
        "up-first": "concentric → eccentric (deadlift-style)",
        auto: "auto-detected",
      }[a.rep_style] || "—";

    let cards =
      metricCard("Reps", a.rep_count, "") ;
    if (s.avg_mcv != null) {
      cards +=
        metricCard("Avg MCV", s.avg_mcv, " m/s", "accent") +
        metricCard("Best MCV", s.best_mcv, " m/s", "good") +
        metricCard("Velocity loss", s.velocity_loss_pct, "%");
    }
    $("summaryCards").innerHTML = cards;

    // ---- calibration line ----
    const mmPerPx = (a.m_per_px * 1000).toFixed(2);
    $("calibLine").innerHTML =
      `Structure <b>${styleLabel}</b>` +
      (s.last_mcv != null ? ` · Last MCV <b>${s.last_mcv} m/s</b>` : "") +
      ` · Calibration: plate <b>${a.plate_diameter_mm} mm</b> ` +
      `&approx; <b>${a.plate_size_px} px</b> &rarr; <b>${mmPerPx} mm/px</b>`;

    // ---- per-rep table ----
    const tbody = document.querySelector("#repTable tbody");
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
    document
      .querySelector("#repTable")
      .closest(".table-wrap")
      .classList.toggle("hidden", a.reps.length === 0);

    drawVelocityChart($("velChart"), a);
  }

  /* --------------------- 7. Velocity chart (canvas) ------------------------ */
  /* Faithful port of the original hand-rolled chart, re-coloured to the
     light palette via CSS variables. Phase bands behind the curve, rep
     markers along the top, dpr/retina aware, axis labels, +up velocity. */
  function drawVelocityChart(canvas, a) {
    const t = a.series.t,
      v = a.series.v,
      phase = a.series.phase,
      rep = a.series.rep;
    if (!t || t.length < 2) return;

    // palette pulled from CSS variables so it re-themes for free
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

  function niceStep(raw) {
    const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  }

  window.addEventListener("resize", () => {
    if (lastAnalysis && lastAnalysis.calibrated)
      drawVelocityChart($("velChart"), lastAnalysis);
  });
})();
