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
     6. Results — delegates to window.VBT (static/js/analysis-view.js)
   ========================================================================= */

(function () {
  "use strict";

  /* ------------------------------- 1. Helpers ------------------------------ */
  const $ = (id) => document.getElementById(id);

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

      lastAnalysis = data.analysis;
      const panel = $("analysisPanel");
      panel.classList.remove("hidden");
      window.VBT.renderAnalysisInto(panel, data.analysis);
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

  /* ----- 6. Results: rendering lives in window.VBT (analysis-view.js) ----- */
  window.addEventListener("resize", () => {
    if (lastAnalysis && lastAnalysis.calibrated)
      window.VBT.drawVelocityChart($("velChart"), lastAnalysis);
  });
})();
