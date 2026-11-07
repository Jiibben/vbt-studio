/* =========================================================================
   VBT Studio — bulk import (sessions)
   Vanilla JS, no frameworks. Flow:
     1. pick/drop many clips -> instant client-side thumbnails (no upload yet)
     2. arrange: assign exercises, reorder, per-clip overrides
     3. "Process all" -> POST /sessions, upload each clip, POST .../process
     4. poll GET /sessions/<sid> -> live per-set status -> session review
   Result rendering is shared with the single-clip page via window.VBT.
   ========================================================================= */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* -------------------------------- state -------------------------------- */
  let uid = 0;
  const sets = [];                 // {key, file, url, duration, exerciseId, overrides:{}, serverId, els}
  const exercises = [];            // {id, name}
  let exCounter = 0;
  let sessionId = null;
  let pollTimer = null;

  const grid = $("setGrid");
  const arrangeCard = $("arrangeCard");
  const reviewCard = $("reviewCard");

  function setStatus(html, kind) {
    const el = $("status");
    el.className = "status" + (kind ? " is-" + kind : "");
    el.innerHTML = html;
  }

  /* --------------------------- confidence slider ------------------------- */
  const confEl = $("conf");
  const syncConf = () => { $("confVal").textContent = Number(confEl.value).toFixed(2); };
  confEl.addEventListener("input", syncConf);
  syncConf();

  /* ------------------------------ file intake ---------------------------- */
  $("files").addEventListener("change", (e) => addFiles(e.target.files));

  const dz = $("dropzone");
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
  dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  function addFiles(fileList) {
    const vids = [...fileList].filter((f) => f.type.startsWith("video/"));
    if (!vids.length) return;
    for (const file of vids) {
      const s = {
        key: ++uid,
        file,
        url: URL.createObjectURL(file),
        duration: 0,
        exerciseId: exercises[0] ? exercises[0].id : null,
        overrides: {},
        serverId: null,
        els: {},
      };
      sets.push(s);
      buildCard(s);
    }
    arrangeCard.classList.remove("hidden");
    renderOrder();
  }

  /* ------------------------------ exercises ------------------------------ */
  $("addExercise").addEventListener("click", () => {
    const name = (prompt("Exercise name (e.g. Squat):") || "").trim();
    if (!name) return;
    exercises.push({ id: "ex" + ++exCounter, name });
    renderExerciseChips();
    sets.forEach(fillExerciseSelect);
  });

  function renderExerciseChips() {
    $("exerciseChips").innerHTML =
      exercises.map((e) => `<span class="chip">${escapeHtml(e.name)}</span>`).join("") ||
      `<span class="muted">none yet</span>`;
  }

  function fillExerciseSelect(s) {
    const sel = s.els.exercise;
    if (!sel) return;
    const cur = s.exerciseId;
    sel.innerHTML =
      `<option value="">— unassigned —</option>` +
      exercises.map((e) => `<option value="${e.id}">${escapeHtml(e.name)}</option>`).join("");
    sel.value = cur && exercises.some((e) => e.id === cur) ? cur : "";
    s.exerciseId = sel.value || null;
  }

  /* ------------------------------- cards --------------------------------- */
  function buildCard(s) {
    const card = document.createElement("div");
    card.className = "set-card";
    card.innerHTML = `
      <div class="set-thumb">
        <video muted playsinline preload="metadata"></video>
        <span class="badge" data-role="badge">ready</span>
      </div>
      <div class="set-meta">
        <div class="set-name" title="${escapeHtml(s.file.name)}">${escapeHtml(s.file.name)}</div>
        <div class="set-row">
          <select data-role="exercise" title="Exercise"></select>
          <div class="reorder">
            <button class="btn btn-ghost btn-sm" data-role="up" title="Move up">↑</button>
            <button class="btn btn-ghost btn-sm" data-role="down" title="Move down">↓</button>
            <button class="btn btn-ghost btn-sm" data-role="remove" title="Remove">✕</button>
          </div>
        </div>
        <details class="set-overrides">
          <summary>Overrides</summary>
          <div class="set-row">
            <label class="field-label">Plate
              <select data-role="plate">
                <option value="">default</option>
                ${["25","20","15","10","5","2.5"].map((k)=>`<option value="${k}">${k} kg</option>`).join("")}
              </select>
            </label>
            <label class="field-label">Rep structure
              <select data-role="rep">
                <option value="">default</option>
                <option value="auto">auto</option>
                <option value="down-first">eccentric first</option>
                <option value="up-first">concentric first</option>
              </select>
            </label>
          </div>
          <div class="set-row trim-row">
            <label class="field-label">Trim start
              <input data-role="start" type="number" min="0" step="0.01" placeholder="0">
            </label>
            <button class="btn btn-ghost btn-sm" data-role="setStart" type="button">⇠ playhead</button>
            <label class="field-label">Trim end
              <input data-role="end" type="number" min="0" step="0.01" placeholder="full">
            </label>
            <button class="btn btn-ghost btn-sm" data-role="setEnd" type="button">playhead ⇢</button>
          </div>
        </details>
        <div class="set-progress hidden" data-role="progresswrap">
          <div class="bar"><span data-role="bar"></span></div>
        </div>
      </div>`;
    grid.appendChild(card);

    const q = (r) => card.querySelector(`[data-role="${r}"]`);
    const video = card.querySelector("video");
    video.src = s.url;
    video.addEventListener("loadedmetadata", () => {
      s.duration = video.duration;
      makeThumb(video);
    });

    s.els = {
      card, video, badge: q("badge"), exercise: q("exercise"),
      progresswrap: q("progresswrap"), bar: q("bar"),
    };
    fillExerciseSelect(s);

    q("exercise").addEventListener("change", (e) => { s.exerciseId = e.target.value || null; });
    q("plate").addEventListener("change", (e) => setOverride(s, "plate", e.target.value || null));
    q("rep").addEventListener("change", (e) => setOverride(s, "rep_style", e.target.value || null));
    q("start").addEventListener("change", () => updateTrim(s, q));
    q("end").addEventListener("change", () => updateTrim(s, q));
    q("setStart").addEventListener("click", () => { q("start").value = s.els.video.currentTime.toFixed(2); updateTrim(s, q); });
    q("setEnd").addEventListener("click", () => { q("end").value = s.els.video.currentTime.toFixed(2); updateTrim(s, q); });
    q("up").addEventListener("click", () => move(s, -1));
    q("down").addEventListener("click", () => move(s, +1));
    q("remove").addEventListener("click", () => removeSet(s));
  }

  function setOverride(s, key, val) {
    if (val == null) delete s.overrides[key];
    else s.overrides[key] = val;
  }

  function updateTrim(s, q) {
    const start = parseFloat(q("start").value) || 0;
    const end = parseFloat(q("end").value) || 0;
    if (end > start) s.overrides.trim = { start, end };
    else delete s.overrides.trim;
  }

  function makeThumb(video) {
    const seekTo = Math.min(1, (video.duration || 2) / 2);
    const onSeek = () => {
      try {
        const c = document.createElement("canvas");
        c.width = 160; c.height = Math.round(160 * (video.videoHeight / video.videoWidth || 0.56));
        c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
        video.poster = c.toDataURL("image/jpeg", 0.6);
      } catch (e) { /* cross-origin / decode quirk — leave the live frame */ }
      video.removeEventListener("seeked", onSeek);
    };
    video.addEventListener("seeked", onSeek);
    try { video.currentTime = seekTo; } catch (e) { /* ignore */ }
  }

  function move(s, dir) {
    const i = sets.indexOf(s);
    const j = i + dir;
    if (j < 0 || j >= sets.length) return;
    sets[i] = sets[j]; sets[j] = s;
    renderOrder();
  }

  function removeSet(s) {
    const i = sets.indexOf(s);
    if (i < 0) return;
    URL.revokeObjectURL(s.url);
    s.els.card.remove();
    sets.splice(i, 1);
    if (!sets.length) arrangeCard.classList.add("hidden");
    renderOrder();
  }

  function renderOrder() {
    sets.forEach((s, i) => {
      grid.appendChild(s.els.card);           // re-append in array order
      s.els.card.querySelector(".set-name").dataset.idx = i + 1;
    });
  }

  /* --------------------------- process the batch ------------------------- */
  $("process").addEventListener("click", processAll);

  async function processAll() {
    if (!sets.length || sessionId) return;
    $("process").disabled = true;
    setStatus('<span class="spinner"></span> Creating session…');
    reviewCard.classList.add("hidden");

    const payload = {
      title: $("sessionTitle").value || "session",
      defaults: {
        model: $("model").value,
        conf: parseFloat(confEl.value),
        plate: $("plate").value,
        trail_style: $("trailStyle").value,
        rep_style: $("repStyle").value,
      },
      exercises: exercises.map((e) => ({ id: e.id, name: e.name })),
      sets: sets.map((s, i) => ({
        original_filename: s.file.name,
        exercise_id: s.exerciseId,
        order: i,
        overrides: s.overrides,
      })),
    };

    try {
      const resp = await fetch("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Could not create session.");
      sessionId = data.session_id;

      // server returns sets in the same order we sent them
      data.sets.forEach((srv, i) => { sets[i].serverId = srv.id; sets[i].uploadUrl = srv.upload_url; });

      // upload sequentially with per-card progress
      for (let i = 0; i < sets.length; i++) {
        setBadge(sets[i], "uploading", "is-processing");
        showBar(sets[i]);
        await uploadOne(sets[i]);
        setBadge(sets[i], "uploaded", "");
        hideBar(sets[i]);
      }

      setStatus('<span class="spinner"></span> Processing in the background…');
      const pr = await fetch(`/sessions/${sessionId}/process`, { method: "POST" });
      const pd = await pr.json();
      if (!pr.ok) throw new Error(pd.error || "Could not start processing.");
      poll();
    } catch (e) {
      setStatus(`<span class="ico">&#10007;</span> ${e.message}`, "err");
      $("process").disabled = false;
      sessionId = null;
    }
  }

  function uploadOne(s) {
    return new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append("video", s.file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", s.uploadUrl);
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) setBar(s, ev.loaded / ev.total);
      };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300)
        ? resolve() : reject(new Error("Upload failed for " + s.file.name));
      xhr.onerror = () => reject(new Error("Upload failed for " + s.file.name));
      xhr.send(fd);
    });
  }

  /* ------------------------------- polling ------------------------------- */
  function poll() {
    clearInterval(pollTimer);
    pollTimer = setInterval(refresh, 1500);
    refresh();
  }

  async function refresh() {
    try {
      const resp = await fetch(`/sessions/${sessionId}`);
      if (!resp.ok) return;
      const m = await resp.json();
      applyManifest(m);
      if (["done", "error", "interrupted"].includes(m.state)) {
        clearInterval(pollTimer);
        const ok = m.state === "done";
        setStatus(ok ? '<span class="ico">&#10003;</span> All sets processed.'
                     : '<span class="ico">&#9888;</span> Finished with some errors.',
                  ok ? "ok" : "err");
        renderReview(m);
      }
    } catch (e) { /* transient; next tick retries */ }
  }

  function applyManifest(m) {
    const byId = {};
    m.sets.forEach((s) => (byId[s.id] = s));
    sets.forEach((local) => {
      const srv = byId[local.serverId];
      if (!srv) return;
      const cls = { processing: "is-processing", done: "is-ok",
                    error: "is-err", uploaded: "" }[srv.status] || "";
      setBadge(local, srv.status, cls);
    });
  }

  /* ------------------------------- review -------------------------------- */
  async function renderReview(m) {
    reviewCard.classList.remove("hidden");
    const root = $("review");
    root.innerHTML = "";

    const exName = {};
    (m.exercises || []).forEach((e) => (exName[e.id] = e.name));
    const exOrder = {};
    (m.exercises || []).forEach((e, i) => (exOrder[e.id] = e.order != null ? e.order : i));
    const ordered = [...m.sets].sort((a, b) =>
      (exOrder[a.exercise_id] ?? 0) - (exOrder[b.exercise_id] ?? 0) ||
      (a.order ?? 0) - (b.order ?? 0));

    let lastEx = " ";
    for (const s of ordered) {
      const ex = s.exercise_id ? (exName[s.exercise_id] || s.exercise_id) : "Unassigned";
      if (ex !== lastEx) {
        const h = document.createElement("h3");
        h.className = "review-exercise";
        h.textContent = ex;
        root.appendChild(h);
        lastEx = ex;
      }
      root.appendChild(await buildReviewSet(m.session_id, s));
    }
  }

  async function buildReviewSet(sid, s) {
    const block = document.createElement("div");
    block.className = "review-set";
    const title = escapeHtml(s.original_filename || s.id);

    if (s.status !== "done" || !s.result) {
      block.innerHTML = `<div class="review-head"><b>${title}</b>
        <span class="badge ${s.status === "error" ? "is-err" : ""}">${s.status}</span></div>
        ${s.error ? `<p class="notice warn">${escapeHtml(s.error)}</p>` : ""}`;
      return block;
    }

    const r = s.result;
    const media = (name) => `/media/${sid}/${name}`;
    block.innerHTML = `
      <div class="review-head"><b>${title}</b> <span class="badge is-ok">done</span></div>
      <div class="grid cols-2">
        <div><video src="${media(r.boxed_name)}" controls loop playsinline muted></video></div>
        <div><video src="${media(r.barpath_name)}" controls loop playsinline muted></video></div>
      </div>
      <div class="analysis-root" data-analysis>
        <div data-role="body">
          <div class="metric-grid" data-role="summary"></div>
          <p class="calib-line" data-role="calib"></p>
          <div class="table-wrap">
            <table class="rep-table" data-role="repTable">
              <thead><tr>
                <th>Rep</th><th>Duration (s)</th><th>ROM (m)</th>
                <th>Mean conc.</th><th>Peak conc.</th><th>Mean ecc.</th><th>Peak ecc.</th>
              </tr></thead><tbody></tbody>
            </table>
          </div>
          <div class="chart-wrap"><canvas height="260" data-role="chart"></canvas></div>
        </div>
        <p class="notice warn hidden" data-role="warn"></p>
      </div>`;

    try {
      const resp = await fetch(media(r.metrics_name));
      const analysis = await resp.json();
      window.VBT.renderAnalysisInto(block.querySelector("[data-analysis]"), analysis);
    } catch (e) { /* metrics missing — leave video previews only */ }
    return block;
  }

  /* ------------------------------- helpers ------------------------------- */
  function setBadge(s, text, cls) {
    s.els.badge.className = "badge " + (cls || "");
    s.els.badge.textContent = text;
  }
  function showBar(s) { s.els.progresswrap.classList.remove("hidden"); }
  function hideBar(s) { s.els.progresswrap.classList.add("hidden"); }
  function setBar(s, frac) { s.els.bar.style.width = Math.round(frac * 100) + "%"; }

  function escapeHtml(t) {
    return String(t).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
