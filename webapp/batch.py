"""
batch.py — bulk-import web layer for VBT Studio.

Adds a *session* concept on top of the single-clip ``/track`` flow: import many
clips, arrange them into Session → Exercise → Sets, then process them in the
background (one worker thread, the YOLO model loaded once per batch via
``vbt_core.run_batch``) with per-set status polled from disk.

Like ``app.py`` this is the web layer only — every bit of CV lives in
``vbt_core``. State is a single ``manifest.json`` per session folder (no DB); the
in-memory queue is just a scheduler and is fully rebuildable from the manifest.

Routes (Flask Blueprint ``batch_bp``):
    GET    /batch                              the bulk-import page
    GET    /sessions                           list sessions (scan manifests)
    POST   /sessions                           create from arrangement JSON
    GET    /sessions/<sid>                     full manifest (frontend polls)
    POST   /sessions/<sid>/sets/<set_id>/video upload one raw clip
    POST   /sessions/<sid>/process             enqueue for background processing
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

from flask import (Blueprint, abort, jsonify, render_template, request,
                   url_for)
from werkzeug.utils import secure_filename

from vbt_core import BatchItem, load_plate_config, run_batch
from vbt_core.rendering import trim_clip

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TRACKED_DIR = ROOT / "data" / "tracked"      # tests monkeypatch this

SCHEMA_VERSION = 1
VALID_TRAIL = ("green", "phase")
VALID_REP = ("auto", "down-first", "up-first")

batch_bp = Blueprint("batch", __name__)


# --------------------------------------------------------------------------- #
# Small helpers (kept local so this module never imports app.py).
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _list_models() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.glob("*.pt"))


def _default_model() -> str:
    models = _list_models()
    for preferred in ("plate_best.pt", "best.pt"):
        if preferred in models:
            return preferred
    return models[0] if models else ""


def _slugify(text: str) -> str:
    return secure_filename((text or "").strip().lower().replace(" ", "-"))[:40]


def _session_dir(sid: str) -> Path:
    return TRACKED_DIR / sid


def _manifest_path(sid: str) -> Path:
    return _session_dir(sid) / "manifest.json"


# --------------------------------------------------------------------------- #
# Manifest IO — atomic writes + per-session lock (mirrors plate_config).
# --------------------------------------------------------------------------- #
_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_alloc_lock = threading.Lock()


def _lock_for(sid: str) -> threading.RLock:
    with _locks_guard:
        lk = _locks.get(sid)
        if lk is None:
            lk = _locks[sid] = threading.RLock()
        return lk


def read_manifest(sid: str) -> dict | None:
    p = _manifest_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write_manifest_atomic(sid: str, manifest: dict) -> None:
    d = _session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _now_iso()
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, _manifest_path(sid))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_manifest(sid: str, mutate) -> dict | None:
    """Locked read-modify-write: ``mutate(manifest)`` edits in place."""
    with _lock_for(sid):
        manifest = read_manifest(sid)
        if manifest is None:
            return None
        mutate(manifest)
        write_manifest_atomic(sid, manifest)
        return manifest


def allocate_session_id(title: str) -> str:
    """Date (+ title slug), suffixed -2, -3 … on collision. Claims the dir."""
    base = date.today().isoformat()
    slug = _slugify(title)
    stem = f"{base}-{slug}" if slug else base
    with _alloc_lock:
        candidate, n = stem, 2
        while _session_dir(candidate).exists():
            candidate, n = f"{stem}-{n}", n + 1
        _session_dir(candidate).mkdir(parents=True)
        return candidate


def _find_set(manifest: dict, set_id: str) -> dict | None:
    return next((s for s in manifest["sets"] if s["id"] == set_id), None)


def _set_fields(manifest: dict, set_id: str, **fields) -> None:
    s = _find_set(manifest, set_id)
    if s is not None:
        s.update(fields)


# --------------------------------------------------------------------------- #
# Validation of the arrangement JSON posted to POST /sessions.
# --------------------------------------------------------------------------- #
class _BadRequest(ValueError):
    """Raised for invalid arrangement input -> HTTP 400."""


def _validate_defaults(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise _BadRequest("defaults must be an object.")
    plates = load_plate_config()

    model = raw.get("model") or _default_model()
    if not model or not (MODELS_DIR / model).exists():
        raise _BadRequest(f"Model not found: {model!r}")

    try:
        conf = float(raw.get("conf", 0.25))
    except (TypeError, ValueError):
        raise _BadRequest("conf must be a number.")
    conf = min(1.0, max(0.0, conf))

    plate = str(raw.get("plate", "20"))
    if plate not in plates:
        raise _BadRequest(f"Unknown plate weight: {plate}")

    trail_style = raw.get("trail_style", "green")
    if trail_style not in VALID_TRAIL:
        raise _BadRequest(f"Unknown trail style: {trail_style}")

    rep_style = raw.get("rep_style", "auto")
    if rep_style not in VALID_REP:
        raise _BadRequest(f"Unknown rep style: {rep_style}")

    return {"model": model, "conf": conf, "plate": plate,
            "trail_style": trail_style, "rep_style": rep_style}


def _validate_overrides(raw, plates) -> dict:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise _BadRequest("overrides must be an object.")
    out: dict = {}
    if raw.get("plate") is not None:
        plate = str(raw["plate"])
        if plate not in plates:
            raise _BadRequest(f"Unknown plate weight: {plate}")
        out["plate"] = plate
    if raw.get("rep_style") is not None:
        if raw["rep_style"] not in VALID_REP:
            raise _BadRequest(f"Unknown rep style: {raw['rep_style']}")
        out["rep_style"] = raw["rep_style"]
    trim = raw.get("trim")
    if trim:
        try:
            start, end = float(trim.get("start", 0)), float(trim.get("end", 0))
        except (TypeError, ValueError, AttributeError):
            raise _BadRequest("trim must be {start, end} numbers.")
        if end > start:
            out["trim"] = {"start": start, "end": end}
    return out


def _build_manifest(sid: str, data: dict) -> dict:
    title = str(data.get("title") or "session")
    defaults = _validate_defaults(data.get("defaults") or {})
    plates = load_plate_config()

    exercises = []
    seen_ids = set()
    for i, ex in enumerate(data.get("exercises") or []):
        ex_id = str(ex.get("id") or f"ex{i + 1}")
        exercises.append({"id": ex_id, "name": str(ex.get("name") or ex_id),
                          "order": i})
        seen_ids.add(ex_id)

    sets_in = data.get("sets") or []
    if not isinstance(sets_in, list) or not sets_in:
        raise _BadRequest("At least one set is required.")
    sets = []
    for i, s in enumerate(sets_in):
        ex_id = s.get("exercise_id")
        if ex_id is not None and ex_id not in seen_ids:
            raise _BadRequest(f"Set references unknown exercise: {ex_id}")
        sets.append({
            "id": uuid.uuid4().hex[:8],
            "exercise_id": ex_id,
            "order": int(s.get("order", i)),
            "original_filename": secure_filename(
                str(s.get("original_filename") or f"clip{i}")),
            "raw_filename": None,
            "overrides": _validate_overrides(s.get("overrides"), plates),
            "status": "pending",      # pending -> uploaded -> processing -> done/error
            "progress": 0.0,
            "error": None,
            "result": None,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "title": title,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "state": "uploading",
        "defaults": defaults,
        "exercises": exercises,
        "sets": sets,
    }


# --------------------------------------------------------------------------- #
# Background worker — one daemon thread, FIFO over session ids.
# --------------------------------------------------------------------------- #
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop,
                                       name="vbt-batch-worker", daemon=True)
            _worker.start()


def enqueue(sid: str) -> None:
    _ensure_worker()
    _queue.put(sid)


def _worker_loop() -> None:
    while True:
        sid = _queue.get()
        try:
            process_session(sid)
        except Exception as exc:                       # last-resort safety net
            update_manifest(sid, lambda m: m.update(state="error",
                                                    error=str(exc)))
        finally:
            _queue.task_done()


def _result_payload(item: BatchItem, result) -> dict:
    d = result.analysis.to_dict() if result.analysis else {}
    summary = {"rep_count": d.get("rep_count")}
    summary.update(d.get("summary") or {})
    return {
        "boxed_name": Path(item.out_boxed).name,
        "barpath_name": Path(item.out_barpath).name,
        "csv_name": Path(item.out_csv).name,
        "metrics_name": Path(item.out_metrics).name,
        "fps": round(result.fps, 2),
        "n_frames": result.n_frames,
        "coverage": round(result.coverage, 1),
        "track_ids": result.track_ids,
        "summary": summary,
    }


def process_session(sid: str) -> None:
    """Process every not-yet-done set in `sid`, newest manifest each step.

    Builds BatchItems in (exercise order, set order), applies per-clip trim, then
    runs them through one shared detector via ``run_batch``. Callbacks persist
    each set's status so the polling UI sees live progress. Idempotent: already
    ``done`` sets are skipped, so re-enqueue after a restart just finishes the
    rest.
    """
    manifest = read_manifest(sid)
    if manifest is None:
        return
    update_manifest(sid, lambda m: m.update(state="processing"))

    plates = load_plate_config()
    defaults = manifest["defaults"]
    weights = MODELS_DIR / defaults["model"]
    sdir = _session_dir(sid)
    raw_dir = sdir / "_raw"

    ex_order = {e["id"]: e.get("order", i)
                for i, e in enumerate(manifest.get("exercises", []))}
    todo = [s for s in manifest["sets"] if s["status"] != "done"]
    todo.sort(key=lambda s: (ex_order.get(s.get("exercise_id"), 0),
                             s.get("order", 0)))

    items: list[BatchItem] = []
    set_ids: list[str] = []
    for s in todo:
        ov = s.get("overrides") or {}
        plate = ov.get("plate") or defaults["plate"]
        diameter = plates.get(plate, plates[defaults["plate"]])
        rep_style = ov.get("rep_style") or defaults["rep_style"]

        raw = raw_dir / s["raw_filename"]
        source = raw
        trim = ov.get("trim")
        if trim:
            trimmed = raw_dir / f"{s['id']}_trim.mp4"
            try:
                trim_clip(raw, trimmed, trim["start"], trim["end"])
                source = trimmed
            except Exception:
                source = raw       # fall back to the full clip on trim failure

        sid_ = s["id"]
        items.append(BatchItem(
            source=source,
            out_boxed=sdir / f"{sid_}_tracked.mp4",
            out_barpath=sdir / f"{sid_}_barpath.mp4",
            out_csv=sdir / f"{sid_}_track.csv",
            out_metrics=sdir / f"{sid_}_metrics.json",
            conf=float(defaults["conf"]),
            plate_diameter_mm=float(diameter),
            plate_kg=plate,
            trail_style=defaults["trail_style"],
            rep_style=rep_style,
        ))
        set_ids.append(sid_)

    def on_start(i, item):
        update_manifest(sid, lambda m: _set_fields(m, set_ids[i],
                                                   status="processing"))

    def on_done(i, item, result):
        payload = _result_payload(item, result)
        update_manifest(sid, lambda m: _set_fields(
            m, set_ids[i], status="done", progress=1.0, result=payload))

    def on_error(i, item, exc):
        update_manifest(sid, lambda m: _set_fields(
            m, set_ids[i], status="error", error=str(exc)))

    # Throttle frame-level progress to ~5% steps so long clips don't thrash the
    # manifest on disk (one atomic write per bucket, not per frame).
    last_bucket: dict[str, int] = {}

    def progress_cb(i, frac):
        sid_i = set_ids[i]
        bucket = int(frac * 20)
        if last_bucket.get(sid_i) == bucket:
            return
        last_bucket[sid_i] = bucket
        update_manifest(sid, lambda m: _set_fields(m, sid_i,
                                                  progress=round(frac, 3)))

    run_batch(weights, items, on_start=on_start, on_done=on_done,
              on_error=on_error, progress_cb=progress_cb)

    final = read_manifest(sid) or manifest
    any_err = any(s["status"] == "error" for s in final["sets"])
    update_manifest(sid, lambda m: m.update(state="error" if any_err else "done"))
    if not any_err:
        # keep the rendered outputs; drop the (large) raw/trim uploads
        shutil.rmtree(_session_dir(sid) / "_raw", ignore_errors=True)


def recover_interrupted() -> list[str]:
    """Reconcile manifests with the (now-empty) in-memory queue after a restart.

    The queue doesn't survive a process restart but the manifests do. Any session
    left ``queued``/``processing`` is flipped to ``interrupted`` and its in-flight
    sets reset to ``uploaded`` so the user can re-process — which only re-runs the
    not-yet-``done`` sets. Call once at server startup.
    """
    recovered: list[str] = []
    if not TRACKED_DIR.exists():
        return recovered
    for mp in TRACKED_DIR.glob("*/manifest.json"):
        sid = mp.parent.name
        m = read_manifest(sid)
        if m is None or m.get("state") not in ("queued", "processing"):
            continue

        def fix(mm):
            for s in mm["sets"]:
                if s["status"] == "processing":
                    s["status"], s["progress"] = "uploaded", 0.0
            mm["state"] = "interrupted"

        update_manifest(sid, fix)
        recovered.append(sid)
    return recovered


# --------------------------------------------------------------------------- #
# Routes.
# --------------------------------------------------------------------------- #
@batch_bp.route("/batch", methods=["GET"])
def batch_page():
    return render_template("batch.html", models=_list_models(),
                           default_model=_default_model(),
                           plate_config=load_plate_config())


@batch_bp.route("/sessions", methods=["GET"])
def list_sessions():
    out = []
    if TRACKED_DIR.exists():
        for mp in TRACKED_DIR.glob("*/manifest.json"):
            try:
                m = json.loads(mp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            sets = m.get("sets", [])
            out.append({
                "session_id": m.get("session_id"),
                "title": m.get("title"),
                "state": m.get("state"),
                "created_at": m.get("created_at"),
                "n_sets": len(sets),
                "n_done": sum(1 for s in sets if s.get("status") == "done"),
            })
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return jsonify(out)


@batch_bp.route("/sessions", methods=["POST"])
def create_session():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object."), 400

    sid = allocate_session_id(str(data.get("title") or "session"))
    try:
        manifest = _build_manifest(sid, data)
    except _BadRequest as exc:
        # release the claimed (empty) directory so the id can be reused
        try:
            _session_dir(sid).rmdir()
        except OSError:
            pass
        return jsonify(error=str(exc)), 400

    write_manifest_atomic(sid, manifest)
    return jsonify(
        session_id=sid,
        sets=[{
            "id": s["id"],
            "original_filename": s["original_filename"],
            "upload_url": url_for("batch.upload_video", sid=sid, set_id=s["id"]),
        } for s in manifest["sets"]],
    ), 201


@batch_bp.route("/sessions/<sid>", methods=["GET"])
def get_session(sid):
    manifest = read_manifest(sid)
    if manifest is None:
        abort(404)
    return jsonify(manifest)


@batch_bp.route("/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    manifest = read_manifest(sid)
    if manifest is None:
        abort(404)
    if manifest["state"] in ("queued", "processing"):
        return jsonify(error="Cannot delete a session while it is processing."), 409
    shutil.rmtree(_session_dir(sid), ignore_errors=True)
    with _locks_guard:
        _locks.pop(sid, None)
    return jsonify(ok=True)


@batch_bp.route("/sessions/<sid>/sets/<set_id>/video", methods=["POST"])
def upload_video(sid, set_id):
    manifest = read_manifest(sid)
    if manifest is None or _find_set(manifest, set_id) is None:
        abort(404)
    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify(error="No video uploaded."), 400

    raw_dir = _session_dir(sid) / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_name = f"{set_id}_raw.mp4"
    request.files["video"].save(raw_dir / raw_name)      # streams to disk

    update_manifest(sid, lambda m: _set_fields(
        m, set_id, status="uploaded", raw_filename=raw_name))
    return jsonify(ok=True, set_id=set_id, status="uploaded")


@batch_bp.route("/sessions/<sid>/process", methods=["POST"])
def start_processing(sid):
    manifest = read_manifest(sid)
    if manifest is None:
        abort(404)
    if manifest["state"] in ("queued", "processing"):
        return jsonify(error="Session already processing."), 409

    pending = [s for s in manifest["sets"] if s["status"] != "done"]
    if not pending:
        return jsonify(error="Nothing to process."), 400
    missing = [s["id"] for s in pending if s["status"] == "pending"]
    if missing:
        return jsonify(error="Some sets have no uploaded video.",
                       sets=missing), 400

    update_manifest(sid, lambda m: m.update(state="queued"))
    enqueue(sid)
    return jsonify(ok=True, state="queued"), 202
