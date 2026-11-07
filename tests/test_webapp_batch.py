"""Tests for the bulk-import web layer (webapp/batch.py) + a guard that the
single-clip /track contract is unchanged after the Phase 0 refactor.

No real CV runs: ``batch.run_batch`` and ``app.run_tracking`` are monkeypatched.
``batch.MODELS_DIR`` / ``batch.TRACKED_DIR`` are redirected into a tmp dir so the
suite never writes to the user's real ``data/tracked``.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import webapp.app as appmod
import webapp.batch as batch


# --------------------------------------------------------------------------- #
# Fixtures + fakes
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "plate_best.pt").write_bytes(b"fake-weights")
    tracked = tmp_path / "tracked"
    tracked.mkdir()
    monkeypatch.setattr(batch, "MODELS_DIR", models)
    monkeypatch.setattr(batch, "TRACKED_DIR", tracked)
    appmod.app.config.update(TESTING=True)
    return appmod.app.test_client()


def _fake_result():
    return SimpleNamespace(analysis=None, fps=30.0, n_frames=10,
                           coverage=100.0, track_ids=[1])


def fake_run_batch(weights, items, *, on_start=None, on_done=None,
                   on_error=None, progress_cb=None):
    """Every clip succeeds; writes a tiny metrics file like the real pipeline."""
    results = []
    for i, item in enumerate(items):
        if on_start:
            on_start(i, item)
        Path(item.out_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(item.out_metrics).write_text(json.dumps({"calibrated": False}))
        result = _fake_result()
        results.append(result)
        if on_done:
            on_done(i, item, result)
    return results


def fake_run_batch_one_error(weights, items, *, on_start=None, on_done=None,
                             on_error=None, progress_cb=None):
    for i, item in enumerate(items):
        if on_start:
            on_start(i, item)
        if i == 0:
            if on_error:
                on_error(i, item, RuntimeError("boom: bad clip"))
        else:
            Path(item.out_metrics).parent.mkdir(parents=True, exist_ok=True)
            Path(item.out_metrics).write_text("{}")
            if on_done:
                on_done(i, item, _fake_result())


def _create(client, **over):
    payload = {
        "title": over.get("title", "Leg day"),
        "defaults": over.get("defaults", {"plate": "20"}),
        "exercises": over.get("exercises", []),
        "sets": over.get("sets", [{"original_filename": "a.mov"}]),
    }
    return client.post("/sessions", json=payload)


def _upload(client, upload_url, name="clip.mp4"):
    return client.post(upload_url,
                       data={"video": (io.BytesIO(b"rawbytes"), name)},
                       content_type="multipart/form-data")


def _wait_state(client, sid, target, timeout=5.0):
    deadline = time.time() + timeout
    m = None
    while time.time() < deadline:
        m = client.get(f"/sessions/{sid}").get_json()
        if m["state"] == target:
            return m
        time.sleep(0.02)
    raise AssertionError(f"state never reached {target!r}: {m and m['state']}")


# --------------------------------------------------------------------------- #
# Create / validation
# --------------------------------------------------------------------------- #
def test_create_session_writes_valid_manifest(client):
    r = _create(client, exercises=[{"id": "ex1", "name": "Squat"}],
                sets=[{"original_filename": "a.mov", "exercise_id": "ex1",
                       "order": 0}])
    assert r.status_code == 201
    d = r.get_json()
    sid = d["session_id"]
    assert len(d["sets"]) == 1 and d["sets"][0]["upload_url"]

    m = client.get(f"/sessions/{sid}").get_json()
    assert m["schema_version"] == batch.SCHEMA_VERSION
    assert m["state"] == "uploading"
    assert m["defaults"]["plate"] == "20"
    assert m["defaults"]["model"] == "plate_best.pt"   # defaulted
    assert m["sets"][0]["status"] == "pending"


def test_session_id_collision_suffixes(client):
    s1 = _create(client, title="Leg day").get_json()["session_id"]
    s2 = _create(client, title="Leg day").get_json()["session_id"]
    assert s1 != s2
    assert s2.endswith("-2")


def test_bad_plate_override_is_400(client):
    r = _create(client, sets=[{"original_filename": "a.mov",
                               "overrides": {"plate": "999"}}])
    assert r.status_code == 400
    assert "999" in r.get_json()["error"]


def test_unknown_model_is_400(client):
    r = _create(client, defaults={"model": "nope.pt"})
    assert r.status_code == 400


def test_unknown_exercise_ref_is_400(client):
    r = _create(client, sets=[{"original_filename": "a.mov",
                               "exercise_id": "ghost"}])
    assert r.status_code == 400


def test_no_sets_is_400(client):
    r = _create(client, sets=[])
    assert r.status_code == 400


def test_arrangement_fields_round_trip(client):
    r = _create(
        client,
        defaults={"plate": "20", "rep_style": "auto"},
        exercises=[{"id": "ex1", "name": "Squat"}, {"id": "ex2", "name": "Bench"}],
        sets=[{"original_filename": "s1.mov", "exercise_id": "ex2", "order": 5,
               "overrides": {"plate": "25", "rep_style": "down-first",
                             "trim": {"start": 1, "end": 3}}}],
    )
    sid = r.get_json()["session_id"]
    m = client.get(f"/sessions/{sid}").get_json()
    assert [e["name"] for e in m["exercises"]] == ["Squat", "Bench"]
    s = m["sets"][0]
    assert s["exercise_id"] == "ex2"
    assert s["order"] == 5
    assert s["overrides"] == {"plate": "25", "rep_style": "down-first",
                              "trim": {"start": 1.0, "end": 3.0}}


def test_trim_dropped_when_end_not_after_start(client):
    r = _create(client, sets=[{"original_filename": "a.mov",
                               "overrides": {"trim": {"start": 5, "end": 2}}}])
    sid = r.get_json()["session_id"]
    m = client.get(f"/sessions/{sid}").get_json()
    assert "trim" not in m["sets"][0]["overrides"]


# --------------------------------------------------------------------------- #
# Upload + process
# --------------------------------------------------------------------------- #
def test_upload_sets_status_and_writes_file(client):
    d = _create(client).get_json()
    sid, st = d["session_id"], d["sets"][0]
    assert _upload(client, st["upload_url"]).status_code == 200

    m = client.get(f"/sessions/{sid}").get_json()
    s = m["sets"][0]
    assert s["status"] == "uploaded"
    assert s["raw_filename"] == f"{st['id']}_raw.mp4"
    assert (batch.TRACKED_DIR / sid / "_raw" / s["raw_filename"]).exists()


def test_process_requires_all_uploaded(client, monkeypatch):
    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    sid = _create(client).get_json()["session_id"]
    r = client.post(f"/sessions/{sid}/process")
    assert r.status_code == 400          # nothing uploaded yet


def test_full_flow_create_upload_process_poll(client, monkeypatch):
    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    d = _create(client, exercises=[{"id": "ex1", "name": "Squat"}],
                sets=[{"original_filename": "a.mov", "exercise_id": "ex1",
                       "order": 0},
                      {"original_filename": "b.mov", "exercise_id": "ex1",
                       "order": 1}]).get_json()
    sid = d["session_id"]
    for st in d["sets"]:
        assert _upload(client, st["upload_url"]).status_code == 200

    assert client.post(f"/sessions/{sid}/process").status_code == 202
    m = _wait_state(client, sid, "done")

    assert all(s["status"] == "done" for s in m["sets"])
    res = m["sets"][0]["result"]
    assert res["metrics_name"].endswith("_metrics.json")
    assert res["boxed_name"].endswith("_tracked.mp4")
    assert res["fps"] == 30.0


def test_double_process_is_rejected(client, monkeypatch):
    # block run_batch so the session stays 'processing' while we double-submit
    started = {"e": __import__("threading").Event()}

    def slow_run_batch(weights, items, **kw):
        started["e"].set()
        time.sleep(0.3)
        return []

    monkeypatch.setattr(batch, "run_batch", slow_run_batch)
    d = _create(client).get_json()
    sid = d["session_id"]
    _upload(client, d["sets"][0]["upload_url"])
    assert client.post(f"/sessions/{sid}/process").status_code == 202
    assert started["e"].wait(2.0)
    assert client.post(f"/sessions/{sid}/process").status_code == 409
    _wait_state(client, sid, "done")


def test_one_set_error_marks_session_error(client, monkeypatch):
    monkeypatch.setattr(batch, "run_batch", fake_run_batch_one_error)
    d = _create(client, sets=[{"original_filename": "a.mov", "order": 0},
                              {"original_filename": "b.mov", "order": 1}]).get_json()
    sid = d["session_id"]
    for st in d["sets"]:
        _upload(client, st["upload_url"])
    client.post(f"/sessions/{sid}/process")
    m = _wait_state(client, sid, "error")

    statuses = {s["original_filename"]: s["status"] for s in m["sets"]}
    assert statuses["a.mov"] == "error"
    assert statuses["b.mov"] == "done"
    assert any(s["status"] == "error" and s["error"] for s in m["sets"])


def test_list_sessions(client):
    _create(client, title="Alpha")
    out = client.get("/sessions").get_json()
    assert any(x["title"] == "Alpha" and x["n_sets"] == 1 for x in out)


def test_get_missing_session_404(client):
    assert client.get("/sessions/does-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
# Phase 3 — cleanup, delete, restart recovery
# --------------------------------------------------------------------------- #
def test_raw_uploads_purged_after_success(client, monkeypatch):
    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    d = _create(client).get_json()
    sid = d["session_id"]
    _upload(client, d["sets"][0]["upload_url"])
    assert (batch.TRACKED_DIR / sid / "_raw").exists()
    client.post(f"/sessions/{sid}/process")
    _wait_state(client, sid, "done")
    assert not (batch.TRACKED_DIR / sid / "_raw").exists()   # raw dropped, outputs kept


def test_delete_session_removes_folder(client):
    sid = _create(client).get_json()["session_id"]
    assert (batch.TRACKED_DIR / sid).exists()
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert not (batch.TRACKED_DIR / sid).exists()
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_delete_rejected_while_processing(client):
    sid = _create(client).get_json()["session_id"]
    batch.update_manifest(sid, lambda m: m.update(state="processing"))
    assert client.delete(f"/sessions/{sid}").status_code == 409
    assert (batch.TRACKED_DIR / sid).exists()


def test_recover_interrupted_resets_inflight(client):
    d = _create(client, sets=[{"original_filename": "a.mov", "order": 0},
                              {"original_filename": "b.mov", "order": 1}]).get_json()
    sid = d["session_id"]
    for st in d["sets"]:
        _upload(client, st["upload_url"])

    # simulate a crash mid-process: one set processing, session processing
    def crash(m):
        m["state"] = "processing"
        m["sets"][0]["status"] = "processing"
    batch.update_manifest(sid, crash)

    recovered = batch.recover_interrupted()
    assert sid in recovered
    m = client.get(f"/sessions/{sid}").get_json()
    assert m["state"] == "interrupted"
    assert m["sets"][0]["status"] == "uploaded"   # reset so re-process can finish it
    assert m["sets"][1]["status"] == "uploaded"


def test_reprocess_after_recovery_only_runs_not_done(client, monkeypatch):
    seen = {"sources": []}

    def spy_run_batch(weights, items, *, on_start=None, on_done=None,
                      on_error=None, progress_cb=None):
        seen["sources"] = [Path(it.source).name for it in items]
        return fake_run_batch(weights, items, on_start=on_start,
                              on_done=on_done, on_error=on_error)

    monkeypatch.setattr(batch, "run_batch", spy_run_batch)
    d = _create(client, sets=[{"original_filename": "a.mov", "order": 0},
                              {"original_filename": "b.mov", "order": 1}]).get_json()
    sid = d["session_id"]
    for st in d["sets"]:
        _upload(client, st["upload_url"])

    # mark the first set already done (as if it finished before the crash)
    done_id = d["sets"][0]["id"]
    batch.update_manifest(sid, lambda m: _set_fields_test(m, done_id, status="done"))

    client.post(f"/sessions/{sid}/process")
    _wait_state(client, sid, "done")
    # only the not-done set was handed to run_batch
    assert seen["sources"] == [f"{d['sets'][1]['id']}_raw.mp4"]


def _set_fields_test(m, set_id, **fields):
    for s in m["sets"]:
        if s["id"] == set_id:
            s.update(fields)


# --------------------------------------------------------------------------- #
# Constraint #1 — single-clip /track contract unchanged
# --------------------------------------------------------------------------- #
def test_track_response_shape_unchanged(client, monkeypatch, tmp_path):
    models = tmp_path / "m2"
    models.mkdir()
    (models / "plate_best.pt").write_bytes(b"x")
    monkeypatch.setattr(appmod, "MODELS_DIR", models)
    monkeypatch.setattr(appmod, "UPLOAD_DIR", tmp_path)

    fake = SimpleNamespace(
        boxed_video=Path("a_tracked.mp4"), barpath_video=Path("a_barpath.mp4"),
        csv_path=Path("a_track.csv"), metrics_path=Path("a_metrics.json"),
        fps=30.0, n_frames=100, coverage=99.0, track_ids=[1],
        analysis=SimpleNamespace(to_dict=lambda: {"calibrated": True}))
    monkeypatch.setattr(appmod, "run_tracking", lambda **kw: fake)

    r = client.post("/track",
                    data={"video": (io.BytesIO(b"v"), "a.mp4"), "plate": "20"},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    for key in ("boxed_url", "barpath_url", "saved_dir", "boxed_name",
                "barpath_name", "csv_name", "metrics_name", "fps", "n_frames",
                "coverage", "track_ids", "analysis"):
        assert key in d
    assert d["boxed_name"] == "a_tracked.mp4"
    assert d["analysis"] == {"calibrated": True}
