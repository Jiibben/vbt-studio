"""Tests for vbt_core.config.plate_config.

Isolation: ``config_path()`` reads ``VBT_PLATE_CONFIG`` on every call, so each
test sets it to a file under ``tmp_path``. The autouse conftest fixture also
redirects it, but these set it explicitly for clarity.
"""
from __future__ import annotations

import json

import pytest

from vbt_core.config import plate_config as pc


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    p = tmp_path / "pc.json"
    monkeypatch.setenv("VBT_PLATE_CONFIG", str(p))
    return p


def test_config_path_honours_env(tmp_path, monkeypatch):
    target = tmp_path / "somewhere" / "custom.json"
    monkeypatch.setenv("VBT_PLATE_CONFIG", str(target))
    assert pc.config_path() == target


def test_config_path_default_when_unset(monkeypatch):
    monkeypatch.delenv("VBT_PLATE_CONFIG", raising=False)
    assert pc.config_path() == pc.ROOT / "data" / "plate_config.json"


def test_defaults_returned_when_no_file(cfg_path):
    assert not cfg_path.exists()
    assert pc.load_plate_config() == pc.DEFAULTS
    # Returned dict is a copy — mutating it must not corrupt DEFAULTS.
    loaded = pc.load_plate_config()
    loaded["25"] = 1.0
    assert pc.DEFAULTS["25"] == 450.0


def test_defaults_returned_on_bad_json(cfg_path):
    cfg_path.write_text("{ this is not valid json ")
    assert pc.load_plate_config() == pc.DEFAULTS


def test_load_merges_saved_over_defaults(cfg_path):
    cfg_path.write_text(json.dumps({"25": 451.0}))
    cfg = pc.load_plate_config()
    assert cfg["25"] == 451.0
    # Untouched keys keep their defaults.
    assert cfg["20"] == pc.DEFAULTS["20"]


@pytest.mark.parametrize("saved,expected", [
    (10.0, pc.MIN_MM),     # below MIN clamps up
    (49.999, pc.MIN_MM),
    (pc.MIN_MM, pc.MIN_MM),
    (300.0, 300.0),        # in range, untouched
    (pc.MAX_MM, pc.MAX_MM),
    (600.001, pc.MAX_MM),
    (99999.0, pc.MAX_MM),  # above MAX clamps down
])
def test_load_clamps_out_of_range(cfg_path, saved, expected):
    cfg_path.write_text(json.dumps({"25": saved}))
    assert pc.load_plate_config()["25"] == expected


def test_load_ignores_unknown_keys(cfg_path):
    cfg_path.write_text(json.dumps({"999": 300.0, "25": 460.0}))
    cfg = pc.load_plate_config()
    assert "999" not in cfg
    assert cfg["25"] == 460.0
    assert set(cfg) == set(pc.DEFAULTS)


@pytest.mark.parametrize("bad", ["not-a-number", None, [1, 2], {"x": 1}])
def test_load_ignores_non_numeric_values(cfg_path, bad):
    cfg_path.write_text(json.dumps({"25": bad}))
    cfg = pc.load_plate_config()
    # Falls back to the default for that key without crashing.
    assert cfg["25"] == pc.DEFAULTS["25"]


def test_load_accepts_numeric_string(cfg_path):
    # float("300") works, so a numeric string is honoured (and clamped).
    cfg_path.write_text(json.dumps({"25": "300"}))
    assert pc.load_plate_config()["25"] == 300.0


def test_save_persists_and_merges(cfg_path):
    result = pc.save_plate_config({"25": 455.0})
    assert result["25"] == 455.0
    assert result["20"] == pc.DEFAULTS["20"]
    # Atomic persist — reload from disk returns the saved value.
    assert cfg_path.exists()
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["25"] == 455.0
    assert pc.load_plate_config()["25"] == 455.0


def test_save_returns_full_merged_config(cfg_path):
    result = pc.save_plate_config({"10": 320.0})
    assert set(result) == set(pc.DEFAULTS)
    assert result["10"] == 320.0


def test_save_repeated_accumulates(cfg_path):
    pc.save_plate_config({"25": 455.0})
    pc.save_plate_config({"20": 448.0})
    cfg = pc.load_plate_config()
    assert cfg["25"] == 455.0
    assert cfg["20"] == 448.0


def test_save_accepts_int_diameter(cfg_path):
    result = pc.save_plate_config({"5": 230})
    assert result["5"] == 230.0
    assert isinstance(result["5"], float)


def test_save_rejects_non_dict():
    for bad in ([1, 2, 3], "string", 42, None):
        with pytest.raises(ValueError):
            pc.save_plate_config(bad)


def test_save_rejects_unknown_key(cfg_path):
    with pytest.raises(ValueError, match="Unknown plate weight"):
        pc.save_plate_config({"999": 300.0})


@pytest.mark.parametrize("bad", ["abc", None, [1], {"a": 1}])
def test_save_rejects_non_numeric_diameter(cfg_path, bad):
    with pytest.raises(ValueError, match="Invalid diameter"):
        pc.save_plate_config({"25": bad})


@pytest.mark.parametrize("bad", [49.0, 0.0, -5.0, 600.1, 1000.0])
def test_save_rejects_out_of_range(cfg_path, bad):
    with pytest.raises(ValueError, match="must be"):
        pc.save_plate_config({"25": bad})


@pytest.mark.parametrize("good", [pc.MIN_MM, pc.MAX_MM, 300.0])
def test_save_accepts_range_boundaries(cfg_path, good):
    result = pc.save_plate_config({"25": good})
    assert result["25"] == good


def test_save_failure_does_not_corrupt_existing(cfg_path):
    pc.save_plate_config({"25": 455.0})
    with pytest.raises(ValueError):
        pc.save_plate_config({"25": 9999.0})
    # The earlier good value survives; no partial/temp file left as the store.
    assert pc.load_plate_config()["25"] == 455.0


def test_save_leaves_no_tmp_files(cfg_path):
    pc.save_plate_config({"25": 455.0})
    leftovers = list(cfg_path.parent.glob("*.tmp"))
    assert leftovers == []
