import json
import math

import pytest
import torch

from carexp.config import CONFIG_DIR, load_stage_config
from carexp.controller.model import Controller
from carexp.eval.periodic import cell_key, run_periodic_table
from carexp.eval.stats import mean_se, paired_difference, summarize
from carexp.eval.store import CellStore
from carexp.schedulers import PeriodicScheduler


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """An untrained controller saved in the training-checkpoint format."""
    cfg = load_stage_config(CONFIG_DIR / "controller.yaml", smoke=True)
    torch.manual_seed(0)
    model = Controller.from_config(cfg["receiver"]["k_buf"], cfg["model"])
    path = tmp_path_factory.mktemp("ckpt") / "ckpt_latest.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, path)
    return path


def eval_cfg(**grid):
    cfg = load_stage_config(CONFIG_DIR / "eval_periodic.yaml", smoke=True)
    cfg["tracks"]["n"] = 2
    cfg["grid"] = {"p_s": [0.5, 1.0], "m": [1, 2], **grid}
    cfg["env_overrides"] = {"max_steps": 10}
    cfg["runtime"].update(num_workers=2, num_slots=4, device="cpu")
    return cfg


def test_periodic_scheduler():
    s = PeriodicScheduler(3)
    assert [s.decide(0, n, None, None) for n in range(1, 8)] == [False, False, True, False, False, True, False]
    with pytest.raises(ValueError):
        PeriodicScheduler(0)


def test_table_runs_saves_and_resumes(tmp_path, checkpoint):
    res = run_periodic_table(eval_cfg(), checkpoint, tmp_path)
    assert res == {"reason": "done", "cells_run": 4}
    for p in (0.5, 1.0):
        for m in (1, 2):
            s = json.loads((tmp_path / "cells" / f"{cell_key(p, m)}.json").read_text())
            assert s["n"] == 2 and s["p_s"] == p and s["m"] == m
    assert (tmp_path / "table_iii.csv").exists()
    assert "| 0.5 |" in (tmp_path / "table_iii.md").read_text()

    # restart: nothing to do
    assert run_periodic_table(eval_cfg(), checkpoint, tmp_path)["cells_run"] == 0
    # a lost cell is recomputed, and only that one
    (tmp_path / "cells" / f"{cell_key(1.0, 2)}.csv").unlink()
    assert run_periodic_table(eval_cfg(), checkpoint, tmp_path)["cells_run"] == 1
    # growing the grid only runs the new cells
    assert run_periodic_table(eval_cfg(m=[1, 2, 3]), checkpoint, tmp_path)["cells_run"] == 2


def test_tracks_are_paired_across_cells(tmp_path, checkpoint):
    run_periodic_table(eval_cfg(), checkpoint, tmp_path)
    store = CellStore(tmp_path, json.loads((tmp_path / "meta.json").read_text()))
    seeds = {k: [r["seed"] for r in store.load_rows(k)] for k in (cell_key(0.5, 1), cell_key(1.0, 2))}
    assert seeds[cell_key(0.5, 1)] == seeds[cell_key(1.0, 2)] == [0, 1]


def test_changed_setup_is_refused(tmp_path, checkpoint):
    run_periodic_table(eval_cfg(), checkpoint, tmp_path)
    cfg = eval_cfg()
    cfg["channel_seed"] = 1
    with pytest.raises(ValueError, match="different setup"):
        run_periodic_table(cfg, checkpoint, tmp_path)


def test_eval_tracks_must_not_overlap_training(tmp_path, checkpoint):
    cfg = eval_cfg()
    cfg["tracks"]["start"] = 99_999
    with pytest.raises(ValueError):
        run_periodic_table(cfg, checkpoint, tmp_path)


def test_stats():
    rows = [dict(seed=i, ret=r, track_fraction=f, lap_finished=l, length=10, transmissions=5, deliveries=4, end=e)
            for i, (r, f, l, e) in enumerate([(1.0, 0.5, False, "offtrack"), (3.0, 1.0, True, "lap")])]
    s = summarize(rows)
    assert s["return_mean"] == 2.0 and s["return_se"] == pytest.approx(1.0)
    assert s["track_mean"] == 0.75 and s["lap_rate"] == 0.5
    assert s["tx_per_step"] == 0.5 and s["end_lap"] == 1 and s["end_offtrack"] == 1
    other = [{**r, "ret": r["ret"] - 1} for r in rows]
    assert paired_difference(rows, other) == (1.0, 0.0)
    m, se = mean_se([5.0])
    assert m == 5.0 and math.isnan(se)


def test_script_smoke(tmp_path, checkpoint):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_periodic.py"
    out = subprocess.run([sys.executable, str(script), "--smoke", "--checkpoint", str(checkpoint),
                          "--out-dir", str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("[cell p_s=") + out.stderr.count("[cell p_s=") == 4
    assert len(list((tmp_path / "cells").glob("*.csv"))) == 4
