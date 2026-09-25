import csv
import subprocess
import sys
from pathlib import Path

import pytest

from carexp.config import load_stage_config
from carexp.controller.train import train
from carexp.utils.checkpoint import load_latest

REPO = Path(__file__).resolve().parents[1]


def tiny_cfg(total_steps):
    cfg = load_stage_config(REPO / "configs" / "controller.yaml", smoke=True)
    cfg["ppo"].update(num_envs=2, rollout_len=8, total_steps=total_steps, epochs=1, minibatches=1)
    cfg["runtime"].update(num_workers=1, checkpoint_every=1, device="cpu")
    cfg["env"]["max_steps"] = 10
    return cfg


def read_updates(out):
    with open(out / "metrics.csv") as f:
        return [int(r["update"]) for r in csv.DictReader(f)]


def test_train_then_resume(tmp_path):
    res = train(tiny_cfg(32), tmp_path)
    assert res == {"reason": "done", "update": 2, "global_step": 32}
    ck = load_latest(tmp_path)
    assert ck["update"] == 2 and ck["global_step"] == 32
    counters_before = ck["episode_counters"].copy()

    # a restart with the same budget does nothing
    assert train(tiny_cfg(32), tmp_path)["update"] == 2

    # extending the budget continues from the checkpoint
    res = train(tiny_cfg(64), tmp_path)
    assert res["update"] == 4 and res["global_step"] == 64
    assert read_updates(tmp_path) == [1, 2, 3, 4]
    assert (load_latest(tmp_path)["episode_counters"] >= counters_before).all()


def test_metrics_rows_after_checkpoint_are_dropped_on_resume(tmp_path):
    train(tiny_cfg(32), tmp_path)
    # simulate a crash after update 3 was logged but before it was checkpointed
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    with open(tmp_path / "metrics.csv", "a", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writerow({**rows[-1], "update": 3})
    train(tiny_cfg(48), tmp_path)
    assert read_updates(tmp_path) == [1, 2, 3]


def test_num_envs_mismatch_refused(tmp_path):
    train(tiny_cfg(16), tmp_path)
    cfg = tiny_cfg(32)
    cfg["ppo"]["num_envs"] = 3
    with pytest.raises(ValueError):
        train(cfg, tmp_path)


def test_script_smoke(tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "train_controller.py"), "--smoke",
                          "--out-dir", str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert (tmp_path / "ckpt_latest.pt").exists()
    assert read_updates(tmp_path) == [1, 2, 3, 4]


def test_stop_signal_checkpoints_and_exits(tmp_path):
    """SIGUSR1 to the whole process group (as SLURM does): finish the update, checkpoint, exit 3, resume."""
    import os
    import signal
    import time

    proc = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts" / "train_controller.py"), "--smoke",
                             "--out-dir", str(tmp_path), "--total-steps", "100000"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.time() + 120
    for line in proc.stdout:
        if line.startswith("[update") or time.time() > deadline:
            break
    os.killpg(proc.pid, signal.SIGUSR1)
    rest = proc.communicate(timeout=120)[0]
    assert proc.returncode == 3, rest[-2000:]
    assert "stop signal received" in rest
    ck = load_latest(tmp_path)
    assert ck is not None and 1 <= ck["update"] < 100000 // 64

    res = train(tiny_cfg_like_smoke(ck["global_step"] + 64), tmp_path)
    assert res["update"] == ck["update"] + 1


def tiny_cfg_like_smoke(total_steps):
    cfg = load_stage_config(REPO / "configs" / "controller.yaml", smoke=True)
    cfg["ppo"]["total_steps"] = total_steps
    return cfg
