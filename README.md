# carracing-expiration

Expiration-aware scheduling for remote driving on Gymnasium CarRacing-v3. See `CLAUDE.md` for the project spec.

## Setup

```
pip install -e ".[test]"
```

## Tests

```
pytest --smoke   # tiny version, under 2 minutes
pytest           # full version (5 track seeds x 300 control steps; ~10 minutes)
```

## Layout so far

- `src/carexp/env/carracing.py`: CarRacing wrapper (2-D action map, action repeat, grayscale frames, off-track termination, time limit)
- `src/carexp/env/features.py`: the sender's state vector s_n (31 dims; layout in the module docstring)
- `src/carexp/env/replay.py`: rebuilds a state by seed + action replay
- `src/carexp/channel/channel.py`: Bernoulli(p_s) channel with per-attempt cost
- `src/carexp/channel/receiver.py`: receiver buffer of the last K_BUF delivered frames plus Delta_n
- `configs/base.yaml`: env and receiver settings
