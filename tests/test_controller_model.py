import numpy as np
import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from carexp.controller.model import Controller, count_parameters, squashed_log_prob


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Controller(k_buf=4)


def inputs(T, B, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, 256, (T, B, k, 96, 96), generator=g, dtype=torch.uint8)
    delta = torch.randint(0, 10, (T, B), generator=g)
    prev_a = torch.rand(T, B, 2, generator=g) * 2 - 1
    start = torch.rand(T, B, generator=g) < 0.2
    return frames, delta, prev_a, start


def test_parameter_count(model):
    # CLAUDE.md: "about 1.6M parameters"; GRU/value widths are unspecified (256 each here)
    assert 1.5e6 < count_parameters(model) < 1.8e6


def test_architecture(model):
    assert [stage.conv.out_channels for stage in model.encoder.stages] == [16, 32, 32]
    assert model.encoder.stages[0].conv.in_channels == 4  # K_BUF stacked frames
    assert model.encoder.fc.out_features == 256
    assert isinstance(model.encoder.norm, torch.nn.LayerNorm)
    assert model.gru.input_size == 256 + 1 + 2


def test_output_shapes(model):
    frames, delta, prev_a, start = inputs(1, 3)
    a, u, logp, v, h = model.act(frames[0], delta[0], prev_a[0], model.initial_state(3), start[0])
    assert a.shape == (3, 2) and u.shape == (3, 2) and logp.shape == (3,) and v.shape == (3,)
    assert h.shape == (3, model.hidden_size)
    assert torch.all(a.abs() <= 1)


def test_unroll_matches_stepwise(model):
    T, B = 6, 3
    frames, delta, prev_a, start = inputs(T, B)
    h0 = torch.randn(B, model.hidden_size)
    with torch.no_grad():
        mean_u, _, v_u = model.unroll(frames, delta, prev_a, h0, start)
        h, means, vals = h0, [], []
        for t in range(T):
            m, _, v, h = model.step(frames[t], delta[t], prev_a[t], h, start[t])
            means.append(m)
            vals.append(v)
    torch.testing.assert_close(mean_u, torch.stack(means), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v_u, torch.stack(vals), rtol=1e-4, atol=1e-5)


def test_episode_start_resets_hidden(model):
    frames, delta, prev_a, _ = inputs(1, 2)
    start = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        out_a = model.step(frames[0], delta[0], prev_a[0], torch.randn(2, model.hidden_size), start)
        out_b = model.step(frames[0], delta[0], prev_a[0], model.initial_state(2), start)
    for x, y in zip(out_a, out_b):
        torch.testing.assert_close(x, y)


def test_squashed_log_prob_matches_torch():
    torch.manual_seed(0)
    mean, log_std = torch.randn(50, 2), torch.randn(50, 2) * 0.3 - 0.5
    u = mean + torch.randn(50, 2) * log_std.exp()
    ref = TransformedDistribution(Normal(mean, log_std.exp()), TanhTransform()).log_prob(torch.tanh(u)).sum(-1)
    torch.testing.assert_close(squashed_log_prob(u, mean, log_std), ref, rtol=1e-4, atol=1e-4)
    # stable for large |u|, where tanh saturates
    assert torch.isfinite(squashed_log_prob(torch.full((1, 2), 30.0), torch.zeros(1, 2), torch.zeros(1, 2))).all()


def test_deterministic_action_is_tanh_mean(model):
    frames, delta, prev_a, start = inputs(1, 2)
    h = model.initial_state(2)
    a, u, *_ = model.act(frames[0], delta[0], prev_a[0], h, start[0], deterministic=True)
    mean, *_ = model.step(frames[0], delta[0], prev_a[0], h, start[0])
    torch.testing.assert_close(a, torch.tanh(mean))


def test_frame_scale_does_not_matter_for_dtype(model):
    frames, delta, prev_a, start = inputs(1, 2)
    with torch.no_grad():
        z = model.encoder(frames[0])
    assert np.isfinite(z.numpy()).all()
