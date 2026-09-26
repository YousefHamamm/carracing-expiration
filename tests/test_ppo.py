import numpy as np
import pytest
import torch

from carexp.controller.curriculum import delivery_prob
from carexp.controller.model import Controller
from carexp.controller.ppo import RolloutBuffer, adapt_lr, compute_gae, ppo_update, sampled_kl

PPO = dict(epochs=1, minibatches=1, clip=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
           kl_target=0.02, early_stop_kl_mult=1.5, lr_min=2e-5, lr_max=1.2e-3, lr_factor=1.5)


def test_curriculum():
    total = 1000
    assert delivery_prob(0, total) == 1.0
    assert delivery_prob(200, total) == pytest.approx(0.65)
    assert delivery_prob(400, total) == pytest.approx(0.3)
    assert delivery_prob(999, total) == pytest.approx(0.3)
    ps = [delivery_prob(s, total) for s in range(0, 400, 10)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_gae_matches_hand_computation():
    r = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    v = np.array([[0.5], [0.4], [0.3]], dtype=np.float32)
    done = np.array([[False], [True], [False]])
    last = np.array([1.0], dtype=np.float32)
    g, lam = 0.9, 0.8
    d2 = 3 + g * 1.0 - 0.3
    d1 = 2 - 0.4                      # episode ends after step 1: no bootstrap
    d0 = 1 + g * 0.4 - 0.5
    a2, a1 = d2, d1
    a0 = d0 + g * lam * a1
    adv, ret = compute_gae(r, v, done, last, g, lam)
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + v)


@pytest.mark.parametrize("kl, expected", [
    (0.05, 1e-4 / 1.5),   # > 2 * target: shrink
    (0.005, 1e-4 * 1.5),  # < target / 2: grow
    (0.02, 1e-4),         # in band: keep
])
def test_adapt_lr(kl, expected):
    assert adapt_lr(1e-4, kl, 0.02, 2e-5, 1.2e-3) == pytest.approx(expected)


def test_adapt_lr_clipped_to_range():
    assert adapt_lr(2e-5, 1.0, 0.02, 2e-5, 1.2e-3) == 2e-5
    assert adapt_lr(1.2e-3, 0.0, 0.02, 2e-5, 1.2e-3) == 1.2e-3


def test_sampled_kl():
    assert sampled_kl(torch.zeros(10)).item() == 0.0
    assert sampled_kl(torch.tensor([0.3, -0.3])).item() > 0


def make_buffer(model, T=8, N=4, seed=0):
    """Fills a buffer from the model itself, with episode starts mid-sequence."""
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer(T, N, model.k_buf, (96, 96), model.hidden_size)
    h = torch.randn(N, model.hidden_size, generator=g)
    buf.h0 = h.clone()
    for t in range(T):
        buf.frames[t] = torch.randint(0, 256, (N, model.k_buf, 96, 96), generator=g, dtype=torch.uint8).numpy()
        buf.delta[t] = torch.randint(0, 5, (N,), generator=g).numpy()
        buf.prev_action[t] = (torch.rand(N, 2, generator=g) * 2 - 1).numpy()
        buf.episode_start[t] = (torch.rand(N, generator=g) < 0.25).numpy()
        a, u, logp, v, h = model.act(torch.as_tensor(buf.frames[t]), torch.as_tensor(buf.delta[t]),
                                     torch.as_tensor(buf.prev_action[t]), h,
                                     torch.as_tensor(buf.episode_start[t]), generator=g)
        buf.u[t], buf.log_prob[t], buf.value[t] = u.numpy(), logp.numpy(), v.numpy()
        buf.reward[t] = torch.randn(N, generator=g).numpy()
    buf.done[:-1] = buf.episode_start[1:]
    buf.advantage, buf.returns = compute_gae(buf.reward, buf.value, buf.done, np.zeros(N, np.float32), 0.99, 0.95)
    return buf


def test_update_replays_rollout_exactly():
    """Before the first gradient step the recomputed log-probs must equal the rollout's:
    this checks h0 and the episode-start masks are replayed correctly."""
    torch.manual_seed(0)
    model = Controller(k_buf=4)
    buf = make_buffer(model)
    opt = torch.optim.Adam(model.parameters())
    stats, _ = ppo_update(model, opt, buf, PPO, 1e-4, "cpu")
    assert stats.kl < 1e-9
    assert stats.clip_frac == 0.0


def test_update_changes_params_and_adapts_lr():
    torch.manual_seed(0)
    model = Controller(k_buf=4)
    buf = make_buffer(model)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters())
    stats, new_lr = ppo_update(model, opt, buf, {**PPO, "epochs": 2, "minibatches": 2}, 1e-4, "cpu")
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert stats.epochs_done == 2.0 and not stats.early_stopped
    assert new_lr == adapt_lr(1e-4, stats.kl, 0.02, 2e-5, 1.2e-3)


def test_kl_early_stop():
    """A huge lr makes the policy jump after the first step; the next minibatch's KL check stops the update."""
    torch.manual_seed(0)
    model = Controller(k_buf=4)
    buf = make_buffer(model)
    opt = torch.optim.Adam(model.parameters())
    cfg = {**PPO, "epochs": 5, "minibatches": 1, "kl_target": 1e-6, "lr_max": 1.0}
    stats, new_lr = ppo_update(model, opt, buf, cfg, 0.05, "cpu")
    assert stats.early_stopped
    assert stats.epochs_done == 1.0
    assert stats.kl > 1.5e-6
    assert new_lr == pytest.approx(0.05 / 1.5)
