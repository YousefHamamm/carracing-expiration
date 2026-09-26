"""Recurrent actor-critic controller over the receiver's K_BUF delivered frames.

IMPALA residual CNN (3 stages) -> flatten -> linear -> LayerNorm -> concat [Delta_n, previous action]
-> GRU -> tanh-squashed Gaussian policy head and 2-layer value head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_DIM = 2
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        y = self.conv0(F.relu(x))
        y = self.conv1(F.relu(y))
        return x + y


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_ch)
        self.res1 = ResidualBlock(out_ch)

    def forward(self, x):
        return self.res1(self.res0(self.pool(self.conv(x))))


class ImpalaEncoder(nn.Module):
    def __init__(self, in_ch: int, channels=(16, 32, 32), embed_dim: int = 256, frame_hw=(96, 96)):
        super().__init__()
        stages, ch = [], in_ch
        h, w = frame_hw
        for out_ch in channels:
            stages.append(ConvStage(ch, out_ch))
            ch = out_ch
            h, w = (h + 1) // 2, (w + 1) // 2
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(ch * h * w, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frames_u8):
        x = frames_u8.float() / 255.0
        x = F.relu(self.stages(x)).flatten(1)
        return self.norm(self.fc(x))


def squashed_log_prob(u, mean, log_std):
    """log pi(tanh(u)) for pre-tanh sample u under N(mean, exp(log_std)), summed over action dims."""
    var = torch.exp(2 * log_std)
    gauss = -((u - mean) ** 2) / (2 * var) - log_std - 0.5 * math.log(2 * math.pi)
    # log(1 - tanh(u)^2), numerically stable
    correction = 2 * (math.log(2.0) - u - F.softplus(-2 * u))
    return (gauss - correction).sum(-1)


def gaussian_entropy(log_std):
    """Entropy of the pre-tanh Gaussian (the squashed one has no closed form)."""
    return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)


class Controller(nn.Module):
    def __init__(self, k_buf: int, cnn_channels=(16, 32, 32), embed_dim: int = 256,
                 gru_hidden: int = 256, value_hidden: int = 256, delta_scale: float = 20.0,
                 init_log_std: float = -0.5):
        super().__init__()
        self.k_buf = k_buf
        self.delta_scale = delta_scale
        self.hidden_size = gru_hidden
        self.encoder = ImpalaEncoder(k_buf, cnn_channels, embed_dim)
        self.gru = nn.GRUCell(embed_dim + 1 + ACTION_DIM, gru_hidden)
        self.pi_mean = nn.Linear(gru_hidden, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(init_log_std)))
        self.value = nn.Sequential(nn.Linear(gru_hidden, value_hidden), nn.ReLU(),
                                   nn.Linear(value_hidden, 1))
        nn.init.orthogonal_(self.pi_mean.weight, gain=0.01)
        nn.init.zeros_(self.pi_mean.bias)

    @classmethod
    def from_config(cls, k_buf: int, cfg: dict) -> "Controller":
        return cls(k_buf, tuple(cfg["cnn_channels"]), cfg["embed_dim"], cfg["gru_hidden"],
                   cfg["value_hidden"], cfg["delta_scale"], cfg["init_log_std"])

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def _features(self, frames, delta, prev_action):
        z = self.encoder(frames)
        d = (delta.float() / self.delta_scale).unsqueeze(-1)
        return torch.cat([z, d, prev_action.float()], dim=-1)

    def _heads(self, h):
        mean = self.pi_mean(h)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std, self.value(h).squeeze(-1)

    def step(self, frames, delta, prev_action, h, episode_start):
        """One step for a batch. episode_start (B,) zeroes the hidden state before use.

        Returns (mean, log_std, value, new_h).
        """
        h = h * (1.0 - episode_start.float()).unsqueeze(-1)
        h = self.gru(self._features(frames, delta, prev_action), h)
        return (*self._heads(h), h)

    def unroll(self, frames, delta, prev_action, h0, episode_start):
        """Sequence version; inputs are (T, B, ...). The CNN runs over all T*B frames at once.

        Returns mean, log_std, value of shape (T, B, ...).
        """
        T, B = delta.shape
        x = self._features(frames.flatten(0, 1), delta.flatten(0, 1),
                           prev_action.flatten(0, 1)).view(T, B, -1)
        keep = (1.0 - episode_start.float()).unsqueeze(-1)
        h, hs = h0, []
        for t in range(T):
            h = self.gru(x[t], h * keep[t])
            hs.append(h)
        return self._heads(torch.stack(hs))

    @torch.no_grad()
    def act(self, frames, delta, prev_action, h, episode_start, deterministic: bool = False,
            generator: torch.Generator | None = None):
        """Returns (action in [-1,1]^2, pre-tanh u, log_prob, value, new_h)."""
        mean, log_std, value, h = self.step(frames, delta, prev_action, h, episode_start)
        if deterministic:
            u = mean
        else:
            noise = torch.randn(mean.shape, generator=generator, device=mean.device)
            u = mean + noise * log_std.exp()
        return torch.tanh(u), u, squashed_log_prob(u, mean, log_std), value, h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_controller(path, device="cpu") -> tuple[Controller, dict]:
    """Loads a training checkpoint into a frozen (eval-mode, no-grad) controller.
    Returns (model, the training config stored in the checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Controller.from_config(cfg["receiver"]["k_buf"], cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg
