"""Flow Matching / PCFM / CAFM(ステージ10)。

3手法は同じ学習ループの設定違いで実装する:

- FM: 標準的なFlow Matching。射影なし。
- PCFM: 学習はFMと同一。サンプリング時のみ射影する(``sample`` の
  ``projection_mode`` で切り替える)。
- CAFM: 学習損失に微分可能な射影(``differentiable_projection.py``、
  企業ごとに固定されたcvxpylayers)を組み込む。

class 2 は68社しかなく1エポックあたりのペア数が少ないため、エポックごとに
OT輸送計画の行(確率分布)から対応先 j を再サンプリングし、実質的な
データ拡張とする(§2.2)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from diffcp.cone_program import SolverError as DiffcpSolverError
from torch import nn

from fsgen.errors import SolverNotConvergedError
from fsgen.models.differentiable_projection import ProjectionLayer

_DEFAULT_HIDDEN = 128
_DEFAULT_LAYERS = 3


class VelocityField(nn.Module):
    """v_theta: (z, t) -> 速度(z と同じ次元)(§2.3)。"""

    def __init__(self, dim: int, hidden: int = _DEFAULT_HIDDEN, n_layers: int = _DEFAULT_LAYERS):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == z.dim() - 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([z, t], dim=-1)
        result: torch.Tensor = self.net(inp)
        return result


def sample_ot_targets(plan_row: np.ndarray, rng: np.random.Generator) -> int:
    """輸送計画の1行(確率分布とみなす)から対応先indexを1つサンプリングする(§2.2)。

    行の質量がほぼ0(Sinkhornのアンダーフローで潰れた「孤立企業」、
    ステージ6.5以降で既知の現象)の場合、0/0でNaNになるのを避け、
    一様分布にフォールバックする(対応先の手がかりが無いという状態を
    素直に表現する)。
    """
    total = plan_row.sum()
    if total < 1e-12:
        p = np.full(len(plan_row), 1.0 / len(plan_row))
    else:
        p = plan_row / total
    return int(rng.choice(len(p), p=p))


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    cafm_losses: list[float] = field(default_factory=list)  # CAFM射影損失の推移(該当する場合)
    solver_failures: int = 0  # 微分可能射影(diffcp)が失敗し、その企業を当該epochでスキップした回数


def _resample_pairs(
    plan: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """輸送計画の各行から対応先を1つずつ再サンプリングする((n_train,) の目標index)。"""
    return np.array([sample_ot_targets(plan[k], rng) for k in range(plan.shape[0])])


def train_fm(
    z_train: np.ndarray,
    z_pool: np.ndarray,
    plan: np.ndarray,
    net: VelocityField,
    epochs: int,
    lr: float,
    seed: int,
    z_val: np.ndarray | None = None,
    plan_val: np.ndarray | None = None,
) -> TrainResult:
    """標準的なFlow Matching学習(§3.1)。z_train: (n_train, dim) 始点(class2)。
    z_pool: (n_pool, dim) 終点候補(class0)。plan: (n_train, n_pool) 輸送計画。

    z_val/plan_val を渡すと、各エポックで検証損失も記録する(§7、過学習の監視)。
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    result = TrainResult()

    z_pool_t = torch.tensor(z_pool, dtype=torch.float64)
    z_train_t = torch.tensor(z_train, dtype=torch.float64)
    z_val_t = torch.tensor(z_val, dtype=torch.float64) if z_val is not None else None

    for _epoch in range(epochs):
        net.train()
        targets = _resample_pairs(plan, rng)
        z1 = z_pool_t[targets]
        t = torch.tensor(rng.uniform(size=len(z_train)), dtype=torch.float64)
        z_t = (1 - t)[:, None] * z_train_t + t[:, None] * z1
        v_target = z1 - z_train_t
        v_pred = net(z_t, t)
        loss = torch.mean(torch.sum((v_pred - v_target) ** 2, dim=-1))
        if not torch.isfinite(loss):
            raise SolverNotConvergedError(f"FM学習の損失が非有限(epoch={_epoch})")

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        result.train_losses.append(float(loss.item()))

        if z_val_t is not None and plan_val is not None:
            net.eval()
            with torch.no_grad():
                targets_val = _resample_pairs(plan_val, rng)
                z1v = z_pool_t[targets_val]
                tv = torch.tensor(rng.uniform(size=len(z_val_t)), dtype=torch.float64)
                z_tv = (1 - tv)[:, None] * z_val_t + tv[:, None] * z1v
                v_target_v = z1v - z_val_t
                v_pred_v = net(z_tv, tv)
                loss_v = torch.mean(torch.sum((v_pred_v - v_target_v) ** 2, dim=-1))
            result.val_losses.append(float(loss_v.item()))

    return result


def train_cafm(
    z_train: np.ndarray,
    z_pool: np.ndarray,
    plan: np.ndarray,
    net: VelocityField,
    projection_layers: list[ProjectionLayer],
    x_start_train: np.ndarray,
    b: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    fm_loss_weight: float = 1.0,
    z_val: np.ndarray | None = None,
    plan_val: np.ndarray | None = None,
    max_failure_rate: float = 0.01,
) -> TrainResult:
    """CAFM学習(§3.3)。

    **収束チェック(ステージ13 §A.1)**: 微分可能射影のソルバ失敗は、その企業の
    そのepochの寄与をスキップして続行するが、失敗の割合が
    ``max_failure_rate``(既定1%)を超えたら``SolverNotConvergedError``を投げる
    (以前は件数を数えるだけで、多発しても最後まで走っていた)。前進解の
    制約違反(``ProjectionLayer.project``が投げる)は即座に伝播させる。
損失 = fm_loss_weight * FM損失 + ||P_C(z0+v)-z1||^2。

    原著Eq(8b)は射影損失のみだが、それだけではtから外れた領域でのv_thetaの
    挙動が学習されない(射影損失は「z0からのワンステップ予測」しか見ない)
    ため、FM損失も足し合わせる(fm_loss_weight=0でEq(8b)のみの版も作れる)。

    射影は低次元空間ではなく元空間(x, 正規化済みx̃)で定義されているため、
    z -> x -> P_C(x) -> z の往復が必要(§3.3実装上の要点1):
        x_hat = x_start + B @ (z_hat - z_start)
        x_proj = P(x_hat)
        z_proj = B^T @ (x_proj - x_start) + z_start

    projection_layers[k] は x_start_train[k] に対応する企業の微分可能射影層。
    b: (D, k) 低次元基底。x_start_train: (n_train, D) 正規化済み始点。
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    result = TrainResult()

    z_pool_t = torch.tensor(z_pool, dtype=torch.float64)
    z_train_t = torch.tensor(z_train, dtype=torch.float64)
    z_val_t = torch.tensor(z_val, dtype=torch.float64) if z_val is not None else None
    x_start_t = torch.tensor(x_start_train, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    n_train = len(z_train)

    for _epoch in range(epochs):
        net.train()
        targets = _resample_pairs(plan, rng)
        z1 = z_pool_t[targets]
        t = torch.tensor(rng.uniform(size=n_train), dtype=torch.float64)
        z_t = (1 - t)[:, None] * z_train_t + t[:, None] * z1
        v_target = z1 - z_train_t
        v_pred = net(z_t, t)
        loss_fm = torch.mean(torch.sum((v_pred - v_target) ** 2, dim=-1))

        z_hat = z_train_t + v_pred  # ワンステップ終点予測(§3.3 Eq 8b)
        cafm_terms = []
        for k in range(n_train):
            x_hat_k = x_start_t[k] + b_t @ (z_hat[k] - z_train_t[k])
            try:
                x_proj_k = projection_layers[k].project(x_hat_k)
            except (DiffcpSolverError, SolverNotConvergedError):
                # 学習初期はv_thetaが未訓練で予測が実行可能領域から大きく
                # 外れることがあり、diffcp(SCS)がinfeasible/収束失敗を
                # 返すことがある(ステージ10 §7で想定されていた現象)。前進解の制約違反
                # (ProjectionLayer._verify_forward)も同様に数える。
                # その企業のこのepochの寄与はスキップし、学習全体は続行する。
                result.solver_failures += 1
                continue
            z_proj_k = b_t.T @ (x_proj_k - x_start_t[k]) + z_train_t[k]
            cafm_terms.append(torch.sum((z_proj_k - z1[k]) ** 2))
        loss_cafm = (
            torch.mean(torch.stack(cafm_terms))
            if cafm_terms
            else torch.zeros((), dtype=torch.float64)
        )

        loss = fm_loss_weight * loss_fm + loss_cafm
        if not torch.isfinite(loss):
            raise SolverNotConvergedError(f"CAFM学習の損失が非有限(epoch={_epoch})")
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        result.train_losses.append(float(loss.item()))
        result.cafm_losses.append(float(loss_cafm.item()))

        if z_val_t is not None and plan_val is not None:
            net.eval()
            with torch.no_grad():
                targets_val = _resample_pairs(plan_val, rng)
                z1v = z_pool_t[targets_val]
                tv = torch.tensor(rng.uniform(size=len(z_val_t)), dtype=torch.float64)
                z_tv = (1 - tv)[:, None] * z_val_t + tv[:, None] * z1v
                v_target_v = z1v - z_val_t
                v_pred_v = net(z_tv, tv)
                loss_v = torch.mean(torch.sum((v_pred_v - v_target_v) ** 2, dim=-1))
            result.val_losses.append(float(loss_v.item()))

    n_calls = epochs * n_train
    if result.solver_failures > max_failure_rate * n_calls:
        raise SolverNotConvergedError(
            f"微分可能射影のソルバ失敗が多すぎる: {result.solver_failures}/{n_calls}"
            f"({result.solver_failures / n_calls:.1%}、許容{max_failure_rate:.0%})"
        )
    return result


def euler_sample(
    net: VelocityField,
    z_start: np.ndarray,
    ts: tuple[float, ...],
) -> np.ndarray:
    """Euler法でv_thetaを積分し、経路 (T, dim) を返す(§3.1)。tsは昇順、ts[0]=0を仮定。"""
    net.eval()
    z = torch.tensor(z_start, dtype=torch.float64)
    path = [z.detach().numpy().copy()]
    with torch.no_grad():
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            t_tensor = torch.full((), ts[i - 1], dtype=torch.float64)
            v = net(z.unsqueeze(0), t_tensor.unsqueeze(0)).squeeze(0)
            z = z + dt * v
            path.append(z.detach().numpy().copy())
    return np.stack(path)


def euler_sample_batch(
    net: VelocityField,
    z_start: np.ndarray,
    ts: tuple[float, ...],
) -> np.ndarray:
    """euler_sample のバッチ版。z_start: (n, dim) -> (n, T, dim)。"""
    net.eval()
    z = torch.tensor(z_start, dtype=torch.float64)
    n = z.shape[0]
    path = [z.detach().numpy().copy()]
    with torch.no_grad():
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i - 1]
            t_tensor = torch.full((n,), ts[i - 1], dtype=torch.float64)
            v = net(z, t_tensor)
            z = z + dt * v
            path.append(z.detach().numpy().copy())
    return np.stack(path, axis=1)


__all__ = [
    "TrainResult",
    "VelocityField",
    "euler_sample",
    "euler_sample_batch",
    "sample_ot_targets",
    "train_cafm",
    "train_fm",
]
