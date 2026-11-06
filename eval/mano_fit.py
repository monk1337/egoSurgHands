"""
MANO fitting to Aria MPS keypoints.

Aria MPS hand tracking and MANO have different anatomical models, so raw
comparison yields inflated PA-MPJPE. Fitting MANO to MPS keypoints produces
a MANO-compatible GT at the cost of ~5 mm residual, *and* provides full
mesh vertices so we can compute PA-PVE on Aria too.

Strategy:
    1. Mirror left hands in X so a single MANO_RIGHT model can fit both.
    2. LBFGS stage 1: fit translation + global_orient only (rigid alignment).
    3. LBFGS stage 2: full fit (add hand_pose + betas with regularization).
    4. Un-mirror left hands in output (both joints and vertices).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .config import MANO_TO_OPENPOSE, ARIA_MPS_TO_OPENPOSE, OP_VALID


@dataclass
class FitResult:
    joints_21_op: np.ndarray   # (B, 21, 3) OpenPose order, same frame as input
    vertices: np.ndarray       # (B, 778, 3) same frame as input
    residual_mm: np.ndarray    # (B,) per-sample mean joint error at 20 valid landmarks


class MANOFitter:
    """
    Fits MANO to Aria MPS keypoints using LBFGS.

    Parameters
    ----------
    smplx_mano_right : smplx.MANO
        A pre-loaded smplx MANO_RIGHT model with ``use_pca=False`` and
        ``flat_hand_mean=True``. Must return 16 joints + 778 vertices.
    extra_joints_idxs : list[int]
        5 vertex indices used to derive fingertips (HaMeR convention).
        From ``hamer_model.mano.extra_joints_idxs``.
    device : torch.device
        Target device. All tensors will be on this device.
    """

    def __init__(self, smplx_mano_right, extra_joints_idxs, device):
        self.mano = smplx_mano_right
        self.extra_idxs = extra_joints_idxs
        self.device = device
        self.mano_to_op = torch.tensor(MANO_TO_OPENPOSE, device=device, dtype=torch.long)
        self.op_valid_t = torch.tensor(OP_VALID, device=device, dtype=torch.long)

    def mano_forward_21(self, betas, global_orient, hand_pose, transl):
        """Run MANO and return 21 joints (OpenPose order) + 778 vertices.

        Inputs are axis-angle, (B, *).
        """
        out = self.mano(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl,
        )
        extra = out.vertices[:, self.extra_idxs]
        joints_21_mano = torch.cat([out.joints, extra], dim=1)
        joints_21_op = joints_21_mano[:, self.mano_to_op]
        return joints_21_op, out.vertices

    def fit_batch(self, mps_joints_batch: torch.Tensor, hand_sides,
                  n_iters_rigid: int = 5, n_iters_full: int = 10,
                  verbose: bool = False) -> FitResult:
        """Fit MANO to a batch of Aria MPS joints, return joints + vertices + residual."""
        B = mps_joints_batch.shape[0]
        device = self.device

        is_left = torch.tensor([1.0 if s == "left" else 0.0 for s in hand_sides], device=device)
        sign = torch.where(
            is_left.bool(),
            torch.tensor(-1.0, device=device),
            torch.tensor(1.0, device=device),
        )
        mps_mirrored = mps_joints_batch.clone()
        mps_mirrored[:, :, 0] = mps_mirrored[:, :, 0] * sign[:, None]

        target_op = torch.zeros(B, 21, 3, device=device)
        for mps_i, op_i in ARIA_MPS_TO_OPENPOSE.items():
            target_op[:, op_i] = mps_mirrored[:, mps_i]
        target_valid = target_op[:, self.op_valid_t]

        betas = torch.zeros(B, 10, device=device, requires_grad=True)
        global_orient = torch.zeros(B, 3, device=device, requires_grad=True)
        hand_pose = torch.zeros(B, 45, device=device, requires_grad=True)
        transl = mps_mirrored[:, 5].clone().detach().requires_grad_(True)

        opt1 = torch.optim.LBFGS(
            [transl, global_orient],
            lr=0.5, max_iter=50, line_search_fn="strong_wolfe",
        )

        def closure1():
            opt1.zero_grad()
            joints, _ = self.mano_forward_21(betas, global_orient, hand_pose, transl)
            diff = joints[:, self.op_valid_t] - target_valid
            loss = (diff ** 2).sum()
            loss.backward()
            return loss

        for it in range(n_iters_rigid):
            loss = opt1.step(closure1)
            if verbose:
                print(f"  stage1 step {it}: {float(loss) * 1e6 / B:.1f} mm² avg")

        opt2 = torch.optim.LBFGS(
            [transl, global_orient, hand_pose, betas],
            lr=0.5, max_iter=50, line_search_fn="strong_wolfe",
        )

        def closure2_bw():
            opt2.zero_grad()
            joints, _ = self.mano_forward_21(betas, global_orient, hand_pose, transl)
            diff = joints[:, self.op_valid_t] - target_valid
            kp_loss = (diff ** 2).sum()
            beta_reg = 0.001 * (betas ** 2).sum()
            pose_reg = 0.0001 * (hand_pose ** 2).sum()
            loss = kp_loss + beta_reg + pose_reg
            loss.backward()
            return loss

        for it in range(n_iters_full):
            loss = opt2.step(closure2_bw)
            if verbose:
                print(f"  stage2 step {it}: {float(loss) * 1e6 / B:.1f} mm² avg")

        with torch.no_grad():
            joints_op, vertices = self.mano_forward_21(betas, global_orient, hand_pose, transl)
            diff = joints_op[:, self.op_valid_t] - target_valid
            residual_mm = (diff.norm(dim=-1).mean(dim=-1) * 1000).cpu().numpy()
            # Un-mirror left hands
            joints_unmirrored = joints_op.clone()
            joints_unmirrored[:, :, 0] = joints_unmirrored[:, :, 0] * sign[:, None]
            verts_unmirrored = vertices.clone()
            verts_unmirrored[:, :, 0] = verts_unmirrored[:, :, 0] * sign[:, None]

        return FitResult(
            joints_21_op=joints_unmirrored.cpu().numpy(),
            vertices=verts_unmirrored.cpu().numpy(),
            residual_mm=residual_mm,
        )
