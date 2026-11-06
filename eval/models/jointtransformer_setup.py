"""JointTransformer environment bootstrap for zero-shot eval.

Eliminates the two heavy dependencies of the JointTransformer repo so we
can run the TransformerSF single-frame model on Aria val + POV-Surgery
without building pytorch3d from source or downloading ARCTIC object
templates (which the model won't use for surgical data anyway).

Call `bootstrap()` BEFORE importing `src.models.transformer_sf.model`.
"""
import os
import sys
import textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- pytorch3d stub (pure-torch) ----
def _install_pytorch3d_stub(site_dir=None):
    if site_dir is None:
        import site
        site_dir = site.getsitepackages()[0]
    root = os.path.join(site_dir, 'pytorch3d')
    if os.path.isdir(root):
        return root
    os.makedirs(os.path.join(root, 'transforms'), exist_ok=True)
    os.makedirs(os.path.join(root, 'ops'), exist_ok=True)
    open(os.path.join(root, '__init__.py'), 'w').write(
        "__version__ = '0.0.0-stub'\n")
    open(os.path.join(root, 'transforms', '__init__.py'), 'w').write(
        'from .rotation_conversions import *\n'
        'from .rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix\n')
    open(os.path.join(root, 'transforms', 'rotation_conversions.py'), 'w').write(
        _ROT_STUB_SRC)
    open(os.path.join(root, 'ops', '__init__.py'), 'w').write(
        "def knn_points(*args, **kwargs):\n"
        "    raise NotImplementedError('stub')\n")
    return root

_ROT_STUB_SRC = r"""
import torch
import torch.nn.functional as F

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix(q):
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(q.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix):
    bd = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(bd + (9,)), -1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
    ], dim=-1))
    qr = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(q_abs)
    qc = qr / (2.0 * q_abs[..., None].max(flr))
    return qc[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(bd + (4,))

def axis_angle_to_quaternion(aa):
    a = torch.norm(aa, p=2, dim=-1, keepdim=True)
    ha = a * 0.5
    eps = 1e-6
    small = a.abs() < eps
    s = torch.empty_like(a)
    s[~small] = torch.sin(ha[~small]) / a[~small]
    s[small] = 0.5 - (a[small] ** 2) / 48
    return torch.cat([torch.cos(ha), aa * s], dim=-1)

def quaternion_to_axis_angle(q):
    n = torch.norm(q[..., 1:], p=2, dim=-1, keepdim=True)
    ha = torch.atan2(n, q[..., :1])
    a = 2 * ha
    eps = 1e-6
    small = a.abs() < eps
    s = torch.empty_like(a)
    s[~small] = torch.sin(ha[~small]) / a[~small]
    s[small] = 0.5 - (a[small] * a[small]) / 48
    return q[..., 1:] / s

def axis_angle_to_matrix(aa): return quaternion_to_matrix(axis_angle_to_quaternion(aa))
def matrix_to_axis_angle(mat): return quaternion_to_axis_angle(matrix_to_quaternion(mat))

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(mat):
    return mat[..., :2, :].clone().reshape(mat.shape[:-2] + (6,))
"""

# ---- ArtiHead / ObjectTensors stubs (no ARCTIC object templates) ----
def _install_arti_stubs():
    import common.object_tensors as ot
    class _StubObjectTensors(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.obj_tensors = {}
        def forward_7d_batch(self, *a, **kw):
            return {}
    ot.ObjectTensors = _StubObjectTensors

    import src.nets.obj_heads.obj_head as oh
    class _StubArtiHead(nn.Module):
        def __init__(self, focal_length, img_res):
            super().__init__()
            self.focal_length = focal_length
            self.img_res = img_res
        def forward(self, rot, angle, query_names, cam, K):
            return {'rot': rot.detach(), 'angle': angle.detach(),
                    'cam_t.wp': cam.detach()}
    oh.ArtiHead = _StubArtiHead


def _link_mano(jt_repo, mano_src_dir='/content/mano_assets'):
    dst = os.path.join(jt_repo, 'data', 'body_models', 'mano')
    os.makedirs(dst, exist_ok=True)
    for f in ('MANO_RIGHT.pkl', 'MANO_LEFT.pkl'):
        s = os.path.join(mano_src_dir, f); d = os.path.join(dst, f)
        if os.path.exists(s) and not os.path.exists(d):
            os.symlink(s, d)


def bootstrap(jt_repo='/content/JointTransformer',
              mano_src_dir='/content/mano_assets'):
    """Make TransformerSF importable + loadable. Idempotent."""
    _install_pytorch3d_stub()
    _link_mano(jt_repo, mano_src_dir)
    if jt_repo not in sys.path:
        sys.path.insert(0, jt_repo)
    os.chdir(jt_repo)
    # arti stubs must go after path mangling
    _install_arti_stubs()


def load_transformer_sf(ckpt_path, device='cuda'):
    """Returns (model, hparams_dict). Model is .eval() on device."""
    from types import SimpleNamespace
    from src.models.transformer_sf.model import TransformerSF
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    args = SimpleNamespace(**hp)
    model = TransformerSF(backbone=hp['backbone'],
                          focal_length=hp['focal_length'],
                          img_res=hp['img_res'],
                          args=args)
    sd = {k[len('model.'):] if k.startswith('model.') else k: v
          for k, v in ck['state_dict'].items()}
    m, u = model.load_state_dict(sd, strict=False)
    # 28 unexpected MANO buffers are expected (we load MANO from .pkl directly).
    assert len(m) == 0, f'missing keys: {m[:5]}...'
    model = model.eval().to(device)
    return model, hp


# MANO 778-vertex indices for fingertips (right/left hand share indices)
MANO_TIP_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky


def to_openpose_21(joints16, vertices778):
    """Extend MANO 16-joint skeleton to 21-joint OpenPose convention.

    MANO joints: wrist + 3 per finger × 5 = 16 (no fingertips).
    OpenPose: wrist + 4 per finger × 5 = 21 (adds tips).
    We concat fingertips from the predicted vertices.

    Returns joints21 of shape (..., 21, 3).
    """
    tips = vertices778[..., MANO_TIP_IDS, :]  # (..., 5, 3)
    # OpenPose order: wrist, thumb[0..3], index[0..3], ..., pinky[0..3]
    # MANO order (smplx): wrist, index[0..2], middle[0..2], pinky[0..2], ring[0..2], thumb[0..2]
    # Reorder MANO → OpenPose base: wrist + thumb(3) + index(3) + middle(3) + ring(3) + pinky(3)
    # smplx MANO indices: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
    op_idx = [0,
              13, 14, 15,  # thumb 3 joints
              1, 2, 3,     # index
              4, 5, 6,     # middle
              10, 11, 12,  # ring
              7, 8, 9]     # pinky
    j_base = joints16[..., op_idx, :]        # (..., 16, 3)
    # Append fingertip in order thumb, index, middle, ring, pinky
    tip_order = [0, 1, 2, 3, 4]
    # Interleave tips to fit 21-joint layout: wrist, thumb_0..3+tip, index_0..3+tip,...
    chunks = [j_base[..., :1, :]]  # wrist
    for fi, fname in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
        start = 1 + fi * 3
        chunks.append(j_base[..., start:start+3, :])
        chunks.append(tips[..., fi:fi+1, :])
    return torch.cat(chunks, dim=-2)
