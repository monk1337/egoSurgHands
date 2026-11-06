"""
Minimal pytorch3d stub for WildHands.

WildHands imports only `pytorch3d.transforms.rotation_conversions` and uses
three functions: axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix.
Building pytorch3d from source on Colab takes 15+ min. This stub avoids it.

Usage:
    from phase0.pytorch3d_stub import install
    install()  # must be called BEFORE importing WildHands
"""
import sys
import types
import torch
import torch.nn.functional as F


def _quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _axis_angle_to_matrix(axis_angle):
    """Convert (..., 3) axis-angle to (..., 3, 3) rotation matrices via Rodrigues."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small = angles.abs() < eps
    sin_half_over_angles = torch.empty_like(angles)
    sin_half_over_angles[~small] = torch.sin(half_angles[~small]) / angles[~small]
    sin_half_over_angles[small] = 0.5 - (angles[small] ** 2) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angles], dim=-1)
    return _quaternion_to_matrix(quaternions)


def _matrix_to_rotation_6d(matrix):
    """Drop last row of a 3x3 rotation matrix → (..., 6)."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def _rotation_6d_to_matrix(d6):
    """Reconstruct rotation matrix from 6D continuous representation."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def install():
    """Install fake pytorch3d modules into sys.modules."""
    if "pytorch3d" in sys.modules:
        return  # real pytorch3d already installed, leave it
    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms_mod = types.ModuleType("pytorch3d.transforms")
    rc_mod = types.ModuleType("pytorch3d.transforms.rotation_conversions")
    setattr(rc_mod, "axis_angle_to_matrix", _axis_angle_to_matrix)
    setattr(rc_mod, "matrix_to_rotation_6d", _matrix_to_rotation_6d)
    setattr(rc_mod, "rotation_6d_to_matrix", _rotation_6d_to_matrix)
    setattr(rc_mod, "quaternion_to_matrix", _quaternion_to_matrix)
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod
    sys.modules["pytorch3d.transforms.rotation_conversions"] = rc_mod
