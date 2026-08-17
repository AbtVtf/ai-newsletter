"""ARDY actor: the cskel27 skinned body driven by a generated motion npz.

Numpy-only so it loads in the jarvis venv next to render_face.py. The
skinning formula matches game/export_skin.py in the ardy repo:

    v'(t) = sum_k w_k * (G_{j_k}(t) @ inv_bind_{j_k}) @ v_bind

where G is built from the motion file's global_rot_mats and posed_joints.
Motion is 20 fps; pose() lerps the skinning matrices between frames so a
30 fps render does not step. Lerped rotation matrices are slightly
non-orthonormal between frames; at 50 ms deltas the error is invisible.
"""

from pathlib import Path

import numpy as np

# ardy is a sibling repo of ai-newsletter
SKIN_DEFAULT = (Path(__file__).resolve().parents[2] / "ardy" / "ardy" /
                "assets" / "skeletons" / "cskel27" / "skin_standard.npz")


class Actor:
    def __init__(self, motion_path, skin_path=None, color=(0.50, 0.60, 0.52)):
        skin = np.load(skin_path or SKIN_DEFAULT)
        self.bind = skin["bind_vertices"].astype(np.float32)        # (V,3)
        self.faces = skin["faces"].astype(np.int32)                 # (F,3)
        self.lbs_idx = skin["lbs_indices"].astype(np.int64)         # (V,5)
        self.lbs_w = skin["lbs_weights"].astype(np.float32)         # (V,5)
        self.joint_names = [str(n) for n in skin["rig_joint_names"]]
        inv_bind = np.linalg.inv(skin["bind_rig_transform"].astype(np.float64))

        mo = np.load(motion_path, allow_pickle=True)
        rot = mo["global_rot_mats"].astype(np.float64)              # (T,J,3,3)
        pos = mo["posed_joints"].astype(np.float64)                 # (T,J,3)
        self.fps = float(mo["fps"])
        self.num_frames = rot.shape[0]
        self.duration = self.num_frames / self.fps
        self.joints = pos.astype(np.float32)
        self.text = str(mo["text"]) if "text" in mo.files else ""

        T, J = rot.shape[:2]
        G = np.tile(np.eye(4), (T, J, 1, 1))
        G[:, :, :3, :3] = rot
        G[:, :, :3, 3] = pos
        self.global_tf = G.astype(np.float32)                       # (T,J,4,4)
        S = np.einsum("tjab,jbc->tjac", G, inv_bind)
        self.skin_R = S[:, :, :3, :3].astype(np.float32)            # (T,J,3,3)
        self.skin_t = S[:, :, :3, 3].astype(np.float32)             # (T,J,3)

        self.nverts = self.bind.shape[0]
        self.flat_idx = self.faces.reshape(-1)
        self.colors = np.tile(np.asarray(color, dtype=np.float32),
                              (self.nverts, 1))

    def _frame_pair(self, t, loop=False):
        f = t * self.fps
        if loop:
            f = f % self.num_frames
        f = min(max(f, 0.0), self.num_frames - 1)
        f0 = int(f)
        f1 = min(f0 + 1, self.num_frames - 1)
        return f0, f1, np.float32(f - f0)

    def pose(self, t, loop=False):
        """Skinned vertex positions (V,3) float32 at time t seconds."""
        f0, f1, a = self._frame_pair(t, loop)
        R = self.skin_R[f0] * (1 - a) + self.skin_R[f1] * a
        tr = self.skin_t[f0] * (1 - a) + self.skin_t[f1] * a
        out = np.zeros_like(self.bind)
        for k in range(self.lbs_idx.shape[1]):
            j = self.lbs_idx[:, k]
            out += self.lbs_w[:, k, None] * (
                np.einsum("vab,vb->va", R[j], self.bind) + tr[j])
        return out

    def joint_matrix(self, t, name, loop=False):
        """World 4x4 of a joint at time t — for parenting props."""
        j = self.joint_names.index(name)
        f0, f1, a = self._frame_pair(t, loop)
        return self.global_tf[f0, j] * (1 - a) + self.global_tf[f1, j] * a

    def bounds(self, stride=4):
        """(lo, hi) over the whole clip, for framing. World units (meters)."""
        pts = self.joints[::stride].reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)
