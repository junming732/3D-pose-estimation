import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml


def extract_camera(mat, cam_idx):
    """Extract (T,J,2) array for one camera index from annot.mat."""
    a2 = mat["annot2"][cam_idx]  # (nFrames, nJoints*2)
    nframes, njx2 = a2.shape
    njoints = njx2 // 2
    arr2d = a2.reshape(nframes, njoints, 2).astype(np.float32)

    a3 = mat["annot3"][cam_idx]  # (nFrames, nJoints*3)
    nframes3, njx3 = a3.shape
    njoints3 = njx3 // 3
    arr3d = a3.reshape(nframes3, njoints3, 3).astype(np.float32)
    return arr2d, arr3d


parser = argparse.ArgumentParser(description="Convert MPI-INF-3DHP annot.mat files to NPZ.")
parser.add_argument("--config", type=str, required=True, help="YAML config.")
parser.add_argument("--out-prefix", type=str, default="gt", help="Prefix for output npz files.")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

data_root = Path(cfg.get("data_root", "data")).expanduser().resolve()
print("Searching annot.mat under", data_root)

out2d = {}
out3d = {}

for annot in data_root.glob("S*/Seq*/annot.mat"):
    mat = sio.loadmat(str(annot), squeeze_me=True, struct_as_record=False)
    # For each camera index we create a key matching video_0.avi etc.
    for cam_idx in range(mat["annot2"].shape[0]):
        arr2d, arr3d = extract_camera(mat, cam_idx)

        # build key like S1_Seq1_imageSequence_video_{cam_idx}
        video_path = annot.parent / "imageSequence" / f"video_{cam_idx}.avi"
        rel_path = video_path.relative_to(data_root)
        key = str(rel_path).replace("/", "_").replace("\\", "_").replace(".avi", "")

        out2d[key] = arr2d
        out3d[key] = arr3d
        print(f"Converted {annot} cam{cam_idx} → key {key} 2d shape={arr2d.shape}")

if not out2d:
    print("WARNING: no annot.mat files found!")
else:
    np.savez_compressed(f"{args.out_prefix}_2d_all.npz", **out2d)
    np.savez_compressed(f"{args.out_prefix}_3d_all.npz", **out3d)
    print(f"Wrote {args.out_prefix}_2d_all.npz and {args.out_prefix}_3d_all.npz")
