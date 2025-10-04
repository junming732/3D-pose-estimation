from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .keypoints import PoseDetector
from .visualize import draw_skeleton_2d


def _gather_sources(
    data_root: Path,
    pattern: str,
    allow_images: bool = True,
) -> List[Path]:
    """Collect source files (videos or image folders) from a dataset root.

    Args:
        data_root: base folder of the dataset.
        pattern: glob pattern relative to data_root. E.g., "**/*.mp4" or "**/imageSequence/*.avi"
        allow_images: if True, treat a folder containing images as a sequence source.

    Returns:
        List of paths. For videos, the path is the video file. For image-sequences,
        the path is the folder containing images.
    """
    # Only return files that actually exist and are files
    paths = sorted((data_root).glob(pattern))
    # Filter out any directories and deduplicate
    uniq = []
    seen = set()
    for p in paths:
        if p.is_file() and p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _iter_frames_from_video(video_path: Path) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _iter_frames_from_images(folder: Path) -> Iterable[np.ndarray]:
    images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        yield img


def _video_writer_like(input_video: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
    # Attempt to mirror FPS and frame size if available
    fps = input_video.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))


def _video_writer_fixed(
    size_hw: Tuple[int, int], out_path: Path, fps: float = 30.0
) -> cv2.VideoWriter:
    h, w = size_hw
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(
        str(out_path), fourcc, float(fps), (w, w if isinstance(w, int) else int(w))
    )


def _compute_2d_mpjpe(
    pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None
) -> float:
    """Average per-joint pixel error for 2D arrays of shape (T, J, 2)."""
    T = min(len(pred), len(gt))
    pred = pred[:T]
    gt = gt[:T]
    J = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :J]
    gt = gt[:, :J]
    if valid is not None:
        valid = valid[:T, :J].astype(bool)
    else:
        valid = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
    diffs = pred - gt
    err = np.linalg.norm(diffs, axis=-1)  # (T, J)
    err = err[valid]
    if err.size == 0:
        return float("nan")
    return float(np.mean(err))


def _compute_3d_mpjpe(
    pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None
) -> float:
    """Average per-joint position error in millimeters for 3D arrays of shape (T, J, 3)."""
    T = min(len(pred), len(gt))
    pred = pred[:T]
    gt = gt[:T]
    J = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :J]
    gt = gt[:, :J]
    if valid is not None:
        valid = valid[:T, :J].astype(bool)
    else:
        valid = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
    diffs = pred - gt  # assume both in the same units
    err = np.linalg.norm(diffs, axis=-1)  # (T, J)
    err = err[valid]
    if err.size == 0:
        return float("nan")
    return float(np.mean(err))


def _save_npz(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)


def process_sequence(
    source: Path,
    detector: PoseDetector,
    save_npz_to: Optional[Path] = None,
    save_vis_to: Optional[Path] = None,
    eval_gt_2d: Optional[np.ndarray] = None,
    eval_gt_3d: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run inference on one sequence (video or image folder).

    Args:
        source: path to video file or folder of images.
        detector: PoseDetector instance.
        save_npz_to: optional path to save predictions as NPZ.
        save_vis_to: optional path to save visualization MP4 video (2D overlay).
        eval_gt_2d: optional ground-truth 2D keypoints (T, J, 2) in pixel coords.
        eval_gt_3d: optional ground-truth 3D joints (T, J, 3) in same joint layout & units as predictions.

    Returns:
        dict with any computed metrics (e.g., {"mpjpe_2d_px": 4.2})
    """
    # Frame iterator and (optionally) a VideoWriter for visualization
    frames_iter = None
    cap_for_meta = None
    vis_writer = None

    if source.is_file():
        cap_for_meta = cv2.VideoCapture(str(source))
        frames_iter = _iter_frames_from_video(source)
    elif source.is_dir():
        frames_iter = _iter_frames_from_images(source)
    else:
        raise ValueError(f"Unsupported source: {source}")

    preds_list: list[np.ndarray] = []
    vis_list: list[np.ndarray] = []

    try:
        for idx, frame in enumerate(frames_iter):
            # debug show progress
            if idx % 200 == 0:
                print(f"[{source.name}] processing frame {idx}")

            frame_in = cv2.resize(frame, (640, 360))
            keypoints_2d, visibility = detector.detect(frame_in)
            if keypoints_2d is None:
                keypoints_2d = np.full((33, 2), np.nan, dtype=np.float32)
                visibility = np.zeros((33,), dtype=np.float32)
            else:
                # scale keypoints back to original frame size
                h_orig, w_orig = frame.shape[:2]
                h_in, w_in = frame_in.shape[:2]
                sx, sy = w_orig / w_in, h_orig / h_in
                keypoints_2d = keypoints_2d.astype(np.float32)
                keypoints_2d[:, 0] *= sx
                keypoints_2d[:, 1] *= sy

            preds_list.append(keypoints_2d)
            vis_list.append(visibility)

            # Visualization overlay
            if save_vis_to is not None:
                if vis_writer is None:
                    if cap_for_meta is not None and cap_for_meta.isOpened():
                        fps = cap_for_meta.get(cv2.CAP_PROP_FPS) or 30.0
                        w = int(cap_for_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
                        h = int(cap_for_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
                    else:
                        fps = 30.0
                        h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore[attr-defined]
                    save_vis_to.parent.mkdir(parents=True, exist_ok=True)
                    vis_writer = cv2.VideoWriter(str(save_vis_to), fourcc, float(fps), (w, h))
                vis = frame.copy()
                vis = draw_skeleton_2d(vis, preds_list[-1], vis_list[-1])
                cv2.putText(
                    vis, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                vis_writer.write(vis)
    finally:
        if cap_for_meta is not None:
            cap_for_meta.release()
        if vis_writer is not None:
            vis_writer.release()

    preds_2d = np.stack(preds_list, axis=0) if preds_list else np.empty((0, 33, 2), dtype=np.float32)
    visibilities = (
        np.stack(vis_list, axis=0) if vis_list else np.empty((0, 33), dtype=np.float32)
    )

    # Save predictions
    if save_npz_to is not None:
        _save_npz(save_npz_to, keypoints2d=preds_2d, visibility=visibilities)

    # Optional evaluation
    metrics: Dict[str, float] = {}
    if eval_gt_2d is not None and eval_gt_2d.size > 0:
        metrics["mpjpe_2d_px"] = float(_compute_2d_mpjpe(preds_2d, eval_gt_2d))
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run pose3d pipeline on a dataset (videos or image folders)."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config describing dataset.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="If provided and GT is available, compute simple metrics.",
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save 2D overlay videos to output_dir/vis/."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N sequences (0 = no limit)."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("data_root", "data")).expanduser().resolve()
    pattern = cfg.get("pattern", "**/*.mp4")
    output_dir = Path(cfg.get("output_dir", "outputs")).expanduser().resolve()

    # Intrinsics are optional here unless you add depth / 3D recon later.
    """intrinsics = {
        "fx": float(cfg.get("fx", 1000.0)),
        "fy": float(cfg.get("fy", 1000.0)),
        "cx": float(cfg.get("cx", 640.0)),
        "cy": float(cfg.get("cy", 360.0)),
    }"""

    # Optional ground-truth discovery: either a single path per sequence via templating,
    # or a single npz file containing arrays for ALL sequences.
    gt_2d_npz_path = cfg.get(
        "gt_2d_npz", None
    )  # e.g., "data/gt_2d_all.npz" with dict of {seq_key: (T,J,2)}
    gt_3d_npz_path = cfg.get("gt_3d_npz", None)

    # Discover sources
    sources = _gather_sources(data_root, pattern)
    if args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"No sources matched pattern='{pattern}' under '{data_root}'.")

    # Load GT dictionaries if provided
    gt2d_dict: Dict[str, np.ndarray] = {}
    gt3d_dict: Dict[str, np.ndarray] = {}
    if gt_2d_npz_path:
        data = np.load(str(Path(gt_2d_npz_path)), allow_pickle=True)
        # allow both {key: array} or a single 'keypoints2d' with shape (T,J,2)
        if "keypoints2d" in data:
            gt2d_dict["__single__"] = data["keypoints2d"]
        else:
            for k in data.files:
                gt2d_dict[str(k)] = data[k]
    if gt_3d_npz_path:
        data = np.load(str(Path(gt_3d_npz_path)), allow_pickle=True)
        if "joints3d" in data:
            gt3d_dict["__single__"] = data["joints3d"]
        else:
            for k in data.files:
                gt3d_dict[str(k)] = data[k]

    detector = PoseDetector()

    per_seq_metrics: Dict[str, Dict[str, float]] = {}
    for src in sources:
        src = Path(src)
        # Key used to look up GT in dicts (by filename stem)
        key = src.stem

        # build a unique key from the path relative to data_root
        rel_path = src.relative_to(data_root)  # e.g. S1/Seq1/imageSequence/video_0.avi
        # make it safe for filenames
        key = (
            str(rel_path)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".avi", "")
            .replace(".mp4", "")
        )

        out_npz = output_dir / "preds" / f"{key}.npz"
        out_vis = (output_dir / "vis" / f"{key}.mp4") if args.save_vis else None

        gt2d = gt2d_dict.get(key) if key in gt2d_dict else gt2d_dict.get("__single__")
        gt3d = gt3d_dict.get(key) if key in gt3d_dict else gt3d_dict.get("__single__")

        metrics = process_sequence(
            source=src,
            detector=detector,
            save_npz_to=out_npz,
            save_vis_to=out_vis,
            eval_gt_2d=gt2d if args.eval else None,
            eval_gt_3d=gt3d if args.eval else None,
        )
        per_seq_metrics[key] = metrics

    # Save metrics summary if any
    if args.eval and per_seq_metrics:
        out_metrics = output_dir / "metrics_summary.json"
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(out_metrics, "w") as f:
            json.dump(per_seq_metrics, f, indent=2)
        print(f"Wrote metrics to: {out_metrics}")

    print(f"Done. Processed {len(sources)} sequence(s). Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
