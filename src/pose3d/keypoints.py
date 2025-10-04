from __future__ import annotations

import cv2
import numpy as np

# MediaPipe import is optional at import time to avoid hard failure in environments without it.
try:
    import mediapipe as mp

    _HAVE_MP = True
except Exception:
    mp = None
    _HAVE_MP = False


class PoseDetector:
    """Wrapper around MediaPipe Pose to return 2D keypoints in image pixel coords."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
    ):
        if not _HAVE_MP:
            raise ImportError(
                "mediapipe is required for PoseDetector. Install with `pip install mediapipe`."
            )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarks = mp.solutions.pose.PoseLandmark

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)
        if not res.pose_landmarks:
            return None, None

        lm = res.pose_landmarks.landmark
        pts = []
        vis = []
        for i in range(len(lm)):
            x = int(lm[i].x * w)
            y = int(lm[i].y * h)
            pts.append([x, y])
            vis.append(lm[i].visibility)
        return np.array(pts, dtype=np.int32), np.array(vis, dtype=np.float32)
