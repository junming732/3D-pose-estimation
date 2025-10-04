from __future__ import annotations

import cv2
import numpy as np

# Simple skeleton edges for MediaPipe's 33 landmarks (subset)
# You can expand this list for a more complete skeleton.
EDGES = [
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (11, 23),
    (12, 24),  # torso
    (23, 25),
    (25, 27),  # left leg
    (24, 26),
    (26, 28),  # right leg
    (11, 12),  # shoulders
    (23, 24),  # hips
]


def draw_skeleton_2d(img, keypoints, visibility=None):
    keypoints = np.asarray(keypoints)
    J = keypoints.shape[0]
    if visibility is None:
        visibility = np.ones((J,), dtype=np.float32)
    else:
        visibility = np.asarray(visibility, dtype=np.float32)

    def joint_valid(i):
        return i >= 0 and i < J and visibility[i] > 0.5 and np.isfinite(keypoints[i]).all()

    # draw edges
    for i, j in EDGES:
        if joint_valid(i) and joint_valid(j):
            p1 = (int(round(keypoints[i, 0])), int(round(keypoints[i, 1])))
            p2 = (int(round(keypoints[j, 0])), int(round(keypoints[j, 1])))
            cv2.line(img, p1, p2, (0, 255, 0), 2)

    # draw joints
    for i in range(J):
        if joint_valid(i):
            p = (int(round(keypoints[i, 0])), int(round(keypoints[i, 1])))
            cv2.circle(img, p, 3, (0, 0, 255), -1)

    return img
