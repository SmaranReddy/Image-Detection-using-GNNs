"""
MediaPipe Pose feature extractor for interaction-aware pose features.

Extracts compact 20-dim interaction-relevant features from human pose keypoints.
Designed to augment the relation MLP with body interaction cues for:
  - sitting vs standing near
  - riding vs beside
  - holding vs touching
  - looking at vs interacting with

Features are normalised and interaction-relevant (not raw keypoints).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


POSE_FEATURE_DIM = 20
POSE_OBJECT_FEATURE_DIM = 7


class PoseExtractor:
    """
    Lightweight wrapper around MediaPipe Pose.

    Extracts 20 compact interaction-relevant features per detected person.
    All features are normalised for direct MLP consumption.

    Features (20-dim):
        0-1:   torso_dir_x, torso_dir_y        — torso orientation
        2:     avg_knee_angle_norm              — knee bend (0=straight, 1=bent)
        3-4:   left_wrist_rel_x, y              — wrist relative to body center
        5-6:   right_wrist_rel_x, y
        7-8:   left_elbow_rel_x, y              — elbow relative to body center
        9-10:  right_elbow_rel_x, y
        11-12: head_dir_x, head_dir_y           — head direction from neck
        13:    sitting_score                     — posture score (0-1)
        14:    riding_score                      — posture score (0-1)
        15:    hand_distance                     — wrist separation (normalised)
        16:    arm_reach                         — max arm extension
        17-18: body_center_rel_x, y             — body center vs box center
        19:    height_ratio                      — pose height / box height
    """

    _pose = None
    _extraction_stats = {"attempts": 0, "success": 0, "fail_no_detection": 0, "fail_unavailable": 0}

    @classmethod
    def _lazy_load(cls) -> None:
        if cls._pose is None:
            try:
                import mediapipe as mp
                cls._pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                )
                print("[PoseExtractor] MediaPipe Pose loaded.")
            except ImportError:
                print("[PoseExtractor] WARNING: mediapipe not installed. Pose features disabled.")
                cls._pose = False

    @classmethod
    def is_available(cls) -> bool:
        cls._lazy_load()
        return cls._pose is not None and cls._pose is not False

    @torch.no_grad()
    def extract_pose_features(
        self,
        image: Image.Image,
        person_box: Tuple[float, float, float, float],
    ) -> Optional[torch.Tensor]:
        """
        Extract compact 20-dim pose features for a detected person.

        Args:
            image: PIL Image (RGB)
            person_box: (x1, y1, x2, y2) in pixel coordinates

        Returns:
            Tensor (POSE_FEATURE_DIM,) or None if no person/pose detected.
        """
        self._lazy_load()
        PoseExtractor._extraction_stats["attempts"] += 1
        if self._pose is False:
            PoseExtractor._extraction_stats["fail_unavailable"] += 1
            return None

        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            PoseExtractor._extraction_stats["fail_no_detection"] += 1
            return None
        margin_x, margin_y = w * 0.2, h * 0.2
        crop_box = (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(image.width, x2 + margin_x),
            min(image.height, y2 + margin_y),
        )
        person_crop = image.crop(crop_box)
        if person_crop.width == 0 or person_crop.height == 0:
            PoseExtractor._extraction_stats["fail_no_detection"] += 1
            return None
        results = self._pose.process(np.array(person_crop))

        if not results or results.pose_landmarks is None:
            PoseExtractor._extraction_stats["fail_no_detection"] += 1
            return None

        PoseExtractor._extraction_stats["success"] += 1
        L = results.pose_landmarks.landmark

        def kp(idx):
            return np.array([L[idx].x, L[idx].y, L[idx].z], dtype=np.float32)

        nose = kp(0)
        lsho = kp(11); rsho = kp(12)
        lelb = kp(13); relb = kp(14)
        lwri = kp(15); rwri = kp(16)
        lhip = kp(23); rhip = kp(24)
        lkne = kp(25); rkne = kp(26)
        lank = kp(27); rank = kp(28)

        body_center = (lhip + rhip + lsho + rsho) / 4.0
        shoulder_mid = (lsho + rsho) / 2.0
        hip_mid = (lhip + rhip) / 2.0
        torso_vec = shoulder_mid - hip_mid

        torso_norm_v = np.linalg.norm(torso_vec[:2])
        if torso_norm_v > 1e-6:
            torso_dir = torso_vec[:2] / torso_norm_v
        else:
            torso_dir = np.zeros(2, dtype=np.float32)

        def angle_at(b, a, c):
            ba = a - b
            bc = c - b
            dot = np.dot(ba, bc)
            n = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
            return np.arccos(np.clip(dot / n, -1.0, 1.0))

        left_knee_angle = angle_at(lkne, lhip, lank)
        right_knee_angle = angle_at(rkne, rhip, rank)
        avg_knee = (left_knee_angle + right_knee_angle) / 2.0

        left_wrist_rel = lwri[:2] - body_center[:2]
        right_wrist_rel = rwri[:2] - body_center[:2]
        left_elbow_rel = lelb[:2] - body_center[:2]
        right_elbow_rel = relb[:2] - body_center[:2]

        neck = shoulder_mid
        head_dir = nose[:2] - neck[:2]
        head_norm = np.linalg.norm(head_dir)
        if head_norm > 1e-6:
            head_dir_n = head_dir / head_norm
        else:
            head_dir_n = np.zeros(2, dtype=np.float32)

        sitting_score = float(np.clip(1.0 - (avg_knee / np.pi), 0.0, 1.0))

        leg_spread = float(np.linalg.norm(lkne[:2] - rkne[:2]))
        hip_height = float(hip_mid[1])
        riding_score = float(np.clip(leg_spread * (1.0 - hip_height) * 2.0, 0.0, 1.0))

        hand_distance = float(np.linalg.norm(lwri[:2] - rwri[:2]))
        left_arm_len = float(np.linalg.norm(lwri[:2] - lsho[:2]))
        right_arm_len = float(np.linalg.norm(rwri[:2] - rsho[:2]))
        arm_reach = max(left_arm_len, right_arm_len)

        box_center = np.array([
            (x1 + x2) / (2.0 * image.width),
            (y1 + y2) / (2.0 * image.height),
        ], dtype=np.float32)
        body_center_rel = body_center[:2] - box_center

        head_top = nose[1]
        foot_bottom = max(lank[1], rank[1])
        pose_height = abs(foot_bottom - head_top)
        box_height_n = (y2 - y1) / max(image.height, 1.0)
        height_ratio = pose_height / max(box_height_n, 1e-6)

        features = np.array([
            torso_dir[0], torso_dir[1],
            float(avg_knee / np.pi),
            left_wrist_rel[0], left_wrist_rel[1],
            right_wrist_rel[0], right_wrist_rel[1],
            left_elbow_rel[0], left_elbow_rel[1],
            right_elbow_rel[0], right_elbow_rel[1],
            head_dir_n[0], head_dir_n[1],
            sitting_score,
            riding_score,
            hand_distance,
            arm_reach,
            body_center_rel[0], body_center_rel[1],
            float(np.clip(height_ratio, 0.0, 2.0)),
        ], dtype=np.float32)

        return torch.from_numpy(features)

    @torch.no_grad()
    def extract_pose_object_features(
        self,
        image: Image.Image,
        person_box: Tuple[float, float, float, float],
        obj_box: Tuple[float, float, float, float],
    ) -> Optional[torch.Tensor]:
        """Extract interaction-aware pose-object relative features.

        Computes distances between body keypoints and object bounding box
        to capture interaction semantics (sitting, riding, holding, etc.).

        Features (7-dim):
          0: wrist_to_obj_center    — min wrist-to-object-center distance
          1: ankle_to_obj_center    — min ankle-to-object-center distance
          2: hip_to_obj_bottom      — hip center to object bottom (sitting cue)
          3: shoulder_to_obj_center — shoulder center to object center
          4: knee_to_obj_center     — min knee-to-object-center distance
          5: head_to_obj_center     — nose to object center (looking at cue)
          6: limb_obj_overlap       — count of limb keypoints inside obj bbox

        Args:
            image: PIL Image (RGB)
            person_box: (x1, y1, x2, y2) in pixel coordinates
            obj_box: (x1, y1, x2, y2) in pixel coordinates

        Returns:
            Tensor (POSE_OBJECT_FEATURE_DIM,) or None if pose detection fails.
        """
        self._lazy_load()
        if self._pose is False:
            return None

        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        margin_x, margin_y = w * 0.2, h * 0.2
        crop_box = (
            max(0, x1 - margin_x), max(0, y1 - margin_y),
            min(image.width, x2 + margin_x), min(image.height, y2 + margin_y),
        )
        person_crop = image.crop(crop_box)
        if person_crop.width == 0 or person_crop.height == 0:
            return None
        results = self._pose.process(np.array(person_crop))
        if not results or results.pose_landmarks is None:
            return None

        L = results.pose_landmarks.landmark

        ox1, oy1, ox2, oy2 = obj_box
        obj_center_x = (ox1 + ox2) / 2.0
        obj_center_y = (oy1 + oy2) / 2.0
        img_w, img_h = image.width, image.height
        denom = max(img_w, img_h, 1.0)

        # Normalize object box
        obj_top = oy1 / max(img_h, 1.0)
        obj_bottom = oy2 / max(img_h, 1.0)

        def kp(idx):
            # MediaPipe coordinates are already in [0,1] relative to image
            # But the crop changes the coordinate space.
            # We convert back to image coordinates.
            return (
                crop_box[0] + L[idx].x * (crop_box[2] - crop_box[0]),
                crop_box[1] + L[idx].y * (crop_box[3] - crop_box[1]),
            )

        # Key landmarks
        nose = kp(0)
        lsho, rsho = kp(11), kp(12)
        lelb, relb = kp(13), kp(14)
        lwri, rwri = kp(15), kp(16)
        lhip, rhip = kp(23), kp(24)
        lkne, rkne = kp(25), kp(26)
        lank, rank = kp(27), kp(28)

        # Helper: distance between a keypoint and object center
        def _dist(keypoint):
            dx = (keypoint[0] - obj_center_x) / denom
            dy = (keypoint[1] - obj_center_y) / denom
            return float(np.sqrt(dx * dx + dy * dy))

        # 1. Wrist-to-object-center (min of both wrists)
        wrist_dist = min(_dist(lwri), _dist(rwri))

        # 2. Ankle-to-object-center (min of both ankles)
        ankle_dist = min(_dist(lank), _dist(rank))

        # 3. Hip-to-object-bottom (vertical distance from hip center to obj bottom)
        hip_center_y = (lhip[1] + rhip[1]) / 2.0
        hip_to_obj_bottom = (hip_center_y - obj_bottom) / denom

        # 4. Shoulder-to-object-center (min of both shoulders)
        shoulder_dist = min(_dist(lsho), _dist(rsho))

        # 5. Knee-to-object-center (min of both knees)
        knee_dist = min(_dist(lkne), _dist(rkne))

        # 6. Head-to-object-center (looking at)
        head_dist = _dist(nose)

        # 7. Limb-object overlap: count how many limb keypoints fall inside obj bbox
        limb_kps = [lwri, rwri, lelb, relb, lkne, rkne, lank, rank]
        overlap_count = 0
        for kx, ky in limb_kps:
            if ox1 <= kx <= ox2 and oy1 <= ky <= oy2:
                overlap_count += 1
        limb_overlap = overlap_count / max(len(limb_kps), 1)

        features = np.array([
            float(np.clip(wrist_dist, 0.0, 2.0)),
            float(np.clip(ankle_dist, 0.0, 2.0)),
            float(np.clip(hip_to_obj_bottom, -1.0, 1.0)),
            float(np.clip(shoulder_dist, 0.0, 2.0)),
            float(np.clip(knee_dist, 0.0, 2.0)),
            float(np.clip(head_dist, 0.0, 2.0)),
            float(np.clip(limb_overlap, 0.0, 1.0)),
        ], dtype=np.float32)

        return torch.from_numpy(features)

    @classmethod
    def print_stats(cls) -> None:
        s = cls._extraction_stats
        total = s["attempts"]
        ok = s["success"]
        no_detect = s["fail_no_detection"]
        unavail = s["fail_unavailable"]
        print(f"[PoseExtractor Stats] Attempts={total}, Success={ok}, "
              f"NoDetection={no_detect}, Unavailable={unavail}")
        if total > 0:
            print(f"[PoseExtractor Stats] Success rate: {100.0 * ok / max(total, 1):.1f}%")
