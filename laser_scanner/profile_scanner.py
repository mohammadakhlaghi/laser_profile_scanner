from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(slots=True)
class ScannerConfig:
    input_dir: Path
    file_pattern: str = "*.jpg"
    threshold_factor: float = 0.15
    morph_kernel_size: int = 7
    open_iterations: int = 1
    close_iterations: int = 1


class LaserProfileScanner:
    """Extract centerline points from laser stripe images and stack them as a point cloud."""

    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    def _load_gray(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def _segment_laser(self, gray: np.ndarray) -> np.ndarray:
        threshold = int(np.max(gray) * self.config.threshold_factor)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)
        return binary

    def _skeleton_points(self, binary: np.ndarray) -> np.ndarray:
        skeleton = skeletonize(binary > 0)
        yx = np.argwhere(skeleton)
        if yx.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        xy = yx[:, ::-1].astype(np.float32)
        return xy

    def process_frame(self, image_path: Path, frame_index: int) -> np.ndarray:
        gray = self._load_gray(image_path)
        binary = self._segment_laser(gray)
        xy = self._skeleton_points(binary)
        if xy.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        z = np.full((xy.shape[0], 1), frame_index, dtype=np.float32)
        return np.hstack((xy, z))

    def build_point_cloud(self) -> np.ndarray:
        image_paths = sorted(self.config.input_dir.glob(self.config.file_pattern))
        if not image_paths:
            raise ValueError(f"No files matched pattern '{self.config.file_pattern}' in {self.config.input_dir}")

        frames: list[np.ndarray] = []
        for index, image_path in enumerate(image_paths):
            points = self.process_frame(image_path, index)
            if points.size:
                frames.append(points)

        if not frames:
            raise ValueError("No laser points were extracted from the given images.")

        return np.vstack(frames)

    @staticmethod
    def save_xyz(point_cloud: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, point_cloud, fmt="%.6f", header="x y z", comments="")
