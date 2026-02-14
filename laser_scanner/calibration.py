from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class CalibrationResult:
    """Container for OpenCV camera calibration outputs."""

    reprojection_error: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors: list[np.ndarray]


class CameraCalibrator:
    """Calibrate a camera using chessboard images."""

    def __init__(self, chessboard_size: tuple[int, int] = (5, 6), square_size: float = 1.0) -> None:
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def _build_object_points(self) -> np.ndarray:
        cols, rows = self.chessboard_size
        object_points = np.zeros((rows * cols, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return object_points * self.square_size

    def calibrate(self, image_paths: Iterable[Path]) -> CalibrationResult:
        object_points_template = self._build_object_points()
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        image_size: tuple[int, int] | None = None

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if not found:
                continue

            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            object_points.append(object_points_template)
            image_points.append(refined)

        if not object_points or image_size is None:
            raise ValueError("No valid chessboard detections found. Verify image path and chessboard size.")

        reprojection_error, camera_matrix, distortion, rotation, translation = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )

        return CalibrationResult(
            reprojection_error=float(reprojection_error),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            rotation_vectors=rotation,
            translation_vectors=translation,
        )

    @staticmethod
    def save(result: CalibrationResult, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            reprojection_error=result.reprojection_error,
            camera_matrix=result.camera_matrix,
            distortion_coefficients=result.distortion_coefficients,
            rotation_vectors=np.array(result.rotation_vectors, dtype=object),
            translation_vectors=np.array(result.translation_vectors, dtype=object),
        )
