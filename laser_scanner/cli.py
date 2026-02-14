from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import CameraCalibrator
from .profile_scanner import LaserProfileScanner, ScannerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laser profile scanner utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Calibrate camera from chessboard images")
    cal.add_argument("--images", type=Path, required=True, help="Directory containing calibration images")
    cal.add_argument("--pattern", default="*.jpg", help="Glob pattern for calibration images")
    cal.add_argument("--cols", type=int, default=5, help="Chessboard inner corners (columns)")
    cal.add_argument("--rows", type=int, default=6, help="Chessboard inner corners (rows)")
    cal.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size")
    cal.add_argument("--output", type=Path, default=Path("outputs/calibration.npz"))

    scan = sub.add_parser("scan", help="Extract laser profile point cloud")
    scan.add_argument("--images", type=Path, required=True, help="Directory containing scan frames")
    scan.add_argument("--pattern", default="*.jpg", help="Glob pattern for frame images")
    scan.add_argument("--threshold-factor", type=float, default=0.15)
    scan.add_argument("--kernel-size", type=int, default=7)
    scan.add_argument("--output", type=Path, default=Path("outputs/point_cloud.xyz"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "calibrate":
        calibrator = CameraCalibrator((args.cols, args.rows), square_size=args.square_size)
        result = calibrator.calibrate(sorted(args.images.glob(args.pattern)))
        calibrator.save(result, args.output)
        print(f"Saved calibration to {args.output} (RMS error={result.reprojection_error:.6f})")
        return

    if args.command == "scan":
        scanner = LaserProfileScanner(
            ScannerConfig(
                input_dir=args.images,
                file_pattern=args.pattern,
                threshold_factor=args.threshold_factor,
                morph_kernel_size=args.kernel_size,
            )
        )
        cloud = scanner.build_point_cloud()
        scanner.save_xyz(cloud, args.output)
        print(f"Saved {cloud.shape[0]} points to {args.output}")
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
