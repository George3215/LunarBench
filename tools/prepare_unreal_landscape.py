#!/usr/bin/env python3
"""Convert the bundled USD grid to a native Unreal Landscape heightmap.

The source mesh is already a regular 1 cm grid, but importing it as a static
mesh would create 15.6 million triangles.  Unreal Landscape stores the same
surface compactly and supplies terrain collision/LOD support.

Run this script with the ``moonunreal-mujoco`` conda environment because that
environment contains OpenUSD and NumPy.  The output is a little-endian R16
heightmap consumed by the accompanying Unreal editor commandlet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = REPO_ROOT / "assets/environments/lunar/terrain/landscape_cropped" / "Props" / "Landscape_1.usd"
DEFAULT_OUTPUT = REPO_ROOT / "ue/import_data"
DEFAULT_PRIM = "/Root/Landscape_1"

# UE Landscape stores local height in a uint16 with 32768 representing zero.
# At actor Z scale 1.0, one encoded unit is 1/128 cm and the supported range is
# approximately [-256, 256) cm.  The source is entirely inside that range.
UE_HEIGHT_MID = 32768.0
UE_HEIGHT_UNITS_PER_CM_AT_Z_SCALE_ONE = 128.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Landscape_1.usd as a native UE Landscape R16 file."
    )
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--prim", default=DEFAULT_PRIM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    usd_path = args.usd.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not usd_path.is_file():
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"USD could not open: {usd_path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("The converter requires a Z-up USD stage")
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(metres_per_unit, 0.01):
        raise ValueError(
            "This asset is expected to use centimetres (metersPerUnit=0.01), "
            f"but the USD reports {metres_per_unit}"
        )

    prim = stage.GetPrimAtPath(args.prim)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"USD prim is not a mesh: {args.prim}")
    transform = np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    )
    if not np.allclose(transform, np.eye(4), atol=1e-12):
        raise ValueError("The source mesh transform must be identity")

    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Unexpected point array shape: {points.shape}")

    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    width = len(x_values)
    height = len(y_values)
    if width * height != len(points):
        raise ValueError(
            f"Mesh is not a complete XY height grid: {width} x {height} != {len(points)}"
        )
    if width != 2795 or height != 2795:
        raise ValueError(f"Expected the bundled 2795 x 2795 terrain, got {width} x {height}")

    x_step = np.diff(x_values)
    y_step = np.diff(y_values)
    if not np.allclose(x_step, 1.0) or not np.allclose(y_step, 1.0):
        raise ValueError("Expected a regular 1 cm XY grid")

    columns = np.searchsorted(x_values, points[:, 0])
    source_rows = np.searchsorted(y_values, points[:, 1])
    flat_index = source_rows.astype(np.int64) * width + columns
    occupancy = np.bincount(flat_index, minlength=height * width)
    if occupancy.min() != 1 or occupancy.max() != 1:
        raise ValueError("The grid has missing or duplicate XY samples")

    source_z_cm = np.empty((height, width), dtype=np.float32)
    source_z_cm[source_rows, columns] = points[:, 2]

    # USD is right-handed while Unreal is left-handed.  X and Z are retained;
    # Y is negated.  Row zero therefore comes from the maximum USD Y value.
    unreal_z_cm = source_z_cm[::-1, :]
    encoded_float = (
        unreal_z_cm * UE_HEIGHT_UNITS_PER_CM_AT_Z_SCALE_ONE + UE_HEIGHT_MID
    )
    if encoded_float.min() < 0.0 or encoded_float.max() > 65535.0:
        raise ValueError(
            "Terrain height exceeds UE Landscape range at Z scale 1.0: "
            f"[{float(unreal_z_cm.min()):.6f}, {float(unreal_z_cm.max()):.6f}] cm"
        )
    encoded = np.rint(encoded_float).astype("<u2")
    reconstructed_z_cm = (
        encoded.astype(np.float32) - UE_HEIGHT_MID
    ) / UE_HEIGHT_UNITS_PER_CM_AT_Z_SCALE_ONE
    max_error_cm = float(np.max(np.abs(reconstructed_z_cm - unreal_z_cm)))

    r16_path = output_dir / "Landscape_1_2795x2795.r16"
    write_atomic(r16_path, encoded.tobytes(order="C"))

    x_min_cm, x_max_cm = float(x_values[0]), float(x_values[-1])
    # After handedness conversion: UE Y = -USD Y.
    ue_y_min_cm, ue_y_max_cm = float(-y_values[-1]), float(-y_values[0])
    metadata = {
        "source_usd": str(usd_path),
        "source_prim": args.prim,
        "source_sha256": sha256(usd_path),
        "source_points": int(len(points)),
        "source_units": "centimetres",
        "source_up_axis": "Z",
        "output_r16": str(r16_path),
        "output_byte_order": "little-endian",
        "resolution": [width, height],
        "landscape_layout": {
            "component_count": [22, 22],
            "sections_per_component": 1,
            "quads_per_section": 127,
        },
        "actor_location_cm": [x_min_cm, ue_y_min_cm, 0.0],
        "actor_scale": [1.0, 1.0, 1.0],
        "world_bounds_cm": {
            "x": [x_min_cm, x_max_cm],
            "y": [ue_y_min_cm, ue_y_max_cm],
            "z": [float(unreal_z_cm.min()), float(unreal_z_cm.max())],
        },
        "conversion": "UE_X=USD_X, UE_Y=-USD_Y, UE_Z=USD_Z",
        "height_encoding": "round(z_cm * 128 + 32768)",
        "maximum_height_quantization_error_cm": max_error_cm,
    }
    metadata_path = output_dir / "Landscape_1_ue_metadata.json"
    write_atomic(
        metadata_path,
        (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    print(f"USD source     : {usd_path}")
    print(f"Resolution     : {width} x {height}")
    print(
        "UE bounds (cm): "
        f"X[{x_min_cm:.3f}, {x_max_cm:.3f}] "
        f"Y[{ue_y_min_cm:.3f}, {ue_y_max_cm:.3f}] "
        f"Z[{unreal_z_cm.min():.3f}, {unreal_z_cm.max():.3f}]"
    )
    print(f"Max Z error   : {max_error_cm:.9f} cm")
    print(f"Wrote R16     : {r16_path} ({r16_path.stat().st_size} bytes)")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
