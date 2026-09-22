"""腕部、俯视、前景相机的 RGB-D 采集和标定信息。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

#: MuJoCo 的 GL 后端必须在 import mujoco 之前定下来
os.environ.setdefault("MUJOCO_GL", "egl")

from .policy.base import CameraFrame  # noqa: E402

DEFAULT_CAMERAS: tuple[str, ...] = ("wrist", "top", "front")


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fovy_deg: float

    @property
    def focal_px(self) -> float:
        """由垂直视场角与画面高度推出的焦距（像素）。"""
        return 0.5 * self.height / np.tan(np.deg2rad(self.fovy_deg) / 2.0)

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fovy_deg": round(self.fovy_deg, 3),
            "focal_px": round(float(self.focal_px), 2),
        }


class CameraRig:
    """一个模型上的多相机 RGB-D 采集器。"""

    def __init__(self, model, data, width: int, height: int, names: tuple[str, ...] = DEFAULT_CAMERAS):
        import mujoco

        self.model = model
        self.data = data
        self.width = int(width)
        self.height = int(height)
        self.names = tuple(names)
        self.renderer = mujoco.Renderer(model, self.height, self.width)
        self._depth_mode = False

    # ------------------------------------------------------------------ 元信息

    def intrinsics(self, name: str) -> CameraIntrinsics:
        camera_id = self.model.camera(name).id
        return CameraIntrinsics(self.width, self.height, float(self.model.cam_fovy[camera_id]))

    def extrinsics(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        camera_id = self.model.camera(name).id
        return (
            np.asarray(self.data.cam_xpos[camera_id], dtype=float).copy(),
            np.asarray(self.data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy(),
        )

    def camera_names(self) -> tuple[str, ...]:
        return self.names

    # ------------------------------------------------------------------ 采集

    def _set_depth(self, enabled: bool) -> None:
        if enabled == self._depth_mode:
            return
        if enabled:
            self.renderer.enable_depth_rendering()
        else:
            self.renderer.disable_depth_rendering()
        self._depth_mode = enabled

    def render_rgb(self, name: str) -> np.ndarray:
        self._set_depth(False)
        self.renderer.update_scene(self.data, camera=name)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def render_depth(self, name: str) -> np.ndarray:
        self._set_depth(True)
        self.renderer.update_scene(self.data, camera=name)
        depth = np.asarray(self.renderer.render(), dtype=np.float32).copy()
        self._set_depth(False)
        return depth

    def capture(self, names: tuple[str, ...] | None = None, want_depth: bool = True) -> dict[str, CameraFrame]:
        """一次性抓多路相机。返回的每一帧都带位姿与内参。"""
        selected = self.names if names is None else tuple(names)
        frames: dict[str, CameraFrame] = {}
        for name in selected:
            rgb = self.render_rgb(name)
            depth = self.render_depth(name) if want_depth else None
            pos, rot = self.extrinsics(name)
            frames[name] = CameraFrame(
                name=name,
                rgb=rgb,
                depth=depth,
                fovy_deg=float(self.model.cam_fovy[self.model.camera(name).id]),
                width=self.width,
                height=self.height,
                pos=pos,
                rot=rot,
            )
        return frames

    def close(self) -> None:
        """显式释放：让 GL 上下文在解释器退出前干净关闭（否则 PyOpenGL 会在
        `__del__` 里打一堆 EGLError 噪音）。"""
        try:
            self.renderer.close()
        except Exception:  # noqa: BLE001 - 关闭失败不该影响结果
            pass


def open_camera_rig(built_scene, names: tuple[str, ...] = DEFAULT_CAMERAS) -> CameraRig:
    return CameraRig(
        built_scene.model,
        built_scene.data,
        built_scene.options.camera_width,
        built_scene.options.camera_height,
        names=names,
    )
