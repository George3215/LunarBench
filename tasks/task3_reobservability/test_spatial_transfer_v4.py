"""Protocol boundaries that could otherwise create misleading navigation gains."""
import math
import unittest

import numpy as np

from spatial_transfer_v4 import event_frames, odometry, stop_allowed


class ProtocolTests(unittest.TestCase):
    def test_current_range_required_for_stop(self):
        for pixels, distance, expected in [(20, .999, True), (19, .5, False),
                (20, 1., False), (20, 0., False), (20, -1., False),
                (20, None, False), (20, math.nan, False), (20, math.inf, False)]:
            with self.subTest(pixels=pixels, distance=distance):
                self.assertEqual(stop_allowed({"red_pixel_count": pixels,
                    "red_region_median_surface_range_m": distance}), expected)

    def test_relative_odometry_respects_initial_heading_and_wrap(self):
        def r(yaw):
            return np.array([[math.cos(yaw), -math.sin(yaw), 0],
                             [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
        origin = {"xyz": np.array([5., 8., 2.]), "rotation": r(math.pi/2)}
        current = {"xyz": np.array([4., 10., 2.5]), "rotation": r(math.pi)}
        result = odometry(current, origin)
        self.assertAlmostEqual(result["x_m"], 2.)
        self.assertAlmostEqual(result["y_m"], 1.)
        self.assertAlmostEqual(result["z_m"], .5)
        self.assertAlmostEqual(result["yaw_rad"], math.pi/2, places=4)

    def test_memory_keeps_evidence_and_latest_loss_without_future_frames(self):
        frames = [{"frame_id": i, "depth_sensor": {"red_pixel_count": n}}
                  for i, n in enumerate([40, 45, 0, 0, 35, 0, 0, 0])]
        self.assertEqual([f["frame_id"] for f in event_frames(frames)], [0, 4, 5, 7])
        self.assertEqual([f["frame_id"] for f in event_frames(frames[:4])], [0, 1, 2, 3])
        self.assertEqual(len(event_frames(frames[:1])), 1)


if __name__ == "__main__":
    unittest.main()
