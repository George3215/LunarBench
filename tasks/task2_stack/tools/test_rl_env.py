"""离线验证 Gymnasium、固定场景、动作门限及 n-step 终止语义；不在入口自动运行。"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import sys
import unittest
import copy

import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stone_stack.task_config import load_config
from stone_stack.rl.env import StoneStackEnv
from stone_stack.rl.replay import Replay
from stone_stack.rl.reward import potential

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'configs/rl_stack10_validate.yaml')
        cls.env = StoneStackEnv(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_gymnasium_and_real_images(self):
        check_env(self.env, skip_render_check=True)
        obs, _ = self.env.reset(seed=17)
        self.assertEqual(obs.shape, (9, 84, 84))
        self.assertEqual(obs.dtype, np.uint8)
        for view in np.split(obs, 3):
            self.assertGreater(float(view.std()), 5.)

    def test_fixed_reset_and_bounded_motion(self):
        e = self.env
        obs, _ = e.reset(seed=17)
        start = e.data.qpos.copy()
        _, _, terminated, _, info = e.step(np.ones(7, dtype=np.float32))
        self.assertLessEqual(info['target_translation_m'], .05 + 1e-7)
        self.assertLessEqual(info['target_rotation_rad'], .35 + 1e-7)
        self.assertFalse(terminated)
        restored, _ = e.reset(seed=99)
        np.testing.assert_array_equal(start, e.data.qpos)
        np.testing.assert_array_equal(obs, restored)

    def test_time_limit_is_not_success(self):
        e = self.env
        e.reset()
        limit = e.cfg['episode_steps']
        e.step_id = limit - 1
        _, _, terminated, truncated, info = e.step(np.array([0, 0, 0, 0, 0, 0, 1], np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info['is_success'])
        with self.assertRaises(RuntimeError):
            e.step(np.zeros(7, np.float32))

    def test_invalid_action_does_not_advance(self):
        e = self.env
        e.reset()
        before = e.data.time
        with self.assertRaises(ValueError):
            e.step(np.full(7, 2, np.float32))
        self.assertEqual(e.data.time, before)

    def test_state_baseline(self):
        config = copy.deepcopy(self.config)
        config['policy']['rl']['observation'] = 'state'
        env = StoneStackEnv(config)
        try:
            check_env(env, skip_render_check=True)
            obs, _ = env.reset()
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(obs.dtype, np.float32)
        finally:
            env.close()

    def test_no_grasp_reward_from_open_command(self):
        _, info = self.env.reset()
        self.assertEqual(info['reward_terms']['grasp'], 0)
        self.assertEqual(info['reward_terms']['lift'], 0)
        self.assertEqual(info['reward_terms']['placed_count'], 0)
        phi = potential(info['reward_terms'], self.env.cfg['reward'])
        self.assertLessEqual(self.env.cfg['gamma'] * phi - phi, 0)


class ReplayTests(unittest.TestCase):
    def test_parallel_streams_never_mix(self):
        replay = Replay(20, 2, 3, .9, 17)
        for i in range(3):
            replay.add(np.array([i]), np.array([0]), 1, np.array([i + 1]), False, i == 2, env_id=0)
            replay.add(np.array([100 + i]), np.array([0]), 10, np.array([101 + i]), i == 2, False, env_id=1)
        self.assertEqual(len(replay), 6)
        first = {int(item[0][0]): item for item in replay.items}
        np.testing.assert_allclose(first[0][2], [2.71])
        np.testing.assert_allclose(first[0][3], [.9 ** 3])
        np.testing.assert_array_equal(first[0][4], [3])
        np.testing.assert_allclose(first[100][2], [27.1])
        np.testing.assert_allclose(first[100][3], [0])
        np.testing.assert_array_equal(first[100][4], [103])
        self.assertTrue(all(not queue for queue in replay.streams.values()))

    def test_terminal_and_truncated_bootstrap(self):
        for terminated, expected_discount in [(True, 0.), (False, .9 ** 2)]:
            replay = Replay(20, 2, 3, .9, 17)
            replay.add(np.array([0]), np.array([0]), 1, np.array([1]), False, False)
            replay.add(np.array([1]), np.array([1]), 2, np.array([2]), terminated, not terminated)
            self.assertEqual(len(replay), 2)
            np.testing.assert_allclose(replay.items[0][2], [2.8])
            np.testing.assert_allclose(replay.items[0][3], [expected_discount])
            self.assertEqual(len(replay.pending), 0)
            replay.add(np.array([100]), np.array([0]), 4, np.array([101]), True, False)
            np.testing.assert_array_equal(replay.items[-1][0], [100])


if __name__ == '__main__':
    unittest.main()
