"""真实 MuJoCo 场景一致性与 trial 边界测试，无需模型服务。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stone_stack.task_config import load_config
from stone_stack.upstream_scene import build_upstream
from stone_stack.execution import Executor
from stone_stack.policy.direct import DirectPolicy
from stone_stack.trials import evaluate_wall, run_trials, support_layers
from scripts import run_official_ur5e_robotiq_wall_stack as upstream

ROOT = Path(__file__).resolve().parents[1]


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'configs/upstream_qwen_trials.yaml')
        cls.built, cls.workcell, cls.profile = build_upstream(cls.config)

    def test_initial_scene_matches_upstream(self):
        c = self.config | {"scene": self.config["scene"] | {"supply_layout": "upstream"}}
        b, w, p = build_upstream(c)
        from stone_stack.rock_wall_stones import make_rock_wall_stones
        stones = make_rock_wall_stones(seed=17, count=10, style='paper', irregularity=1., subdivisions=5)
        initial = {s.name: upstream.initial_supply_pose(i, dict(quat=[1, 0, 0, 0]), s)
                   for i, s in enumerate(stones)}
        original = mujoco.MjModel.from_xml_string(upstream.build_wall_stack_scene(stones, initial))
        np.testing.assert_array_equal(p.home_qpos, upstream.Q_HOME_ELBOW_UP)
        np.testing.assert_array_equal(p.base_pos, upstream.ROBOT_BASE_POS)
        self.assertEqual(len(w.staging_poses), 10)
        self.assertEqual(w.courses, (4, 3, 2, 1))
        self.assertIsNone(b.synthetic_gripper)
        for name in ('qpos0', 'body_pos', 'body_quat', 'body_mass', 'geom_pos', 'geom_quat',
                     'geom_size', 'geom_contype', 'geom_conaffinity', 'geom_friction', 'actuator_gainprm'):
            np.testing.assert_array_equal(getattr(original, name), getattr(b.model, name), err_msg=name)
        self.assertEqual(original.opt.timestep, b.model.opt.timestep)
        self.assertEqual(b.model.ncam, original.ncam + 2)
        # 每块石头的 xyz / yaw 都来自同一个上游函数，未重新排布或贴地修正。
        for name, (pos, quat) in initial.items():
            address = b.model.jnt_qposadr[b.model.joint(name + '_free').id]
            np.testing.assert_allclose(b.model.qpos0[address:address+7], np.r_[pos, quat], atol=1e-5)

    def test_unbuilt_scene_is_not_success(self):
        e = Executor(self.built, self.profile, DirectPolicy(self.config['policy']['direct']))
        mujoco.mj_resetData(e.model, e.data)
        mujoco.mj_forward(e.model, e.data)
        self.assertFalse(evaluate_wall(e)['success'])

    def test_support_layer_assignment(self):
        supports = dict(a=set(), b=set(), c={'a', 'b'}, d={'c'}, loose=set())
        self.assertEqual(support_layers(list(supports), {'a', 'b'}, supports), dict(a=0, b=0, c=1, d=2))

    def test_boundary_reset_and_review_handoff(self):
        # 真实积分状态被上个 trial 改动后必须恢复；只替换模型/图片 I/O。
        b = self.built
        mujoco.mj_resetData(b.model, b.data)
        mujoco.mj_forward(b.model, b.data)
        e = Executor(b, self.profile, DirectPolicy(self.config['policy']['direct']))
        policy = DirectPolicy(self.config['policy']['direct'] | dict(trials=2))
        snapshots, received = [], []
        def execute(executor, agent, instruction, output, lessons):
            output.mkdir()
            snapshots.append(executor.data.qpos.copy())
            received.append(lessons)
            executor.data.qpos[0] += 0.2
            agent.messages = [dict(role='system', content='test')]
            return dict(success=False, evaluation={'success': False}, status='step_limit', tool_calls=1, executed_actions=1, failures=[], stone_plans={})
        review = dict(summary='missed', failures=[], next_trial_plan='rotate grasp axis')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('stone_stack.trials.run_direct', side_effect=execute), \
             patch('stone_stack.direct_control.StoneTools.observe', return_value=({}, {})), \
             patch.object(policy, 'review', return_value=review):
            report = run_trials(e, policy, 'stack', Path(tmp) / 'run')
            self.assertEqual(len(report['trials']), 2)
            np.testing.assert_array_equal(snapshots[0], snapshots[1])
            self.assertEqual(received[0], {})
            self.assertEqual(received[1], review)
            self.assertTrue((Path(tmp)/'run/trial_002/review.json').exists())


if __name__ == '__main__':
    unittest.main()
