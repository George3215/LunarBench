"""直接控制契约测试；真实 Qwen/MuJoCo smoke 命令见 README。"""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stone_stack.direct_control import StoneTools
from stone_stack.policy.direct import DirectPolicy
from stone_stack.task_config import load_config


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()["policy"]["direct"] | dict(transport="tool_calls")
        self.executor = Mock()
        self.executor.synthetic_gripper = None
        self.executor.pad_thickness_m = 0.01
        self.executor.measured_opening.return_value = 0.05
        self.executor.controller.tcp_pose.return_value = (np.zeros(3), np.eye(3))
        self.executor.profile.gripper.max_opening_m = 0.08
        self.executor.model.opt.timestep = 0.001
        self.tools = StoneTools(self.executor, self.config)

    def test_invalid_actions_never_execute(self):
        actions = [
            dict(position=[float("nan"), 0, 0]),
            dict(position=[0, 0]),
            dict(quaternion_wxyz=[2, 0, 0, 0]),
            dict(grip_width_m=0.09),
        ]
        for change in actions:
            args = dict(position=[0.01, 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                        grip_width_m=0.04, reason="test", frame="world", steps=1) | change
            with self.subTest(change=change):
                self.assertEqual(self.tools.dispatch("move_tcp", args)["status"], "validation_rejected")
        self.executor.controller.servo_to.assert_not_called()

    def test_oversized_and_invalid_contract_never_executes(self):
        from scipy.spatial.transform import Rotation
        base = dict(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                    grip_width_m=0.04, frame="world", steps=1, reason="check")
        changes = [dict(position=[0.03, 0.041, 0]),
                   dict(quaternion_wxyz=np.roll(Rotation.from_rotvec([0, 0, .351]).as_quat(), 1).tolist()),
                   dict(frame="base"), dict(steps=0), dict(steps=6), dict(steps=1.5),
                   dict(left={}), dict(right={})]
        for change in changes:
            result = self.tools.dispatch("move_tcp", base | change)
            self.assertEqual(result["status"], "validation_rejected")
        self.executor.controller.servo_to.assert_not_called()

    def test_near_target_is_unchanged(self):
        from stone_stack.robots.control import ServoResult
        self.executor.controller.servo_to.return_value = ServoResult(300, 0, 0, True)
        result = self.tools.move_tcp([0.01, 0, 0], [1, 0, 0, 0], 0.04, "near target", "world", 1)
        np.testing.assert_allclose(self.executor.controller.servo_to.call_args.args[0], [0.01, 0, 0])
        self.assertTrue(result["executed"])

    def test_delta_action_uses_the_same_guard_and_measured_orientation(self):
        from stone_stack.robots.control import ServoResult
        self.executor.controller.servo_to.return_value = ServoResult(300, 0, 0, True)
        args = dict(delta_position=[0, .02, 0], grip_width_m=.04, frame="world", steps=1, reason="move")
        result = self.tools.dispatch("move_tcp_delta", args)
        self.assertTrue(result["executed"])
        np.testing.assert_allclose(self.executor.controller.servo_to.call_args.args[0], [0, .02, 0])
        np.testing.assert_allclose(self.executor.controller.servo_to.call_args.args[1], np.eye(3))
        self.executor.controller.servo_to.reset_mock()
        result = self.tools.dispatch("move_tcp_delta", args | dict(delta_position=[.04, .04, 0]))
        self.assertEqual(result["status"], "validation_rejected")
        self.executor.controller.servo_to.assert_not_called()

    def test_gripper_only_action_is_not_skipped(self):
        from stone_stack.robots.control import ServoResult
        self.executor.controller.servo_to.return_value = ServoResult(300, 0, 0, True)
        self.tools.dispatch("move_tcp", dict(position=[0, 0, 0],
                            quaternion_wxyz=[1, 0, 0, 0], grip_width_m=0.04, reason="close", frame="world", steps=1))
        args = self.executor.controller.servo_to.call_args.kwargs
        self.assertGreater(args["hold_chunks"], args["seconds"] / self.executor.model.opt.timestep)
        self.executor._ctrl_for_width.assert_called_once_with(0.04)

    def test_structured_transport_uses_the_same_tools(self):
        policy = DirectPolicy(self.config | dict(transport="json_schema", cameras=[]))
        policy.reset("stack", {}, self.tools.specs)
        action = dict(name="finish", arguments=dict(reason="done"))
        reply = dict(choices=[dict(message=dict(content=json.dumps(action)))])
        with patch("stone_stack.policy.direct.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as send:
            decoded = policy.decide({}, {})
        request = json.loads(send.call_args.args[0].data)
        self.assertNotIn("tools", request)
        self.assertEqual(request["response_format"]["json_schema"]["schema"], policy.schema)
        self.assertEqual(decoded, action)
        policy.record_result(decoded, self.tools.dispatch(**decoded))
        self.assertEqual(policy.messages[-1]["role"], "user")

    def test_transport_error_is_not_retried(self):
        policy = DirectPolicy(self.config | dict(cameras=[]))
        policy.reset("stack", {}, self.tools.specs)
        with patch("stone_stack.policy.direct.urlopen", side_effect=TimeoutError("offline")) as send:
            with self.assertRaises(TimeoutError):
                policy.decide({}, {})
            self.assertEqual(send.call_count, 1)

    def test_failure_record_is_available_for_next_action(self):
        failure = self.tools.record_failure("stone_01", "lift", "step 3: stone stayed on table", "empty grasp", "lower TCP")
        self.assertEqual(self.tools.failures[-1], failure)
        self.assertEqual(failure["next_change"], "lower TCP")
        self.executor.controller.servo_to.assert_not_called()

    def test_trial_review_has_no_action_tools(self):
        policy = DirectPolicy(self.config)
        policy.reset("stack", {}, self.tools.specs)
        review = dict(summary="failed", failures=[], next_trial_plan="lower TCP")
        failure = dict(evidence="stone did not rise", next_change="lower TCP")
        with patch.object(policy, "complete", return_value=dict(content=json.dumps(review))) as request:
            self.assertEqual(policy.review(dict(success=False), [failure]), review)
        payload = request.call_args.args[0]
        self.assertNotIn("tools", payload)
        self.assertIn("stone did not rise", payload["messages"][-1]["content"])

    def test_finish_is_not_success(self):
        result = self.tools.dispatch("finish", dict(reason="looks complete"))
        self.assertFalse(self.tools.finished)
        self.assertFalse(result["rollout_finished"])
        self.executor.controller.servo_to.assert_not_called()

    def test_tool_calls_and_history(self):
        policy = DirectPolicy(self.config | dict(cameras=["top"], history_images=1))
        policy.reset("stack", {}, self.tools.specs)
        frame = Mock()
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (512, 384)).save(buffer, format="JPEG")
        frame.jpeg_bytes.return_value = buffer.getvalue()
        frame.width, frame.height, frame.fovy_deg = 640, 480, 58
        frame.pos, frame.rot = np.zeros(3), np.eye(3)
        call = dict(id="call_1", type="function", function=dict(
            name="finish", arguments=json.dumps(dict(reason="done"))))
        reply = dict(choices=[dict(message=dict(content="not an action", tool_calls=[call]))])
        requests = []

        def serve(request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(json.dumps(reply).encode())

        with patch("stone_stack.policy.direct.urlopen", side_effect=serve):
            for step in range(3):
                decoded = policy.decide(dict(step=step), dict(top=frame))
                self.assertEqual(decoded, dict(id="call_1", name="finish", arguments=dict(reason="done")))
                policy.record_result(decoded, dict(success=None))
        latest_state = json.loads([m for m in requests[-1]["messages"] if m["role"] == "user"][-1]["content"][0]["text"])
        camera = latest_state["cameras"]["top"]
        self.assertEqual((camera["width"], camera["height"]), (512, 384))
        self.assertEqual((camera["source_width"], camera["source_height"]), (640, 480))
        self.assertAlmostEqual(camera["K"][0][0], 384 / (2 * np.tan(np.deg2rad(58) / 2)))
        np.testing.assert_array_equal(camera["rotation_camera_to_world"], np.eye(3))
        for request in requests:
            self.assertEqual(request["tool_choice"], "required")
            self.assertFalse(request["parallel_tool_calls"])
        users = [m for m in requests[-1]["messages"] if m["role"] == "user"]
        self.assertEqual(sum(p["type"] == "image_url" for m in users for p in m["content"]), 1)
        self.assertEqual(len([m for m in requests[-1]["messages"] if m["role"] == "tool"]), 2)

    def test_plain_text_and_multiple_calls_are_errors(self):
        policy = DirectPolicy(self.config | dict(cameras=[]))
        for calls in ([], [dict(id="a"), dict(id="b")]):
            policy.reset("stack", {}, self.tools.specs)
            reply = dict(choices=[dict(message=dict(content='{"position":[0,0,0]}', tool_calls=calls))])
            with patch("stone_stack.policy.direct.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())):
                with self.assertRaises(ValueError):
                    policy.decide({}, {})

    def test_environment_owns_termination(self):
        self.executor.data.time = 0.0
        with patch("stone_stack.evaluation.evaluate_wall", return_value={"success": True}):
            self.tools.check_terminal()
            self.assertFalse(self.tools.finished)
            self.executor.data.time = 1.3
            self.tools.check_terminal()
            self.assertTrue(self.tools.success)
        with patch("stone_stack.evaluation.evaluate_wall", return_value={"success": False}):
            self.tools.step_id = self.config["max_steps"]
            self.tools.check_terminal()
            self.assertTrue(self.tools.finished)
            self.assertFalse(self.tools.success)

    def test_policy_cannot_raise_safeguards(self):
        for change in (dict(max_delta_m=.051), dict(max_rotation_rad=.351), dict(max_steps=0)):
            with self.assertRaises(ValueError):
                DirectPolicy(self.config | change)

    def test_missing_config_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/tmp/nonexistent-task2-direct-config.yaml")


if __name__ == "__main__":
    unittest.main()
