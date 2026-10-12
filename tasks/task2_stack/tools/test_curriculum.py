"""离线验证奖励不会把接触/换石/重复夹取当成持续抓取，不在 run 中调用。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stone_stack.task_config import load_config
from stone_stack.rl.reward import CurriculumReward


def signals():
    return dict(grasped=np.zeros(10, bool), resting=np.zeros(10, bool), relative=np.zeros((10, 3)),
                height=np.zeros(10), placed=np.zeros(10, bool), reach=np.zeros(10),
                center=np.zeros(10), touch=np.zeros(10), closeness=np.ones(10), place=np.zeros(10))


class CurriculumTests(unittest.TestCase):
    def tracker(self, stage='grasp'):
        cfg=load_config()['policy']['rl']['curriculum'].copy()
        cfg['stage']=stage
        return CurriculumReward(cfg,10)

    def test_contact_on_ground_is_not_grasp(self):
        tracker=self.tracker();s=signals();s['grasped'][0]=True;s['height'][0]=.1;s['resting'][0]=True
        for _ in range(10):
            _,bonus,success,_=tracker.update(s,.12)
            self.assertFalse(success);self.assertEqual(bonus,0)

    def test_same_stone_hold_and_no_repeat_bonus(self):
        tracker=self.tracker('lift');s=signals();s['grasped'][0]=True;s['height'][0]=.03
        for _ in range(2):self.assertEqual(tracker.update(s,.12)[1],0)
        self.assertEqual(tracker.update(s,.12)[1],10)
        self.assertEqual(tracker.update(s,.12)[1],0)
        s['grasped'][0]=False;tracker.update(s,.12)
        s['grasped'][0]=True
        for _ in range(3):self.assertEqual(tracker.update(s,.12)[1],0)
        s['height'][0]=.09
        for _ in range(4):self.assertFalse(tracker.update(s,.12)[2])
        _,bonus,success,_=tracker.update(s,.12)
        self.assertTrue(success);self.assertEqual(bonus,15)

    def test_switching_stones_or_slipping_resets_hold(self):
        tracker=self.tracker();s=signals();s['height'][:]=.03
        for i in range(8):
            s['grasped'][:]=False;s['grasped'][i%2]=True
            self.assertFalse(tracker.update(s,.12)[2])
        s['grasped'][:]=False;s['grasped'][0]=True
        for i in range(8):
            s['relative'][0,0]=.03*i
            self.assertFalse(tracker.update(s,.12)[2])

    def test_placement_requires_grasp_and_release_stability(self):
        tracker=self.tracker('place');s=signals();s['placed'][0]=True
        for _ in range(8):self.assertFalse(tracker.update(s,.12)[2])
        s['placed'][0]=False;s['grasped'][0]=True;s['height'][0]=.03
        for _ in range(3):tracker.update(s,.12)
        s['grasped'][0]=False;s['placed'][0]=True
        for _ in range(4):self.assertFalse(tracker.update(s,.12)[2])
        _,bonus,success,events=tracker.update(s,.12)
        self.assertTrue(success);self.assertEqual(bonus,25);self.assertEqual(events['place'],[0])


if __name__=='__main__':unittest.main()
