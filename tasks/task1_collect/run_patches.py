"""串行跑 TASK1 的三个地形裁片。

为什么必须串行：UE 只允许一个实例占着工程（ue/launch.py 会直接报错退出），
bridge / 状态 / 键盘 / 相机四个端口也都是固定常量，同时跑两个会互相抢。
所以这里一个跑完再起下一个，每个裁片一个独立进程——同进程里连续调
run.main() 会残留 socket 与子进程状态。

每个裁片用自己那份 task1_<patch>.yaml（只比 task1.yaml 多一个 ue_map）
和自己的输出目录，互不覆盖。task1.sh 与 run.py 的默认行为不受影响。

    python MoonSim/tasks/task1_collect/run_patches.py
    python MoonSim/tasks/task1_collect/run_patches.py --patch southwest --wall-seconds 300
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                  # MoonSim
WORKSPACE = ROOT.parent                 # MoonUnrealEnv
PATCHES = ('center', 'southwest', 'northeast')
PYTHON = os.environ.get('LUNARBENCH_PYTHON',
                        '/home/lry/miniconda3/envs/moonunreal-mujoco/bin/python')

# 在子进程里跑，而不是同进程循环 import——见文件头的说明。
RUNNER = """\
import sys
sys.path.insert(0, {workspace!r})
from MoonSim.tasks.task1_collect import run
run.main(config={config!r}, output={output!r}, policy={policy!r}, wall_seconds={wall!r})
"""


def run_one(args, patch):
    config = HERE / f'task1_{patch}.yaml'
    output = HERE / 'generated' / f'terrain_{patch}'
    if not config.is_file():
        print(f'{patch}: 缺少实验条件 {config}', file=sys.stderr)
        return False
    print(f'\n=== TASK1 {patch} -> {output.relative_to(ROOT)} ===', flush=True)
    env = {**os.environ, 'PYTHONPATH': '', 'OPENBLAS_NUM_THREADS': '1'}
    env.pop('LUNARBENCH_UE_MAP', None)      # 让地图只由 task1_<patch>.yaml 的 ue_map 决定
    started = time.monotonic()
    code = subprocess.run(
        [PYTHON, '-c', RUNNER.format(workspace=str(WORKSPACE), config=str(config),
                                     output=str(output), policy=args.policy,
                                     wall=args.wall_seconds)],
        cwd=WORKSPACE, env=env).returncode
    print(f'{patch}: exit={code} elapsed={time.monotonic() - started:.0f}s', flush=True)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--patch', action='append', choices=PATCHES,
                        help='只跑指定裁片；缺省三个都跑')
    parser.add_argument('--wall-seconds', type=float, default=0.,
                        help='每个裁片的墙钟上限，0 表示不限（与 episode.time_limit_s 无关）')
    parser.add_argument('--policy', default=None,
                        help='覆盖 policy.name；填 none 可跳过 baseline，只起场景')
    args = parser.parse_args()

    if not Path(PYTHON).exists():
        parser.error(f'找不到 Python：{PYTHON}（用 LUNARBENCH_PYTHON 覆盖）')

    patches = args.patch or list(PATCHES)
    for index, patch in enumerate(patches):
        if not run_one(args, patch):
            print(f'\n{patch} 失败，已中止；先前裁片的结果保持不动', file=sys.stderr)
            return 1
        if index + 1 < len(patches):
            # 上个裁片的 UE 刚退出，端口还在 TIME_WAIT；留一点余量再起下一个。
            print('等待端口释放…', flush=True)
            time.sleep(10)
    return 0


if __name__ == '__main__':
    sys.exit(main())
