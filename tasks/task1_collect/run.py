"""TASK1 launch: standalone physics or UE/ MuJoCo together, using one task origin."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import socket
import sys
import time
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT.parent),str(ROOT.parent/'.local/python'),str(HERE),str(ROOT/'mujoco'),str(ROOT/'ue')]
import config as config_module
import scene
from task import Task1
from camera import CameraSync
from viewer import Viewer
from display import export

# Daily use: edit these ordinary Python variables, then run task1.sh. No CLI parser.
ENGINE = 'ue'             # ue: UE window + MuJoCo physics; both: diagnostic dual windows
HEADLESS = False
STATE_HZ = 45.0
VIEWER_HZ = 20.0          # Secondary diagnostic renderer must not consume every physics cycle.


def main(*, config=HERE/'task1.yaml', robot=None, policy=None, engine=ENGINE,
         ue_map=None, headless=HEADLESS, wall_seconds=0., seconds=0., report=None,
         output=scene.GENERATED, rebuild=False):
    args=SimpleNamespace(config=Path(config), robot=robot, policy=policy, engine=engine,
                         ue_map=ue_map, headless=headless, wall_seconds=wall_seconds,
                         seconds=seconds, report=Path(report) if report else None,
                         output=Path(output), rebuild=rebuild)
    if engine not in ('ue','mujoco','both'):raise ValueError('engine must be ue, mujoco or both')
    if headless and engine!='mujoco':raise ValueError('headless requires engine=mujoco')
    cfg=config_module.load(args.config)
    if args.robot:cfg['robot']['name']=args.robot
    if args.policy:cfg['policy']['name']=args.policy
    # config 里的 ue_map 只是给 task1_<patch>.yaml 省一步命令行；显式传参优先。
    if not args.ue_map:args.ue_map=cfg.get('ue_map')
    meta,field=scene.ensure(cfg,args.output,force=args.rebuild)
    task=Task1(cfg,field,args.output/'scene.xml',args.output/'scene.json')
    (args.output/'environment.json').write_text(json.dumps({**task.info(),'config':cfg},indent=2))
    baseline=cfg['policy']['name']=='z_mobile_manip'
    if baseline and (headless or engine=='mujoco'):raise ValueError('Z-Mobile baseline requires the UE RGB-D producer')
    children=[];sensor_source=None;policy=None
    joint=args.engine in ('ue','both');process=None;viewer=None;camera=None;rx=None;tx=None
    started=time.monotonic();next_step=started;last_frame=0.;last_metric=started;last_sim=0.;seq=0;heartbeat=started
    timings={k:[] for k in ('physics_ms','packet_ms','viewer_ms','policy_ms')}
    input_count=0;heartbeat_stops=0;last_viewer=0.
    episode_reported=False
    reported=-1;metrics={};state_port=int(os.environ.get('LUNARBENCH_STATE_PORT','19400'));last_policy=task.steps
    print(f'TASK1 robot={task.robot.name} grid={field.samples}x{field.samples} spacing={field.spacing}m rocks={len(task.rock_adr)}',flush=True)
    try:
        if baseline:
            from MoonSim.bridge.mujoco_source import MuJoCoSource
            from MoonSim.bridge.client import Client
            env={**os.environ,'PYTHONPATH':str(ROOT.parent/'.local/python'),'OPENBLAS_NUM_THREADS':'1'}
            bridge_log=(args.output/'bridge_server.log').open('w')
            children.append(subprocess.Popen([sys.executable,'-m','MoonSim.bridge.server'],cwd=ROOT.parent,env=env,stdout=bridge_log,stderr=bridge_log))
            bridge_log.close();client=Client(timeout=.3);deadline=time.monotonic()+20
            while True:
                if children[0].poll() is not None:raise RuntimeError('Bridge exited; see bridge_server.log')
                try:client.request('GET','/status');break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Bridge startup timeout')
                    time.sleep(.1)
            client.close()
            log=(args.output/'baseline.log').open('w')
            policy_env={**env,'PYTHONPATH':'','YOLO_CONFIG_DIR':str(ROOT.parent/'.local/ultralytics'),'MPLCONFIGDIR':'/tmp/task1-mpl','HF_HOME':str(ROOT.parent/'.local/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            children.append(subprocess.Popen([str(ROOT.parent/'.local/zmobile-env/bin/python'),'-m','MoonSim.baselines.z_mobile_manip.run'],cwd=ROOT.parent,env=policy_env,stdout=log,stderr=log));log.close()
            sensor_source=MuJoCoSource(task);task.robot.manual=False
            print('BASELINE z_mobile_manip | sensor viewer http://127.0.0.1:19530',flush=True)
        if joint:
            export(task,args.output)
            from launch import launch
            (args.output/'ue.log').write_text('')
            ue_env={'MOONSIM_TASK_DIR':str(args.output.resolve()),'MOONSIM_SENSORS':'1' if baseline else '0'}
            if args.ue_map:ue_env['LUNARBENCH_UE_MAP']=args.ue_map
            process=launch(engine='ue',extra_env=ue_env,
                           extra_args=[f'-Abslog={args.output.resolve()/"ue.log"}','-NoSound'])
        if not args.headless and args.engine!='ue':
            viewer=Viewer(task.model,task.data,lambda key:task.key(chr(key)) if key<128 else None,title=f'MuJoCo : {task.robot.name}')
        if not args.headless:
            camera=CameraSync(task,viewer,receive=joint)
            camera.camera.lookat[:]=task.base_position;camera.camera.distance=5.5
            camera.camera.azimuth=-55;camera.camera.elevation=-30;camera.attach(viewer)
        if joint:
            tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            rx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);rx.bind(('127.0.0.1',19401));rx.setblocking(False)
        # UE loading is not counted as simulation time; start when its receiver is ready.
        if joint:
            # UE 的导入耗时随石头数量增长：60 块时约 34 s，150 块时超过 140 s 仍未就绪。
            # 这只是失败上限，不是等待时间——正常启动会提前 break，放宽不影响成功路径。
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                log=args.output/'ue.log'
                if log.exists():
                    content=log.read_text(errors='replace')
                    if 'GO2_BRIDGE_READY' in content:break
                    if 'LogPython: Error: Traceback' in content:raise RuntimeError('UE task initialization failed; see '+str(log))
                if process.poll() is not None:raise RuntimeError('UE exited during task initialization')
                if viewer:viewer.sync()
                time.sleep(.05)
            else:raise TimeoutError('UE task receiver did not become ready')
        started=time.monotonic();next_step=started;last_metric=started;heartbeat=started
        while not(args.seconds and task.data.time>=args.seconds):
            done=task.terminated or task.truncated
            if done and not episode_reported:
                result={**task.info(),'score':task.score,'sim_seconds':float(task.data.time),
                        'policy':cfg['policy']['name']}
                (args.output/'episode_result.json').write_text(json.dumps(result,indent=2))
                print('EPISODE_END '+json.dumps(result)+'; R resets the same scene',flush=True)
                episode_reported=True
            if args.headless and done:break
            if not done:episode_reported=False
            if args.wall_seconds and time.monotonic()-started>=args.wall_seconds:break
            if viewer and not viewer.is_running():break
            if process and process.poll() is not None:break
            if any(p.poll() is not None for p in children):raise RuntimeError('Bridge or baseline exited; see their logs')
            if rx:
                while True:
                    try:payload,_=rx.recvfrom(1024)
                    except BlockingIOError:break
                    key=payload.decode('ascii',errors='ignore')[:1]
                    if key=='_':heartbeat=time.monotonic()
                    elif key:
                        if key=='b' and baseline:task.robot.manual=False
                        else:task.key(key)
                        if key=='r' and baseline:task.robot.manual=False
                        input_count+=1
                if time.monotonic()-heartbeat>.75 and task.motion.active and task.robot.manual:
                    task.key(' ');heartbeat_stops+=1
            if task.steps<last_policy:last_policy=-task.control_steps
            if sensor_source:
                sensor_source.update()
                if not(task.terminated or task.truncated or task.paused or task.robot.manual) and task.steps-last_policy>=task.control_steps:
                    last_policy=task.steps
                    action=sensor_source.command()
                    if action is None:
                        action=task.robot.last_action.copy();action[:3]=0.
                    task.robot.apply_action(action)
            before=time.monotonic();task.physics_step();now=time.monotonic()
            timings['physics_ms'].append((now-before)*1000)
            if not args.headless and now-last_frame>=1/STATE_HZ:
                before=time.monotonic()
                if joint:
                    packet=task.packet();seq+=1
                    packet.update(seq=seq,camera=camera.update(task),camera_ack=camera.ack)
                    payload=json.dumps(packet,separators=(',',':'),allow_nan=False).encode()
                    if len(payload)>60000:raise ValueError('Task display packet exceeds local UDP limit; reduce displayed parts or extend transport')
                    tx.sendto(payload,('127.0.0.1',state_port))
                elif camera:camera.update(task)
                timings['packet_ms'].append((time.monotonic()-before)*1000)
                last_frame=now
            if viewer and now-last_viewer>=1/VIEWER_HZ:
                before=time.monotonic();viewer.sync();last_viewer=time.monotonic()
                timings['viewer_ms'].append((last_viewer-before)*1000)
            if task.data.time<last_sim:last_sim=task.data.time;last_metric=now
            if now-last_metric>=1:
                metrics={'wall':now,'sim_time_s':float(task.data.time),
                         'real_time_factor':(task.data.time-last_sim)/(now-last_metric),
                         'visible_terrain_tiles':viewer.visible_terrain_tiles if viewer else 0,
                         'packets_sent':seq,'terrain_samples':field.samples**2,'rocks':len(task.rock_adr),
                         'input_count':input_count,'heartbeat_stops':heartbeat_stops,
                         'contacts':task.data.ncon,'sleeping_trees':int(np.count_nonzero(task.data.tree_asleep>=0)),
                         'timings':{k:{'mean':float(np.mean(v)), 'p95':float(np.percentile(v,95)),
                                      'max':float(np.max(v)), 'calls':len(v)} for k,v in timings.items() if v}}
                (args.output/'runtime_metrics.json').write_text(json.dumps(metrics))
                with (args.output/'runtime_metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                for v in timings.values():v.clear()
                last_metric=now;last_sim=task.data.time
            if reported!=len(task.collected):
                reported=len(task.collected)
                print(f'COLLECT {reported}/{len(task.rock_adr)} score={task.score:g}',flush=True)
            if not args.headless:
                next_step+=task.model.opt.timestep;now=time.monotonic()
                if task.paused or task.terminated or task.truncated:
                    next_step=now+.01
                elif now-next_step>.25:next_step=now
                if next_step>now:time.sleep(next_step-now)
            elif task.paused:break
    finally:
        if sensor_source:sensor_source.close()
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if viewer:viewer.close()
        if camera:camera.close()
        if tx:tx.close()
        if rx:rx.close()
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except __import__('subprocess').TimeoutExpired:process.kill();process.wait(timeout=10)
    report={**task.info(),'sim_seconds':float(task.data.time),'wall_seconds':time.monotonic()-started,
            'physics_steps':task.steps,'score':task.score,'robot_position_m':task.base_position.tolist(),
            'finite':bool(np.isfinite(task.data.qpos).all()),'terminated':task.terminated,'truncated':task.truncated,
            'terrain_samples':field.samples**2,'terrain_spacing_m':field.spacing,'runtime':metrics,
            'policy':policy.describe() if policy else None}
    task.close();print(json.dumps(report,indent=2),flush=True)
    if args.report:args.report.write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    if len(sys.argv)>1:raise SystemExit('无需命令行参数。修改 run.py 顶部运行变量、task1.yaml 实验条件，或 Python 调用 main(...)。')
    main()
