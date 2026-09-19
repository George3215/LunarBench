"""Existing localhost Go2 demo: latest state only, synchronized editor camera."""
import unreal,socket,json,time,os,math,sys
from pathlib import Path

if hasattr(unreal,'_moon_go2_bridge'):
    old=unreal._moon_go2_bridge
    if 'handle' in old:unreal.unregister_slate_post_tick_callback(old['handle'])
    old['socket'].close()
    if old.get('sensor'):old['sensor'].close()
    if 'camera_socket' in old:old['camera_socket'].close()
root=Path(__file__).resolve().parents[1]/'mujoco'
sub=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
task_dir=os.environ.get('MOONSIM_TASK_DIR')
output=Path(task_dir) if task_dir else root/'generated'
visual=json.loads((output/'visual.json').read_text());expected=len(visual['geoms'])
if task_dir:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import task_scene
    actors=task_scene.install(unreal,sub,output,visual)
else:
    actors={a.get_name():a for a in sub.get_all_level_actors() if a.actor_has_tag('Go2Visual')}
assert len(actors)==expected, f'Expected {expected} parts, found {len(actors)}'
origin=visual['origin_ue_m']
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind(('127.0.0.1',int(os.environ.get('LUNARBENCH_STATE_PORT','19400'))));sock.setblocking(False)
camera_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
state={'socket':sock,'camera_socket':camera_socket,'frames':0,'last_hud':None,'camera_set':False,
       'camera_seq':0,'last_applied':None,'metric_time':0.,'last_seq':0, 'poses':{},
       'gaps':[], 'callbacks':[], 'ages':[]}
if task_dir and os.environ.get('MOONSIM_SENSORS')=='1':
    from sensors import CameraSource
    state['sensor']=CameraSource(unreal,output)
unreal._moon_go2_bridge=state
editor=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
sub.set_selected_level_actors([])
unreal.SystemLibrary.execute_console_command(editor.get_editor_world(),'t.MaxFPS 45')
library=unreal.MoonViewportLibrary


def to_ue(frame):
    return [*(100*(frame[i]*([1,-1,1][i])+origin[i]) for i in range(3)),
            *(frame[i]*(-1 if i in (4,7) else 1) for i in range(3,9)),*frame[9:]]


def to_mj(frame):
    return [*((frame[i]/100-origin[i])*([1,-1,1][i]) for i in range(3)),
            *(frame[i]*(-1 if i in (4,7) else 1) for i in range(3,9)),*frame[9:]]


def tick(dt):
    started=time.monotonic();gap=started-state.get('last_tick',started);state['last_tick']=started
    state['gaps'].append(gap*1000)
    # Detect edits in UE before applying incoming canonical camera. Acknowledgements
    # prevent an old packet from snapping the editor back during RMB navigation.
    now_camera=list(library.read_camera())
    previous=state['last_applied']
    if previous and len(now_camera)==11 and any(abs(a-b)>1e-4 for a,b in zip(now_camera[:10],previous[:10])):
        state['camera_seq']+=1
        packet={'seq':state['camera_seq'],'camera':to_mj(now_camera)}
        camera_socket.sendto(json.dumps(packet).encode(),('127.0.0.1',19402))
        state['last_applied']=now_camera
    payload=None;drained=0
    # Drain stale datagrams without decoding them; visualization never replays backlog.
    for _ in range(256):
        try:payload,_=sock.recvfrom(60000);drained+=1
        except BlockingIOError:break
    latest=None
    if payload:
        try:
            p=json.loads(payload)
            if isinstance(p,dict) and p.get('v')==1 and len(p.get('geoms',[]))==expected and p.get('task')==visual.get('task'):latest=p
        except (ValueError,TypeError):pass
    if latest is None:
        if not state['camera_set'] and os.environ.get('LUNARBENCH_EXTERNAL_PHYSICS')=='1':
            latest=visual['initial']
        else:return
    for ident,x,y,z,qx,qy,qz,qw in latest['geoms']:
        actor=actors.get(f'Go2Geom_{ident:03d}')
        pose=(x,y,z,qx,qy,qz,qw)
        if actor and state['poses'].get(ident)!=pose:
            actor.set_actor_location_and_rotation(unreal.Vector(x,y,z),unreal.Quat(qx,qy,qz,qw).rotator(),False,False)
            state['poses'][ident]=pose
    if state.get('sensor'):state['sensor'].capture(latest)
    state['frames']+=1
    frame=latest.get('camera')
    if frame and latest.get('camera_ack',0)>=state['camera_seq']:
        library.apply_camera(to_ue(frame));state['last_applied']=list(library.read_camera());state['camera_set']=True
    elif not state['camera_set']:
        # UE-only mode has no simulation producer; use the saved initial pose.
        frame=[-2.6,3.4,1.9,2.6,-3.4,-1.6,0,0,1,45,16/9]
        norm=math.sqrt(sum(v*v for v in frame[3:6]));frame[3:6]=[v/norm for v in frame[3:6]]
        library.apply_camera(to_ue(frame));state['last_applied']=list(library.read_camera());state['camera_set']=True
    if task_dir and state.get('last_collected')!=latest.get('collected_count'):
        state['last_collected']=latest.get('collected_count')
        unreal.log(f'TASK1_SCORE {latest.get("collected_count",0)}/{latest.get("rocks_total",0)} score={latest.get("score",0)}')
    hud=(latest.get('gear',2),latest.get('moving',False),latest.get('paused',False))
    if hud!=state['last_hud']:
        state['last_hud']=hud
        unreal.log('GO2_STATUS gear=%d moving=%s paused=%s remaining=%.3fm' % (*hud,latest.get('remaining_m',0)))
        (output/'viewport_last_state.json').write_text(json.dumps(latest,indent=2))
    if state['frames']%30==0:(output/'viewport_last_state.json').write_text(json.dumps(latest,indent=2))
    state['callbacks'].append((time.monotonic()-started)*1000)
    state['ages'].append((time.monotonic_ns()-latest.get('sent_ns',time.monotonic_ns()))/1e6)
    if time.monotonic()-state['metric_time']>1:
        state['metric_time']=time.monotonic()
        metrics={'wall':time.monotonic(),'frame_gap_ms':gap*1000,'callback_ms':(time.monotonic()-started)*1000,
                 'state_age_ms':(time.monotonic_ns()-latest.get('sent_ns',time.monotonic_ns()))/1e6,
                 'sim_time':latest['time'],'frames':state['frames'],'drained':drained,
                 'camera_edits_sent':state['camera_seq'],'camera_ack':latest.get('camera_ack',0)}
        for key,values in (('frame_gap',state['gaps']),('callback',state['callbacks']),('state_age',state['ages'])):
            ordered=sorted(values)
            if ordered:
                metrics[key+'_p95_ms']=ordered[min(len(ordered)-1,int(.95*len(ordered)))]
                metrics[key+'_max_ms']=ordered[-1]
            values.clear()
        if latest.get('camera') and state['last_applied'] and latest.get('camera_ack',0)>=state['camera_seq']:
            # Independent UE projection through the actual constrained scene view.
            f=latest['camera']
            right=[f[4]*f[8]-f[5]*f[7],f[5]*f[6]-f[3]*f[8],f[3]*f[7]-f[4]*f[6]]
            # Three reference points in front of the camera remain valid even
            # when the user intentionally looks away from the robot.
            points=[[f[i]+5*f[i+3]+u*right[i]+v*f[i+6] for i in range(3)]
                    for u,v in ((0,0),(1,.5),(-.7,-.6))]
            projections=library.project_points([unreal.Vector(*to_ue([*p,*f[3:]])[:3]) for p in points])
            errors=[];scale=math.tan(math.radians(f[9]/2))
            for point,projected in zip(points,projections):
                delta=[point[i]-f[i] for i in range(3)]
                depth=sum(delta[i]*f[i+3] for i in range(3))
                x=.5+sum(delta[i]*right[i] for i in range(3))/(2*depth*scale*f[10])
                y=.5-sum(delta[i]*f[i+6] for i in range(3))/(2*depth*scale)
                errors.append(max(abs(x-projected.x)*1280,abs(y-projected.y)*720))
            if len(errors)==3:metrics['camera_projection_error_at_1280x720_px']=max(errors)
            observed=to_mj(state['last_applied'])
            metrics['camera_position_error_m']=math.dist(observed[:3],f[:3])
            # Audit actual robot geometry rather than self-consistent camera-only points.
            pose=latest['geoms'][0];actual=actors[f'Go2Geom_{pose[0]:03d}'].get_actor_location()
            robot_m=[(getattr(actual,axis)/100-origin[i])*([1,-1,1][i]) for i,axis in enumerate(('x','y','z'))]
            expected_m=[(pose[i+1]/100-origin[i])*([1,-1,1][i]) for i in range(3)]
            metrics['camera_robot_offset_error_m']=math.dist([observed[i]-robot_m[i] for i in range(3)],[f[i]-expected_m[i] for i in range(3)])
            metrics['origin_ue_m']=origin
            metrics['display_parts']=len(actors)
            if task_dir:
                metrics['display_rocks']=sum(g.get('role')=='rock' for g in visual['geoms'])
                metrics['actor_position_error_cm']=max(max(abs(getattr(actors[f'Go2Geom_{int(p[0]):03d}'].get_actor_location(),axis)-p[i+1]) for i,axis in enumerate(('x','y','z'))) for p in latest['geoms'])
            metrics['camera_basis_error']=max(abs(a-b) for a,b in zip(observed[3:9],f[3:9]))
            metrics['camera_vfov_error_deg']=abs(observed[9]-f[9])
        (output/'latency_current.json').write_text(json.dumps(metrics))
        with (output/'latency_current.jsonl').open('a') as out:out.write(json.dumps(metrics)+'\n')
    if state['frames']==100:
        worst=max(max(abs(getattr(actors[f'Go2Geom_{ident:03d}'].get_actor_location(),a)-v) for a,v in zip(('x','y','z'),(x,y,z))) for ident,x,y,z,*_ in latest['geoms'])
        result={'frames':state['frames'],'parts':len(actors),'max_position_error_cm':worst,'sim_time':latest['time'],'source':'actual UE actor readback'}
        (output/'ue_bridge_validation.json').write_text(json.dumps(result,indent=2));unreal.log('GO2_BRIDGE_PASS '+json.dumps(result))

state['handle']=unreal.register_slate_post_tick_callback(tick)
unreal.log('GO2_BRIDGE_READY localhost UDP; original geometry; camera pose and projection synchronized')
