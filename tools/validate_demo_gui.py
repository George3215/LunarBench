"""Bounded local GUI validation; only terminates the processes it starts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ue"))
from launch import launch
def workspace():
    return Path(__file__).resolve().parents[2]


def physics_processes():
    result = []
    for comm in Path('/proc').glob('[0-9]*/comm'):
        try:
            if comm.read_text().strip().startswith('python'):
                cmd = (comm.parent/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if str(workspace()/'MoonSim/mujoco/run.py') in cmd:
                    result.append(int(comm.parent.name))
        except OSError:
            pass
    return result


def focus_and_walk(pid, drag_button=None):
    # Exercise real X11 mouse/keyboard events against the actual editor viewport.
    # No simulation state is modified by this helper.
    import ctypes as c
    import re
    listing=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    (workspace()/'.local/validation/x11_windows.txt').write_text(listing)
    window=None
    candidates=[]
    for line in listing.splitlines():
        ident=re.search(r'0x[0-9a-f]+',line)
        size=re.search(r'(\d+)x(\d+)[+-]\d+[+-]\d+\s+([+-]\d+)([+-]\d+)',line)
        if not ident or not size:continue
        width,height,xpos,ypos=map(int,size.groups())
        if width<400 or height<300:continue
        owner=subprocess.run(['xprop','-id',ident.group(),'_NET_WM_PID'],capture_output=True,text=True).stdout
        if re.search(r'=\s*'+str(pid)+r'\b',owner):
            candidates.append((width*height,int(ident.group(),16),xpos+width//2,ypos+int(height*.88)))
    if candidates:
        _,window,click_x,click_y=max(candidates)
    if window is None:
        raise RuntimeError('Cannot find the editor window owned by the validation process')
    x=c.CDLL('libX11.so.6');xt=c.CDLL('libXtst.so.6')
    x.XOpenDisplay.restype=c.c_void_p
    display=x.XOpenDisplay(None)
    if not display:raise RuntimeError('No X11 display')
    class ClientMessage(c.Structure):
        _fields_=[('type',c.c_int),('serial',c.c_ulong),('send_event',c.c_int),('display',c.c_void_p),('window',c.c_ulong),('message_type',c.c_ulong),('format',c.c_int),('data',c.c_long*5)]
    class Event(c.Union):
        _fields_=[('client',ClientMessage),('padding',c.c_long*24)]
    x.XInternAtom.argtypes=[c.c_void_p,c.c_char_p,c.c_int];x.XInternAtom.restype=c.c_ulong
    x.XDefaultRootWindow.argtypes=[c.c_void_p];x.XDefaultRootWindow.restype=c.c_ulong
    x.XSendEvent.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_long,c.POINTER(Event)]
    x.XRaiseWindow.argtypes=[c.c_void_p,c.c_ulong]
    x.XSetInputFocus.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_ulong]
    x.XFlush.argtypes=[c.c_void_p]
    x.XCloseDisplay.argtypes=[c.c_void_p]
    x.XStringToKeysym.argtypes=[c.c_char_p];x.XStringToKeysym.restype=c.c_ulong
    x.XKeysymToKeycode.argtypes=[c.c_void_p,c.c_ulong];x.XKeysymToKeycode.restype=c.c_uint
    xt.XTestFakeMotionEvent.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_ulong]
    xt.XTestFakeRelativeMotionEvent.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_ulong]
    xt.XTestFakeButtonEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
    xt.XTestFakeKeyEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
    original_engine=subprocess.check_output(['ibus','engine'],text=True).strip()
    try:
        subprocess.run(['ibus','engine','xkb:us::eng'],check=True)
        event=Event();event.client.type=33;event.client.display=display;event.client.window=window
        event.client.message_type=x.XInternAtom(display,b'_NET_ACTIVE_WINDOW',0)
        event.client.format=32;event.client.data[0]=2
        x.XSendEvent(display,x.XDefaultRootWindow(display),0,(1<<20)|(1<<19),c.byref(event))
        x.XFlush(display)
        time.sleep(1)
        xt.XTestFakeMotionEvent(display,-1,click_x,click_y,0)
        xt.XTestFakeButtonEvent(display,1,1,0);xt.XTestFakeButtonEvent(display,1,0,0);x.XFlush(display)
        time.sleep(1)
        active=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
        if hex(window) not in active:raise RuntimeError(f'Editor activation failed: {active}')
        if drag_button==0:
            pass
        elif drag_button is None:
            code=x.XKeysymToKeycode(display,x.XStringToKeysym(b'w'))
            xt.XTestFakeKeyEvent(display,code,1,0);xt.XTestFakeKeyEvent(display,code,0,0);x.XFlush(display)
        else:
            xt.XTestFakeButtonEvent(display,drag_button,1,0);x.XFlush(display)
            time.sleep(.3)
            for i in range(10):
                xt.XTestFakeRelativeMotionEvent(display,3,-1,0);x.XFlush(display);time.sleep(.06)
            xt.XTestFakeButtonEvent(display,drag_button,0,0);x.XFlush(display)
    finally:
        time.sleep(1)
        subprocess.run(['ibus','engine',original_engine],check=False)
        x.XCloseDisplay(display)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('engine', choices=['ue','mujoco','both'])
    args=parser.parse_args()
    output=workspace()/'.local/validation'
    output.mkdir(parents=True,exist_ok=True)
    log=output/f'{args.engine}_gui.log'
    report=output/f'{args.engine}_gui.json'
    before=set(physics_processes())
    if before:
        raise SystemExit('Existing physics process; close it before GUI validation')
    if args.engine=='mujoco':
        extra=['--seconds','6','--distance','.5','--report',str(report)]
    else:
        extra=[f'-Abslog={log}', '-NoSound']
        if args.engine=='both':extra.append('-Go2InputSmoke')
    if report.exists():report.unlink()
    if args.engine!='mujoco':log.write_text('')
    process=launch(engine=args.engine,extra_args=extra)
    start=time.monotonic()
    result={'engine':args.engine,'pid':process.pid}
    owned=set()
    try:
        while time.monotonic()-start<150:
            owned.update(set(physics_processes())-before)
            text=log.read_text(errors='replace') if log.exists() else ''
            if args.engine=='ue' and 'GO2_BRIDGE_READY' in text:
                time.sleep(12)
                result.update(receiver_ready=True,physics_processes=physics_processes())
                break
            if args.engine=='both' and 'GO2_NATIVE_INPUT_SMOKE_PASS' in text and 'GO2_BRIDGE_PASS' in text:
                result.update(receiver_ready=True,actor_readback=True,native_key_smoke=True,physics_processes=physics_processes())
                state_path=workspace()/'MoonSim/mujoco/generated/viewport_last_state.json'
                baseline=json.loads(state_path.read_text())
                focus_and_walk(process.pid)
                moving_seen=False
                for _ in range(45):
                    time.sleep(1)
                    try:current=json.loads(state_path.read_text())
                    except (OSError,ValueError):continue
                    moving_seen |= current.get('moving',False)
                    updated_log=log.read_text(errors='replace')
                    real_ue_key=updated_log.count('GO2_VIEWPORT_KEY w')>=2
                    if real_ue_key and moving_seen and current.get('remaining_m',1)<.08:
                        result['walk']={'moving_seen':True,'ue_key_logged':True,'remaining_m':current['remaining_m'],
                                        'displacement_x_m':(current['geoms'][0][1]-baseline['geoms'][0][1])/100,
                                        'fallen':current['fallen'],'sim_time_s':current['time']}
                        break
                else:
                    result['walk']={'moving_seen':moving_seen,'success':False}
                break
            if args.engine=='mujoco' and time.monotonic()-start>2 and not (output/'mujoco_gui.png').exists():
                subprocess.run(['import','-window','root',str(output/'mujoco_gui.png')],timeout=15,check=False)
            if process.poll() is not None:
                result['exit_code']=process.returncode
                break
            time.sleep(1)
        else:
            result['timeout']=True
        if args.engine!='mujoco':
            subprocess.run(['import','-window','root',str(output/f'{args.engine}_gui.png')],timeout=15,check=False)
            state=workspace()/'MoonSim/mujoco/generated/viewport_last_state.json'
            if args.engine=='both' and state.exists():
                result['last_state']=json.loads(state.read_text())
            report.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='last_state'},indent=2),flush=True)
        if args.engine=='mujoco':
            physics=json.loads(report.read_text())
            assert result.get('exit_code')==0 and physics['finite'] and not physics['fallen'], result
        elif args.engine=='ue':
            assert result.get('receiver_ready') and not result['physics_processes'], result
        else:
            walk=result.get('walk',{})
            assert result.get('actor_readback') and walk.get('displacement_x_m',0)>.3 and not walk.get('fallen',True), result
        (output/f'{args.engine}_process.json').write_text(json.dumps({k:v for k,v in result.items() if k!='last_state'},indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=10)
        for pid in owned-before:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass

if __name__=="__main__":
    main()
