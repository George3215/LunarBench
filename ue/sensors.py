"""UE-side RGB-D producer. Standard library only; publishes through the bridge."""
import http.client
import json
import math
from pathlib import Path
import struct
import threading
import time

WIDTH,HEIGHT=640,360
HFOV=69.
CAMERA_HZ=5.


class CameraSource:
    def __init__(self,unreal,output):
        self.unreal=unreal;self.file=Path(output)/'camera_rgbd.bin';self.last=0.;self.seq=0
        self.pending=None;self.closed=False;self.error=None;self.lock=threading.Lock()
        self.thread=threading.Thread(target=self._send,daemon=True);self.thread.start()

    def capture(self,packet):
        now=time.monotonic()
        if now-self.last<1/CAMERA_HZ or 'sensor_camera' not in packet:return
        self.last=now;p=packet['sensor_camera'];u=self.unreal
        rotation=u.MathLibrary.make_rot_from_xz(u.Vector(*p['forward']),u.Vector(*p['up']))
        started=time.monotonic()
        if not u.MoonViewportLibrary.capture_task_rgbd(str(self.file),u.Vector(*p['position_cm']),rotation,WIDTH,HEIGHT,HFOV):
            self.error='UE render target readback failed';return
        raw=self.file.read_bytes()
        if len(raw)!=WIDTH*HEIGHT*7:self.error='RGB-D readback size mismatch';return
        self.seq+=1;f=WIDTH/(2*math.tan(math.radians(HFOV/2)))
        meta={'schema':'moonsim.bridge.v1','source':'ue_scene_capture','seq':self.seq,
              'epoch':packet['epoch'],'sim_time_s':packet['time'],'render_state_seq':packet.get('seq',0),
              'sent_ns':time.monotonic_ns(),'state_sent_ns':packet.get('sent_ns'),
              'frame_id':'d435_color_optical','sensor_model':'Intel RealSense D435',
              'depth_model':'ideal UE axial depth registered to color; not stereo reconstruction',
              'mount_link':'link6','depth_range_m':[.28,10.],
              'native_depth_fov_deg':[87.,58.],'native_rgb_fov_deg':[69.,42.],
              'native_rgb_max_hz':30.,'configured_hz':CAMERA_HZ,'encoding':'rgb8 + float32 axial depth metres; invalid=0',
              'K':[[f,0,(WIDTH-1)/2],[0,f,(HEIGHT-1)/2],[0,0,1]],
              'T_base_camera':p['T_base_camera'],'capture_ms':(time.monotonic()-started)*1000,
              'arrays':[{'name':'rgb','dtype':'|u1','shape':[HEIGHT,WIDTH,3],'bytes':WIDTH*HEIGHT*3},
                        {'name':'depth_m','dtype':'<f4','shape':[HEIGHT,WIDTH],'bytes':WIDTH*HEIGHT*4}]}
        header=json.dumps(meta,separators=(',',':')).encode();payload=struct.pack('<I',len(header))+header+raw
        with self.lock:self.pending=payload

    def _send(self):
        client=http.client.HTTPConnection('127.0.0.1',19530,timeout=.5)
        while not self.closed:
            with self.lock:payload=self.pending;self.pending=None
            if payload:
                try:
                    client.request('POST','/frame/camera',payload,{'Content-Type':'application/octet-stream'})
                    response=client.getresponse();data=response.read()
                    if response.status!=200:raise RuntimeError(data.decode(errors='replace'))
                    self.error=None
                except Exception as e:self.error=str(e);client.close()
            time.sleep(.01)
        client.close()

    def close(self):self.closed=True;self.thread.join(timeout=1)
