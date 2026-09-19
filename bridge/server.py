"""Latest-frame broker and sensor viewer. Only sensor schemas can cross this boundary."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import io
import json
import struct
import threading
import time
import socket
import numpy as np
from PIL import Image
from .client import HOST, PORT, MAX_BYTES, unpack, pack

SOURCES={'camera':'ue_scene_capture','lidar':'mujoco_raycast','robot':'mujoco_proprioception',
         'action':'z_mobile_manip','baseline':'z_mobile_manip','target':'operator_image_roi'}
latest={};counts={};lock=threading.Lock()
OUTPUT=Path(__file__).resolve().parents[1]/'tasks/task1_collect/generated/bridge'
OUTPUT.mkdir(parents=True,exist_ok=True)


def status():
    with lock:
        return {k:{'seq':v[1].get('seq'), 'epoch':v[1].get('epoch'),
                   'sim_time_s':v[1].get('sim_time_s'), 'source':v[1]['source'],
                   'age_s':time.monotonic()-v[2], 'received':counts[k],
                   'reason':v[1].get('reason'),'applied_action_seq':v[1].get('applied_action_seq'),
                   'arrays':v[1].get('arrays',[])} for k,v in latest.items()}


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def setup(self):
        super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

    def log_message(self,*args):pass

    def reply(self,data,content='application/octet-stream',code=200):
        self.send_response(code);self.send_header('Content-Type',content)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(data)

    def do_POST(self):
        try:
            topic=self.path.removeprefix('/frame/')
            if topic not in SOURCES:raise ValueError('Unknown topic')
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>MAX_BYTES:raise ValueError('Invalid frame size')
            payload=self.rfile.read(size)
            if topic=='target':
                roi=json.loads(payload)
                with lock:camera=latest.get('camera')
                if not camera:raise ValueError('Camera is not available')
                x,y,w,h=map(int,roi['bbox'])
                if w<3 or h<3:raise ValueError('Select a nonempty image box')
                meta={'source':SOURCES[topic],'seq':time.monotonic_ns(),'epoch':camera[1]['epoch'],
                      'sim_time_s':camera[1]['sim_time_s'],'camera_seq':camera[1]['seq'],
                      'bbox':[x,y,x+w,y+h],'sent_ns':time.monotonic_ns()}
                payload=pack(meta)
            meta,arrays=unpack(payload)
            if meta.get('source')!=SOURCES[topic]:raise ValueError('Invalid topic source')
            if any(k in meta for k in ('rocks','rock_positions','world_pose','object_id','score','scene')):
                raise ValueError('World-truth fields are not policy inputs')
            required={'camera':{'rgb','depth_m'},'lidar':{'points','ranges_m'},
                      'robot':{'joint_position','joint_velocity','attitude_rpy','angular_velocity','hold_action'}}
            if topic in required and set(arrays)!=required[topic]:raise ValueError('Unexpected sensor arrays')
            with lock:
                previous=latest.get(topic)
                if previous and previous[1].get('epoch')==meta.get('epoch') and meta['seq']<=previous[1]['seq']:
                    raise ValueError('Nonmonotonic sequence')
                latest[topic]=(payload,meta,time.monotonic());counts[topic]=counts.get(topic,0)+1
            self.reply(b'{"accepted":true}','application/json')
        except (ValueError,KeyError,TypeError,struct.error) as e:self.reply(str(e).encode(),code=400)

    def do_GET(self):
        if self.path=='/':
            return self.reply(Path(__file__).with_name('viewer.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/status':return self.reply(json.dumps(status()).encode(),'application/json')
        if self.path.startswith('/frame/'):
            with lock:frame=latest.get(self.path[7:])
            return self.reply(frame[0] if frame else b'not ready',code=200 if frame else 404)
        if self.path.startswith('/image/'):
            topic='lidar' if self.path.startswith('/image/lidar') else 'camera'
            with lock:frame=latest.get(topic)
            if not frame:return self.reply(b'not ready',code=404)
            meta,a=unpack(frame[0])
            if topic=='camera':
                if self.path.startswith('/image/depth'):
                    d=a['depth_m'];u=np.clip(d/8*255,0,255).astype(np.uint8)
                    image=Image.fromarray(np.stack((u,255-u,np.zeros_like(u)),axis=-1))
                else:image=Image.fromarray(a['rgb'])
            else:
                p=a['points'];p=p[np.isfinite(p).all(axis=1)]
                pixels=np.zeros((480,480,3),np.uint8)
                x=(240-p[:,1]*20).astype(int);y=(240-p[:,0]*20).astype(int)
                valid=(x>=0)&(x<480)&(y>=0)&(y<480)
                pixels[y[valid],x[valid]]=[80,225,160];pixels[237:243,237:243]=[255,100,80]
                image=Image.fromarray(pixels)
            buf=io.BytesIO();image.save(buf,format='PNG')
            return self.reply(buf.getvalue(),'image/png')
        self.reply(b'not found',code=404)


def main():
    server=ThreadingHTTPServer((HOST,PORT),Handler);server.daemon_threads=True
    def record():
        while True:
            document=status();document['schema']='moonsim.bridge.status.v1'
            tmp=OUTPUT/'status.tmp';tmp.write_text(json.dumps(document,indent=2));tmp.replace(OUTPUT/'status.json')
            time.sleep(1)
    threading.Thread(target=record,daemon=True).start()
    print(f'BRIDGE_READY http://{HOST}:{PORT}',flush=True)
    server.serve_forever(poll_interval=.2)

if __name__=='__main__':main()
