"""Versioned JSON + typed binary arrays over localhost HTTP, no engine imports."""
import http.client
import json
import struct
import time
import numpy as np

HOST = '127.0.0.1'
PORT = 19530
MAX_BYTES = 16*1024*1024


def pack(meta, arrays=None):
    meta=dict(meta);meta['schema']='moonsim.bridge.v1';blocks=[];specs=[]
    for name,value in (arrays or {}).items():
        a=np.ascontiguousarray(value)
        if a.dtype.kind not in 'uifb':raise ValueError('Only numeric arrays are supported')
        specs.append({'name':name,'dtype':a.dtype.str,'shape':list(a.shape),'bytes':a.nbytes})
        blocks.append(a.tobytes())
    meta['arrays']=specs
    header=json.dumps(meta,allow_nan=False,separators=(',',':')).encode()
    return struct.pack('<I',len(header))+header+b''.join(blocks)


def unpack(payload):
    length=struct.unpack_from('<I',payload)[0]
    meta=json.loads(payload[4:4+length]);offset=4+length;arrays={}
    if meta['schema']!='moonsim.bridge.v1':raise ValueError('Unknown bridge schema')
    for a in meta['arrays']:
        dt=np.dtype(a['dtype']);count=int(np.prod(a['shape']))
        if dt.kind not in 'uifb' or count*dt.itemsize!=a['bytes']:raise ValueError('Invalid array')
        arrays[a['name']]=np.frombuffer(payload,dtype=dt,count=count,offset=offset).reshape(a['shape']).copy()
        offset+=a['bytes']
    if offset!=len(payload):raise ValueError('Invalid frame length')
    return meta,arrays


class Client:
    def __init__(self, timeout=.5):
        self.connection=http.client.HTTPConnection(HOST,PORT,timeout=timeout)

    def request(self,method,path,body=None):
        try:
            self.connection.request(method,path,body,{'Content-Type':'application/octet-stream'})
            response=self.connection.getresponse();data=response.read()
            if response.status==404:return None
            if response.status!=200:raise RuntimeError(data.decode(errors='replace'))
            return data
        except Exception:
            self.connection.close()
            raise

    def publish(self,topic,meta,arrays=None):
        return self.request('POST','/frame/'+topic,pack(meta,arrays))

    def read(self,topic):
        payload=self.request('GET','/frame/'+topic)
        return unpack(payload) if payload else None

    def close(self):self.connection.close()
