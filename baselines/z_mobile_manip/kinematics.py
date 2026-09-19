"""Robot description adapter only. No simulator object or world state is read."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from z_manip.kinematics.chain import Joint, KinematicChain


def load_chain():
    root=ET.parse(Path(__file__).resolve().parents[2]/'assets/robots/go2_piper/go2_piper.xml').getroot()
    base=root.find(".//body[@name='base_link']")
    joints=[]
    def walk(body,parent):
        name=body.attrib['name'];T=np.eye(4)
        T[:3,3]=np.fromstring(body.get('pos','0 0 0'),sep=' ')
        if 'quat' in body.attrib:
            q=np.fromstring(body.get('quat'),sep=' ');T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        elif 'euler' in body.attrib:T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(body.get('euler'),sep=' ')).as_matrix()
        j=body.find('joint');active=j is not None
        limits=np.fromstring(j.get('range'),sep=' ') if active else [0,0]
        joints.append(Joint(j.get('name') if active else name+'_fixed','revolute' if active else 'fixed',
            parent,name,T,np.fromstring(j.get('axis','0 0 1'),sep=' ') if active else np.array([0,0,1]),
            float(limits[0]),float(limits[1]),.5))
        if name=='link6':
            # Piper contact TCP: closing +Y, approach +Z; right-handed source-gripper convention.
            tip=np.eye(4);tip[:3,3]=[0,0,.18];tip[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix()
            joints.append(Joint('tcp_fixed','fixed',name,'grasp_tcp',tip,np.array([0,0,1]),0,0,.5));return
        children=[b for b in body.findall('body') if b.get('name') in ['arm_base','link1','link2','link3','link4','link5','link6']]
        for child in children:walk(child,name)
    walk(base.find("body[@name='arm_base']"),'base_link')
    return KinematicChain(joints,'base_link','grasp_tcp')
