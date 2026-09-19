"""Extract the existing local Piper MJCF and mount it on the reference rover.

No policy or network dependency. Original mesh and inertial data are retained;
position servos, symmetric finger slides and the rover mounting are task additions.
"""
from pathlib import Path
import copy,json,shutil,hashlib
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/lry/ATEC_UE_sim/third_party/LeggedManip_Lab')
MODEL=SOURCE/'mujoco/robots/go2_piper/go2piper.xml'
OUT=ROOT/'assets/robots/piper'

def build():
    original=ET.parse(MODEL).getroot();arm=copy.deepcopy(original.find(".//body[@name='Piper']"))
    arm.set('name','arm_base');arm.set('pos','0 0 0')
    for g in arm.iter('geom'):
        g.set('group','2')
        if g.get('contype')!='0':g.set('condim','3');g.set('friction','.8 .02 .001')
    for joint in arm.iter('joint'):joint.set('armature','.01')
    for number in (7,8):
        finger=arm.find(f".//body[@name='link{number}']")
        ET.SubElement(finger,'joint',name=f'joint{number}',type='slide',axis='0 0 1',range='0 .035',damping='2',armature='.001')
    ET.SubElement(arm.find(".//body[@name='end_effector']"),'site',name='tool_tip',size='.008',rgba='0 1 0 1')
    used={g.get('mesh') for g in arm.iter('geom') if g.get('mesh')}
    root=ET.Element('mujoco',model='Piper fixed base task adapter')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true',meshdir='assets')
    ET.SubElement(root,'option',timestep='.002',gravity='0 0 -9.81',iterations='30')
    asset=ET.SubElement(root,'asset');(OUT/'assets').mkdir(parents=True,exist_ok=True)
    hashes={}
    for mesh in original.findall('asset/mesh'):
        if mesh.get('name') not in used:continue
        item=copy.deepcopy(mesh);file=Path(mesh.get('file')).name;item.set('file',file);asset.append(item)
        shutil.copy2(MODEL.parent/'assets'/file,OUT/'assets'/file)
        hashes[file]=hashlib.sha256((OUT/'assets'/file).read_bytes()).hexdigest()
    world=ET.SubElement(root,'worldbody');world.append(arm)
    ET.SubElement(arm,'geom',name='mounting_pedestal',type='cylinder',size='.13 .10',pos='0 0 -.10',rgba='.2 .2 .23 1',group='2')
    actuator=ET.SubElement(root,'actuator')
    for j,force in zip(range(1,7),(20,20,15,7,5,5)):
        joint=arm.find(f".//joint[@name='joint{j}']")
        ET.SubElement(actuator,'position',name=f'arm_joint{j}',joint=f'joint{j}',kp='100',kv='10',ctrlrange=joint.get('range'),forcerange=f'-{force} {force}')
    ET.SubElement(actuator,'position',name='gripper',joint='joint7',kp='500',kv='20',ctrlrange='0 .035',forcerange='-20 20')
    eq=ET.SubElement(root,'equality');ET.SubElement(eq,'joint',joint1='joint7',joint2='joint8',polycoef='0 1 0 0 0')
    # Nonadjacent finger meshes should contact objects, not fight each other when closed.
    contact=ET.SubElement(root,'contact');ET.SubElement(contact,'exclude',body1='link7',body2='link8')
    # Fixed bases are welded to world; explicitly exclude the overlapping bearing shells.
    ET.SubElement(contact,'exclude',body1='arm_base',body2='link1')
    ET.indent(root);ET.ElementTree(root).write(OUT/'arm.xml',encoding='unicode')
    mobile=ET.parse(ROOT/'assets/robots/reference_rover/rover.xml').getroot();mobile.set('model','Reference rover with Piper arm')
    mobile.find('compiler').set('meshdir','assets');mobile.find('compiler').set('autolimits','true')
    mobile.find('asset').extend(copy.deepcopy(list(asset)))
    mounted=copy.deepcopy(arm);mounted.set('pos','0 0 .10')
    mounted.remove(mounted.find("geom[@name='mounting_pedestal']"))
    mobile.find("worldbody/body[@name='base_link']").append(mounted)
    mobile.find('actuator').extend(copy.deepcopy(list(actuator)))
    mobile.append(copy.deepcopy(eq));mobile.append(copy.deepcopy(contact))
    ET.indent(mobile);ET.ElementTree(mobile).write(OUT/'rover_piper.xml',encoding='unicode')
    shutil.copy2(SOURCE/'LICENSE',OUT/'UPSTREAM_LICENSE')
    record={'source':str(MODEL),'source_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'mesh_sha256':hashes,'modifications':['Extracted arm only','Position servos instead of upstream torque motors','Added coupled finger slide joints','Authored mounting on reference rover'],
            'not_claimed':['hardware calibration','trained control','successful grasping']}
    (OUT/'source.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':build()
