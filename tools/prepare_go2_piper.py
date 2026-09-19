"""Compose the existing Go2 body and the existing Piper arm adapter into one self-contained MJCF.

No policy or network dependency. Go2 meshes, inertia, defaults, leg actuators and sensors are
copied verbatim from assets/robots/go2_demo/go2/go2.xml; the arm keeps the position servos,
coupled finger slides and tool_tip site of assets/robots/piper/arm.xml. Neither model is
modified in place. Only the mounting transform is taken from the upstream LeggedManip_Lab
GO2-PIPER platform, because that is the one thing neither local model records.

The result is self-contained on purpose: MuJoCo honours only the top-level <compiler meshdir>,
so an <include> of a second robot would resolve that robot's meshes against the wrong directory.
Both mesh sets are therefore inlined under a shared meshdir with relative sub-paths.
"""
from pathlib import Path
import copy,hashlib,json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
ROBOTS=ROOT/'assets/robots'
BODY=ROBOTS/'go2_demo/go2/go2.xml'
ARM=ROBOTS/'piper/arm.xml'
UPSTREAM=Path('/home/lry/ATEC_UE_sim/third_party/LeggedManip_Lab/mujoco/robots/go2_piper/go2piper.xml')
OUT=ROBOTS/'go2_piper'

MOUNT='0 0 0.06'  # upstream go2piper.xml: <body name="Piper" pos="0.0 0.0 0.06"> under base_link
ARM_HOME=[0.,1.1,-1.4,0.,.3,0.,.02]  # keep in step with tasks/task1_collect/robot.py::ARM_HOME
GO2_MESH_NAMES=['base_0','base_1','base_2','base_3','base_4','hip_0','hip_1','thigh_0','thigh_1',
                'thigh_mirror_0','thigh_mirror_1','calf_0','calf_1','calf_mirror_0','calf_mirror_1','foot']


def _rewrite(asset,prefix,out):
    for element in asset:
        item=copy.deepcopy(element)
        if item.tag=='mesh':item.set('file',f'{prefix}/{Path(item.get("file")).name}')
        out.append(item)


def build():
    go2=ET.parse(BODY).getroot();arm=ET.parse(ARM).getroot()
    root=ET.Element('mujoco',model='go2_piper')
    # meshdir is relative to this file, so both mesh sets resolve through one directory.
    ET.SubElement(root,'compiler',angle='radian',autolimits='true',meshdir='../')
    option=copy.deepcopy(go2.find('option'));option.set('timestep','.002');option.set('iterations','30')
    root.append(option)
    root.append(copy.deepcopy(go2.find('default')))
    asset=ET.SubElement(root,'asset')
    # Go2 meshes first: UE addresses them by mesh id as SM_Go2_<id>, and the task keeps geom order.
    _rewrite(go2.find('asset'),'go2_demo/go2/assets',asset)
    _rewrite(arm.find('asset'),'piper/assets',asset)

    world=ET.SubElement(root,'worldbody')
    base=copy.deepcopy(go2.find("worldbody/body[@name='base_link']"))
    world.append(base)
    mounted=copy.deepcopy(arm.find("worldbody/body[@name='arm_base']"))
    mounted.set('pos',MOUNT)
    # The pedestal models the rover deck; the Go2 back plate is the mounting surface here.
    mounted.remove(mounted.find("geom[@name='mounting_pedestal']"))
    # base_link carries childclass="go2" (geom margin .001, joint frictionloss .2). The arm must
    # behave exactly as it does in piper/arm.xml, so pin the two inherited values explicitly.
    for geom in mounted.iter('geom'):geom.set('margin','0')
    for joint in mounted.iter('joint'):joint.set('frictionloss','0')
    base.append(mounted)

    actuator=ET.SubElement(root,'actuator')
    for element in go2.find('actuator'):actuator.append(copy.deepcopy(element))
    for element in arm.find('actuator'):actuator.append(copy.deepcopy(element))
    root.append(copy.deepcopy(go2.find('sensor')))
    for tag in ('equality','contact'):root.append(copy.deepcopy(arm.find(tag)))
    # Extend the demo home key so its qpos/ctrl lengths still match the enlarged model.
    keyframe=copy.deepcopy(go2.find('keyframe'));home=keyframe.find("key[@name='home']")
    home.set('qpos',home.get('qpos')+' '+' '.join(f'{v:g}' for v in ARM_HOME+[ARM_HOME[-1]]))
    home.set('ctrl',home.get('ctrl')+' '+' '.join(f'{v:g}' for v in ARM_HOME))
    root.append(keyframe)

    OUT.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    (OUT/'go2_piper.xml').write_text(ET.tostring(root,encoding='unicode'))
    check()
    record={'source':[str(BODY),str(ARM)],'mount_source':str(UPSTREAM),'mount':MOUNT,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BODY,ARM)},
            'mesh_sha256':json.loads((ROBOTS/'piper/source.json').read_text())['mesh_sha256'],
            'modifications':['Composed existing Go2 body and existing Piper arm into one self-contained MJCF',
                             'Both mesh sets referenced through one meshdir with relative sub-paths; no mesh copied',
                             'Mounted the arm on base_link at the upstream GO2-PIPER transform',
                             'Removed the rover mounting pedestal','Pinned inherited geom margin and joint frictionloss',
                             'Extended the demo home keyframe to the enlarged model'],
            'not_claimed':['hardware calibration','trained control','successful grasping',
                           'upstream whole-body policy','upstream torque interface']}
    (OUT/'source.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


def check():
    import numpy as np
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(OUT/'go2_piper.xml'))
    assert m.nu==19,'expected 12 leg motors + 6 arm servos + 1 gripper, got %d'%m.nu
    assert m.nq==27 and m.nv==26,(m.nq,m.nv)  # 7+12+6+2 and 6+12+6+2
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,i) for i in range(m.nmesh)]
    assert names[:16]==GO2_MESH_NAMES,names[:16]
    assert names[16:]==['arm_base_link','link1','link2','link3','link4','link5','link6','gripper_base','link7','link8'],names[16:]
    arm_base=m.body('arm_base').id
    assert m.body_parentid[arm_base]==m.body('base_link').id and np.allclose(m.body_pos[arm_base],[0,0,.06])
    for joint in ('joint1','joint7','joint8'):m.joint(joint)
    m.site('tool_tip')
    arm_dofs=[m.jnt_dofadr[m.joint(f'joint{i}').id] for i in range(1,9)]
    assert not np.any(m.dof_frictionloss[arm_dofs]),'arm joints inherited go2 frictionloss'
    assert np.all(m.dof_frictionloss[[m.jnt_dofadr[m.joint(f'{leg}_{part}_joint').id]
        for leg in ('FR','FL','RR','RL') for part in ('hip','thigh','calf')]]==.2),'leg joints lost go2 frictionloss'
    arm_bodies={m.body(name).id for name in ('arm_base',)+tuple(f'link{i}' for i in range(1,9))}
    arm_geoms=[i for i in range(m.ngeom) if m.geom_bodyid[i] in arm_bodies]
    assert np.all(m.geom_margin[arm_geoms]==0),'arm geoms inherited go2 margin'
    assert m.key_qpos.shape==(1,m.nq) and m.key_ctrl.shape==(1,m.nu)
    print(f'go2_piper.xml: nu={m.nu} nq={m.nq} nv={m.nv} nmesh={m.nmesh} ngeom={m.ngeom} nbody={m.nbody}')


if __name__=='__main__':build()
