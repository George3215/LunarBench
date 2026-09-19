"""Convert the official local Livox STEP to metre-scale MJCF/UE mesh. No runtime dependency."""
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.IFSelect import IFSelect_RetDone
import trimesh,numpy as np,json
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
reader=STEPControl_Reader();assert reader.ReadFile('.local/sensor_sources/mid-360-asm.stp')==IFSelect_RetDone
reader.TransferRoots();shape=reader.OneShape();mesh=BRepMesh_IncrementalMesh(shape,.15,False,.3,True);mesh.Perform()
writer=StlAPI_Writer();writer.ASCIIMode=False;assert writer.Write(shape,'/tmp/task1-mid360-cad.stl')
m=trimesh.load('/tmp/task1-mid360-cad.stl',force='mesh');bottom=float(m.vertices[:,1].min())
R=np.array([[-1,0,0],[0,0,1],[0,1,0]])
v=m.vertices@R.T;v[:,2]-=bottom+39.5;m.vertices=v*.001
m.export('MoonSim/assets/robots/sensors/mid360_official.obj');print('mesh',len(m.vertices),len(m.faces),m.bounds)
Path('.local/sensor_sources/mid360_conversion.json').write_text(json.dumps({'converter':'OpenCascade 8, deflection .15 mm, angle .3 rad','cad_to_sensor_rotation':R.tolist(),'cad_bottom_y_mm':bottom,'optical_height_from_bottom_mm':39.5,'unit_scale':.001,'bounds_m':m.bounds.tolist(),'faces':len(m.faces)},indent=2))
