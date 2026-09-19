"""Back up and deploy plugin source to a local UE project, then build it."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def workspace():
    return Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--engine', type=Path, required=True)
args = parser.parse_args()
project = args.project.resolve()
if not project.is_file():
    parser.error(f'Project not found: {project}')
for comm in Path('/proc').glob('[0-9]*/comm'):
    try:
        if comm.read_text().strip() == 'UnrealEditor' and str(project).encode() in (comm.parent/'cmdline').read_bytes().split(b'\0'):
            parser.error('Close this UE project before rebuilding its plugin')
    except OSError:
        pass
source = Path(__file__).resolve().parents[1]/'ue/importer/MoonTerrainImporter'
destination = project.parent/'Plugins/MoonTerrainImporter'
backup = workspace()/'.local/backups'/datetime.now().strftime('plugin-%Y%m%d-%H%M%S-%f')
if (destination/'Source').exists():
    shutil.copytree(destination/'Source', backup/'Source')
    shutil.copy2(destination/'MoonTerrainImporter.uplugin', backup/'MoonTerrainImporter.uplugin')
    print(f'Source backup: {backup}', flush=True)
shutil.copytree(source/'Source', destination/'Source', dirs_exist_ok=True)
shutil.copy2(source/'MoonTerrainImporter.uplugin', destination/'MoonTerrainImporter.uplugin')
subprocess.run([str(args.engine/'Engine/Build/BatchFiles/Linux/Build.sh'), project.stem+'Editor',
                'Linux', 'Development', f'-Project={project}', '-WaitMutex'], check=True)
