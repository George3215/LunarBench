"""验收换皮地图。

三件事，缺一不可：
1. 源贴图没被换过（sha256 对得上）；
2. 引擎内断言全过（几何、transform、高度场逐点、材质、贴图、Go2 部件数）；
3. 旧地图与旧材质没被改写——这是「只读源资产」这条约定的可执行形式。

引擎内的高度场逐点比对是最重要的一条：它同时证明没做位移、
且副本没有损坏 MuJoCo 那份 1 cm 原始高度场。
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = Path('/home/lry/UE/UE_5.7.4')
DEFAULT_PROJECT = Path('/home/lry/文档/Unreal Projects/Moon/Moon.uproject')
DEFAULT_DST_MAP = '/Game/MoonMacro/Maps/MoonTerrain_Macro01_FullRender'
DEFAULT_TEXTURE_DEST = '/Game/MoonMacro/Textures'
DEFAULT_INSTANCE = '/Game/MoonMacro/Materials/MI_MoonMacro01_Landscape'
SNAPSHOT = ROOT / '.local/moon_macro_pre_snapshot.txt'
PASS_MARKER = 'MACRO_MAP_VALIDATION: PASS'

# 入库的源贴图。改了其中任何一张，生成的地图就不再是同一张皮。
SOURCE_TEXTURES = {
    'moon_macro_01_diff_4k.jpg':
        'cc80ef34a0c558ca3df419e0e15ed4aa01d25e1e9493e3f053eb43761ef9988e',
    'moon_macro_01_nor_gl_4k.exr':
        '72416f90990e6be493a6410d960896d33f19869741f8060618979eb4b9a7531d',
    'moon_macro_01_rough_4k.exr':
        '894eea5fb69b9e327464311ede38ee8c210faa18a6eec499fb5ce8197c26bc9b',
}

failures = []


def check(label, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {label}{"  " + detail if detail else ""}')
    if not ok:
        failures.append(label)
    return ok


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--dst-map', default=DEFAULT_DST_MAP)
    parser.add_argument('--texture-dest', default=DEFAULT_TEXTURE_DEST)
    parser.add_argument('--instance', default=DEFAULT_INSTANCE)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--skip-engine', action='store_true',
                        help='只跑 repo 侧检查，不启动 UE')
    parser.add_argument('--log', type=Path, default=ROOT / '.local/moon_macro_validate.log')
    args = parser.parse_args()

    source_dir = ROOT / 'assets/environments/lunar/materials/moon_macro_01_4k'
    for name, expected in SOURCE_TEXTURES.items():
        path = source_dir / name
        if not path.is_file():
            check(f'source texture {name}', False, 'missing')
        else:
            check(f'source texture {name}', digest(path) == expected)

    if args.snapshot.is_file():
        content_root = args.project.parent / 'Content'
        for line in args.snapshot.read_text().splitlines():
            if not line.strip():
                continue
            expected_sha, expected_mtime, relative = line.split(None, 2)
            path = content_root / relative
            if not path.is_file():
                check(f'untouched {relative}', False, 'disappeared')
                continue
            same = (digest(path) == expected_sha
                    and str(int(path.stat().st_mtime)) == expected_mtime)
            check(f'untouched {relative}', same,
                  '' if same else 'was rewritten')
    else:
        print(f'SKIP  old-asset integrity (no snapshot at {args.snapshot})')

    map_file = args.project.parent / 'Content/MoonMacro/Maps/MoonTerrain_Macro01_FullRender.umap'
    check('new map exists', map_file.is_file(),
          f'{map_file.stat().st_size} bytes' if map_file.is_file() else 'missing')

    if not args.skip_engine:
        editor = args.engine / 'Engine/Binaries/Linux/UnrealEditor-Cmd'
        for comm in Path('/proc').glob('[0-9]*/comm'):
            try:
                if comm.read_text().strip() == 'UnrealEditor' and \
                        str(args.project).encode() in (comm.parent / 'cmdline').read_bytes().split(b'\0'):
                    parser.error('Close this UE project first')
            except OSError:
                pass
        args.log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['LUNARBENCH_WORKSPACE'] = str(ROOT)
        command = [
            str(editor), str(args.project), '-run=MoonTerrainImport', '-ValidateMacroMap',
            '-unattended', '-nosplash',
            f'-DstMap={args.dst_map}', f'-TextureDest={args.texture_dest}',
            f'-InstanceAsset={args.instance}',
        ]
        started = time.monotonic()
        with args.log.open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT)
        text = args.log.read_text(errors='replace')
        for line in text.splitlines():
            if 'VALIDATION_' in line:
                print('      ' + line.split('LogMoonTerrainImport: ')[-1])
        check('engine validation', PASS_MARKER in text,
              f'({time.monotonic() - started:.1f}s, {args.log})')

    print()
    if failures:
        print(f'FAILED: {len(failures)} check(s): {", ".join(failures)}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
