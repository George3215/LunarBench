"""用 moon_macro_01_4k 贴图给地形换皮，另存为一张新地图。

这是 tools/deploy_ue_plugin.py 的姊妹脚本：后者把插件源码编译进 UE 工程，
前者调用已编译好的 commandlet 去生成地图资产。两者都要求 UE 编辑器已关闭。

源地图与源材质一律只读——只换 ALandscape 的材质并另存到新路径。
几何（高度数据、transform、岩石、Go2 部件）原样带过去。
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
DEFAULT_SRC_MAP = '/Game/MoonGo2/Maps/MoonTerrain_Go2_FullRender'
DEFAULT_DST_MAP = '/Game/MoonMacro/Maps/MoonTerrain_Macro01_FullRender'
DEFAULT_TEXTURE_ROOT = 'assets/environments/lunar/materials/moon_macro_01_4k'
DEFAULT_MATERIAL = '/Game/MoonMacro/Materials/M_MoonMacro_Landscape'
DEFAULT_INSTANCE = '/Game/MoonMacro/Materials/MI_MoonMacro01_Landscape'
DEFAULT_TEXTURE_DEST = '/Game/MoonMacro/Textures'
SUCCESS_MARKER = 'MACRO_MAP_SUCCESS'


# 源资产只读。构建前记下 sha256 与 mtime，validate_moon_macro_map.py 据此断言
# 它们没被改写——这是「只读」这条约定的可执行形式，而不是一句注释。
GUARDED = (
    'MoonTerrainV2/Maps/MoonTerrain_Full.umap',
    'MoonGo2/Maps/MoonTerrain_Go2_FullRender.umap',
    'MoonTerrain/Maps/MoonTerrain.umap',
    'MoonTerrainV2/Materials/M_MoonLandscape_Full.uasset',
    'MoonTerrainV2/Materials/M_MoonRock_Full.uasset',
)


def write_snapshot(project, path):
    content = project.parent / 'Content'
    lines = []
    for relative in GUARDED:
        asset = content / relative
        if not asset.is_file():
            print(f'note: no such asset, not guarded: {relative}')
            continue
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        lines.append(f'{digest}  {int(asset.stat().st_mtime)}  {relative}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    print(f'Snapshot of {len(lines)} source asset(s): {path}', flush=True)


def editor_running(project):
    """编辑器开着的话 commandlet 会和它抢同一个工程目录。"""
    for comm in Path('/proc').glob('[0-9]*/comm'):
        try:
            if comm.read_text().strip() != 'UnrealEditor':
                continue
            if str(project).encode() in (comm.parent / 'cmdline').read_bytes().split(b'\0'):
                return True
        except OSError:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--src-map', default=DEFAULT_SRC_MAP)
    parser.add_argument('--dst-map', default=DEFAULT_DST_MAP)
    parser.add_argument('--texture-root', default=DEFAULT_TEXTURE_ROOT)
    parser.add_argument('--texture-dest', default=DEFAULT_TEXTURE_DEST)
    parser.add_argument('--material', default=DEFAULT_MATERIAL)
    parser.add_argument('--instance', default=DEFAULT_INSTANCE)
    parser.add_argument('--tile-cm', type=float, default=1000.0,
                        help='世界空间平铺边长（cm）。4K 贴图下 1000 约合 2.4 mm/像素')
    parser.add_argument('--specular', type=float, default=0.35)
    parser.add_argument('--normal-scale', type=float, default=1.0,
                        help='法线强度；1.0 是恒等变换，调小即压平')
    parser.add_argument('--log', type=Path, default=ROOT / '.local/moon_macro_build.log')
    args = parser.parse_args()

    editor = args.engine / 'Engine/Binaries/Linux/UnrealEditor-Cmd'
    for path in (editor, args.project):
        if not path.exists():
            parser.error(f'Not found: {path}')
    if editor_running(args.project):
        parser.error('Close this UE project before generating a map from it')

    texture_dir = ROOT / args.texture_root
    for name in ('moon_macro_01_diff_4k.jpg', 'moon_macro_01_nor_gl_4k.exr',
                 'moon_macro_01_rough_4k.exr'):
        if not (texture_dir / name).is_file():
            parser.error(f'Missing source texture: {texture_dir / name}')

    args.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['LUNARBENCH_WORKSPACE'] = str(ROOT)
    write_snapshot(args.project, ROOT / '.local/moon_macro_pre_snapshot.txt')

    # 不要加 -nullrhi：纹理的平台数据编译需要 RHI，
    # 无 RHI 时导入会停在 32px 的占位尺寸上。
    command = [
        str(editor), str(args.project), '-run=MoonTerrainImport', '-MoonMacroMap',
        '-unattended', '-nosplash',
        f'-SrcMap={args.src_map}', f'-DstMap={args.dst_map}',
        f'-TextureRoot={args.texture_root}', f'-TextureDest={args.texture_dest}',
        f'-MaterialAsset={args.material}', f'-InstanceAsset={args.instance}',
        f'-TileCm={args.tile_cm:g}', f'-Specular={args.specular:g}',
        f'-NormalScale={args.normal_scale:g}',
    ]
    print(f'{" ".join(command)}\n-> {args.log}', flush=True)
    started = time.monotonic()
    with args.log.open('w') as stream:
        code = subprocess.run(command, cwd=ROOT, env=env,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    text = args.log.read_text(errors='replace')
    elapsed = time.monotonic() - started
    print(f'exit={code} elapsed={elapsed:.1f}s')
    if SUCCESS_MARKER not in text:
        print(f'FAILED: {SUCCESS_MARKER} not in log', file=sys.stderr)
        return 1
    for line in text.splitlines():
        if 'MACRO_' in line:
            print(line.split('LogMoonTerrainImport: ')[-1])
    return 0


if __name__ == '__main__':
    sys.exit(main())
