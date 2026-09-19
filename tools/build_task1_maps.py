"""给 TASK1 的三个地形裁片各换一次皮，另存成三张 UE 地图。

为什么要换皮而不是运行时改材质：Linux 版 UE 5.7 没有导出
EditorSetLandscapeMaterial（见 MoonTerrainImportCommandlet.cpp:503），
仓库里换地形材质一律走 -MoonMacroMap 这个离线 commandlet。
本脚本就是把它跑三遍——每次换一组贴图。

C++ 侧没有任何改动。唯一写死的是三个**源文件名**（MoonMacro.inl:362），
所以每个裁片的贴图先按那个固定文件名摆进 .local/task1_textures/<patch>/，
再用 -TextureRoot 指过去。其余（DstMap / TextureDest / MaterialAsset /
InstanceAsset / TileCm）本来就是这个 commandlet 的参数。

三个 -TextureDest 必须互不相同：纹理资产名 T_MoonMacro01_A/N/R 是写死的，
而 ImportTexture 发现资产已存在就复用、不会重读源文件（MoonMacro.inl:68），
撞名会静默用上一张图。

需要 UE 工程已关闭，且插件已编译（tools/deploy_ue_plugin.py）。

    python tools/build_task1_maps.py                  # 三个裁片都做
    python tools/build_task1_maps.py --patch center --tile-cm 1000
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import build_moon_macro_map as macro                       # noqa: E402

TERRAIN_ROOT = ROOT / 'assets/environments/lunar/terrain'
STAGE_ROOT = ROOT / '.local/task1_textures'
PATCHES = ('center', 'southwest', 'northeast')
CONVERT = 'convert'          # ImageMagick；cv2 这个构建写不了 EXR

# commandlet 认死的三个源文件名（MoonMacro.inl:362-367）。别改。
STAGED = (
    ('{name}_diff_4k.png', 'moon_macro_01_diff_4k.jpg', 'jpg'),
    ('{name}_nor_gl_4k.png', 'moon_macro_01_nor_gl_4k.exr', 'exr'),
    ('{name}_rough_4k.png', 'moon_macro_01_rough_4k.exr', 'exr'),
)


def asset_prefix(patch):
    return 'Task1' + patch.capitalize()


def stage(patch):
    """把烘焙产物摆成 commandlet 要的文件名与格式，返回暂存目录。"""
    name = f'task1_{patch}'
    source = TERRAIN_ROOT / name / 'textures'
    if not source.is_dir():
        raise SystemExit(f'{patch}: 找不到烘焙产物 {source}；'
                         f'先跑 tools/prepare_task1_terrain.py 和 terrain_bake.py')
    target = STAGE_ROOT / patch
    target.mkdir(parents=True, exist_ok=True)

    for template, staged_name, kind in STAGED:
        png = source / template.format(name=name)
        if not png.is_file():
            raise SystemExit(f'{patch}: 缺少贴图 {png}')
        out = target / staged_name
        if kind == 'jpg':
            # commandlet 吃 jpg，得转。95 够用，反照率本来也没有高频细节。
            image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SystemExit(f'{patch}: 写 jpg 失败 {out}')
        else:
            # 法线/粗糙度必须走 EXR：UE 侧按线性数据读，PNG 的 8 位会丢精度。
            # 源保持 GL 约定不预翻转 —— ImportTexture 会设 bFlipGreenChannel=true
            # 交给 UE 翻（MoonMacro.inl:116）。
            done = subprocess.run([CONVERT, str(png), '-depth', '16', str(out)],
                                  capture_output=True, text=True)
            if done.returncode or not out.is_file():
                raise SystemExit(f'{patch}: {CONVERT} 转 EXR 失败：{done.stderr.strip()[:300]}')
        print(f'  {staged_name}  <-  {png.name}  ({out.stat().st_size / 1e6:.1f} MB)')
    return target


def build_one(args, patch):
    prefix = asset_prefix(patch)
    texture_root = STAGE_ROOT / patch
    dst_map = f'/Game/MoonMacro/Maps/MoonTerrain_{prefix}'
    texture_dest = f'/Game/MoonMacro/Textures_{prefix}'

    print(f'\n=== {patch} -> {dst_map} ===')
    if not args.skip_stage:
        stage(patch)

    command = [
        str(args.engine / 'Engine/Binaries/Linux/UnrealEditor-Cmd'),
        str(args.project), '-run=MoonTerrainImport', '-MoonMacroMap',
        '-unattended', '-nosplash',
        f'-SrcMap={args.src_map}', f'-DstMap={dst_map}',
        f'-TextureRoot={texture_root.relative_to(ROOT)}',
        f'-TextureDest={texture_dest}',
        f'-MaterialAsset=/Game/MoonMacro/Materials/M_MoonMacro_{prefix}',
        f'-InstanceAsset=/Game/MoonMacro/Materials/MI_MoonMacro_{prefix}',
        f'-TileCm={args.tile_cm:g}', f'-Specular={args.specular:g}',
        f'-NormalScale={args.normal_scale:g}',
    ]
    log = ROOT / f'.local/task1_map_{patch}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'  {" ".join(command)}\n  -> {log.relative_to(ROOT)}', flush=True)

    env = macro.os.environ.copy()
    env['LUNARBENCH_WORKSPACE'] = str(ROOT)
    started = time.monotonic()
    with log.open('w') as stream:
        code = subprocess.run(command, cwd=ROOT, env=env,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    text = log.read_text(errors='replace')
    print(f'  exit={code} elapsed={time.monotonic() - started:.0f}s')
    for line in text.splitlines():
        if 'MACRO_' in line:
            print('  ' + line.split('LogMoonTerrainImport: ')[-1])

    if macro.SUCCESS_MARKER not in text:
        print(f'  FAILED: 日志里没有 {macro.SUCCESS_MARKER}', file=sys.stderr)
        return None
    return dst_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, default=macro.DEFAULT_ENGINE)
    parser.add_argument('--project', type=Path, default=macro.DEFAULT_PROJECT)
    parser.add_argument('--src-map', default=macro.DEFAULT_SRC_MAP)
    parser.add_argument('--patch', action='append', choices=PATCHES,
                        help='只做指定裁片；缺省三个都做')
    parser.add_argument('--tile-cm', type=float, default=2000.0,
                        help='世界空间平铺边长（cm）；2000 = 20 m，与裁片窗口同尺度')
    parser.add_argument('--specular', type=float, default=0.35)
    parser.add_argument('--normal-scale', type=float, default=1.0)
    parser.add_argument('--skip-stage', action='store_true',
                        help='跳过贴图暂存，直接用 .local/task1_textures 里已有的')
    parser.add_argument('--refresh-snapshot', action='store_true',
                        help='重写源资产 sha256 基线；源资产确实被有意改动过时才用')
    args = parser.parse_args()

    for path in (args.engine / 'Engine/Binaries/Linux/UnrealEditor-Cmd', args.project):
        if not path.exists():
            parser.error(f'Not found: {path}')
    if macro.editor_running(args.project):
        parser.error('Close this UE project before generating maps from it')
    if shutil.which(CONVERT) is None:
        parser.error(f'{CONVERT} not found; needed to write EXR')

    # 快照记的是「第一次换皮之前」的状态，所以已经存在就不覆盖——否则重复跑
    # 会把上次跑完的状态当成基线，源资产真被改了反而检测不出来。
    snapshot = ROOT / '.local/task1_maps_pre_snapshot.txt'
    if snapshot.exists() and not args.refresh_snapshot:
        print(f'Kept existing snapshot: {snapshot}')
    else:
        macro.write_snapshot(args.project, snapshot)

    patches = args.patch or list(PATCHES)
    built = {}
    for patch in patches:
        map_path = build_one(args, patch)
        if map_path is None:
            print(f'\n{patch} 失败，已中止；先前成功的地图保持不动', file=sys.stderr)
            return 1
        built[patch] = map_path

    print('\n换皮完成。跑 TASK1 时按裁片选地图：')
    for patch, map_path in built.items():
        print(f'  {patch:<10} LUNARBENCH_UE_MAP={map_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
