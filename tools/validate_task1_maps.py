"""校验 tools/build_task1_maps.py 的三张换皮地图。

跟着仓库 validate_* 的惯例。只读，不改任何东西。

没有走 commandlet 自带的 -ValidateMacroMap：那个分支在 MoonMacro.inl 里硬断言
贴图导入尺寸是 4096（MoonMacro.inl:489 起），而 TASK1 的裁片源 DEM 只有 2001 px，
烘到 4096 纯属上采样，没意义。所以这里查文件系统与烘焙产物，
再加上「地图必须比它的贴图新」这条——改了贴图忘了重跑换皮是最容易犯的错。

    python tools/validate_task1_maps.py
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import build_moon_macro_map as macro                       # noqa: E402
import build_task1_maps as build                          # noqa: E402

TERRAIN_ROOT = ROOT / 'assets/environments/lunar/terrain'
BAKING_DIR = TERRAIN_ROOT / 'terrain_baking'
SNAPSHOT = ROOT / '.local/task1_maps_pre_snapshot.txt'
TEXTURE_ASSETS = ('T_MoonMacro01_A', 'T_MoonMacro01_N', 'T_MoonMacro01_R')


class Report:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, ok, message):
        if ok:
            self.passes += 1
            print(f'  ok    {message}')
        else:
            self.failures.append(message)
            print(f'  FAIL  {message}')
        return ok


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources(report, project):
    """源资产只读：当前 sha256 必须与换皮前的快照一致。"""
    if not SNAPSHOT.is_file():
        print(f'  skip  源资产基线：{SNAPSHOT.relative_to(ROOT)} 不存在')
        return
    content = project.parent / 'Content'
    for line in SNAPSHOT.read_text().splitlines():
        if not line.strip():
            continue
        expected, _, relative = line.split(None, 2)
        asset = content / relative
        if not asset.is_file():
            report.check(False, f'源资产仍在：{relative}')
            continue
        report.check(sha256(asset) == expected, f'源资产未被改写：{relative}')


def check_baked(report, patch):
    """烘焙产物那一关交给 terrain_baking 自己的校验器，不重复实现。"""
    config = TERRAIN_ROOT / f'task1_{patch}' / 'terrain.yaml'
    done = subprocess.run([sys.executable, 'validate_terrain_bake.py', '--config', str(config)],
                          cwd=BAKING_DIR, capture_output=True, text=True)
    tail = [line for line in done.stdout.splitlines() if '项通过' in line]
    report.check(done.returncode == 0,
                 f'{patch}: 烘焙产物校验通过（{tail[-1] if tail else done.stdout.strip()[-80:]}）')
    return config


def map_texture_mtimes(report, patch, asset_dir):
    """地图必须比它用的贴图新，否则说明改了贴图没重跑换皮。"""
    name = f'task1_{patch}'
    baked = TERRAIN_ROOT / name / 'textures' / f'{name}_diff_4k.png'
    if not baked.is_file() or not asset_dir.is_file():
        return
    report.check(asset_dir.stat().st_mtime >= baked.stat().st_mtime,
                 f'{patch}: 地图不旧于烘焙贴图（改了贴图要重跑 build_task1_maps.py）')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=macro.DEFAULT_PROJECT)
    parser.add_argument('--patch', action='append', choices=build.PATCHES,
                        help='只校验指定裁片；缺省三个都校验')
    args = parser.parse_args()

    if not args.project.exists():
        parser.error(f'Not found: {args.project}')
    content = args.project.parent / 'Content'

    patches = args.patch or list(build.PATCHES)
    print(f'[validate] TASK1 地形地图  {args.project.name}')
    report = Report()

    check_sources(report, args.project)

    destinations = {}
    for patch in patches:
        prefix = build.asset_prefix(patch)
        print(f'\n-- {patch}')
        check_baked(report, patch)

        map_asset = content / f'MoonMacro/Maps/MoonTerrain_{prefix}.umap'
        exists = report.check(map_asset.is_file(), f'地图资产存在：MoonTerrain_{prefix}.umap')
        if exists:
            map_texture_mtimes(report, patch, map_asset)

        for suffix in ('M', 'MI'):
            material = content / f'MoonMacro/Materials/{suffix}_MoonMacro_{prefix}.uasset'
            report.check(material.is_file(), f'材质资产存在：{suffix}_MoonMacro_{prefix}.uasset')

        texture_dest = f'MoonMacro/Textures_{prefix}'
        for asset in TEXTURE_ASSETS:
            path = content / texture_dest / f'{asset}.uasset'
            report.check(path.is_file(), f'纹理资产存在：Textures_{prefix}/{asset}.uasset')
        destinations[patch] = texture_dest

    # 撞车会让 ImportTexture 静默复用上一张图（MoonMacro.inl:68），值得单独断言。
    if len(patches) > 1:
        report.check(len(set(destinations.values())) == len(destinations),
                     f'三个裁片的纹理目录互不相同 {sorted(set(destinations.values()))}')

    print(f'\n{report.passes} 项通过，{len(report.failures)} 项失败')
    for message in report.failures:
        print(f'  - {message}')
    return 1 if report.failures else 0


if __name__ == '__main__':
    sys.exit(main())
