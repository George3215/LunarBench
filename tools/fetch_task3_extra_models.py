"""Download user-requested spatial checkpoints at inventoried immutable revisions."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True, choices=['robobrain25_8b_nv', 'spatialreasoner', 'spatialrgpt'])
    p.add_argument('--record', type=Path, default=Path('paper/vlm_goal_navigation/spatial_candidates_20261018'))
    p.add_argument('--source', choices=['huggingface', 'modelscope'], default='huggingface')
    a = p.parse_args()
    meta = json.loads((a.record / (a.key + '_inventory.json')).read_text())
    output = a.record / (a.key + '_download.json')
    folder = Path('.local/models') / a.key
    state = dict(repo_id=meta['repo_id'], revision=meta['revision'], status='downloading',
                 started_unix=time.time(), local_path=str(folder.resolve()))
    output.write_text(json.dumps(state, indent=2))
    try:
        if a.source == 'modelscope':
            import requests
            inventory = json.loads((a.record / 'robobrain_modelscope_inventory.json').read_text())
            entries = [f for f in inventory['Data']['Files'] if f['Path'].endswith(('.json', '.txt', '.jinja', '.safetensors'))]
            hf_files = {f['name']: f for f in meta['files']}
            for entry in entries:
                if entry['Path'].endswith('.safetensors'):
                    assert entry['Sha256'] == hf_files[entry['Path']]['lfs']['sha256'], 'official mirrors differ'
            state.update(source='official ModelScope linked from FlagOpen/RoboBrain2.5',
                         modelscope_files=[dict(path=f['Path'], revision=f['Revision'], sha256=f['Sha256']) for f in entries])
            output.write_text(json.dumps(state, indent=2))
            def fetch(entry):
                path = folder / entry['Path']
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size == entry['Size']:
                    if hashlib.sha256(path.read_bytes()).hexdigest() == entry['Sha256']:
                        return
                temp = path.with_suffix(path.suffix + '.partial')
                url = 'https://modelscope.cn/api/v1/models/' + meta['repo_id'] + '/repo'
                for attempt in range(4):
                    start = temp.stat().st_size if temp.exists() else 0
                    try:
                        with requests.get(url, params={'Revision':entry['Revision'], 'FilePath':entry['Path']},
                                          headers={'Range':f'bytes={start}-'} if start else {},
                                          stream=True, timeout=(30, 120)) as response:
                            response.raise_for_status()
                            mode = 'ab' if start and response.status_code == 206 else 'wb'
                            with temp.open(mode) as f:
                                for chunk in response.iter_content(8 * 1024**2):
                                    f.write(chunk)
                        if temp.stat().st_size != entry['Size']:
                            raise RuntimeError('incomplete download ' + entry['Path'])
                        digest = hashlib.sha256()
                        with temp.open('rb') as f:
                            for chunk in iter(lambda:f.read(8*1024**2),b''):
                                digest.update(chunk)
                        if digest.hexdigest() != entry['Sha256']:
                            raise RuntimeError('ModelScope SHA256 mismatch ' + entry['Path'])
                        temp.replace(path)
                        print('verified',entry['Path'],flush=True)
                        return
                    except Exception:
                        if attempt == 3:
                            raise
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(fetch, entries))
        else:
            snapshot_download(meta['repo_id'], revision=meta['revision'], local_dir=folder,
                              allow_patterns=['*.json', '*.txt', '*.jinja', '*.safetensors'],
                              ignore_patterns=['trainer_state.json'], max_workers=3)
        checks = []
        for entry in meta['files']:
            path = folder / entry['name']
            if not path.is_file() or not entry.get('lfs'):
                continue
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                    digest.update(chunk)
            expected = entry['lfs']['sha256']
            actual = digest.hexdigest()
            checks.append(dict(file=entry['name'], sha256=actual, upstream_sha256=expected,
                               matched=actual == expected, bytes=path.stat().st_size))
            if actual != expected:
                raise RuntimeError('download hash mismatch: ' + entry['name'])
        if not any(c['file'].endswith('.safetensors') for c in checks):
            raise RuntimeError('no complete checkpoint tensors verified')
        state.update(status='complete', verified_lfs_files=checks, elapsed_s=time.time()-state['started_unix'])
    except Exception as e:
        state.update(status='blocked_or_error', error_type=type(e).__name__,
                     status_code=getattr(getattr(e, 'response', None), 'status_code', None))
        raise
    finally:
        output.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
