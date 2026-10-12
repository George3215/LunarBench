"""Bit-exact float32 depth archiving with byte shuffle and content deduplication.
Usage: load_depth(original_npy_path), whether original .npy or .npy.json exists.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np


def npy_bytes(array):
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False);return stream.getvalue()


def decode_archive(path):
    with np.load(path,allow_pickle=False) as value:
        dtype=np.dtype(str(value['dtype'].item()));shape=tuple(value['shape'].tolist())
        raw=value['byte_planes'].T.copy().reshape(-1)
        return raw.view(dtype).reshape(shape)


def load_depth(path):
    path=Path(path)
    if path.exists():return np.load(path,allow_pickle=False)
    pointer=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    return decode_archive((path.parent/pointer['archive_relative_path']).resolve())


def compact(root):
    archive=root/'depth_archive';archive.mkdir(exist_ok=True)
    count=0;saved=0
    folders=[transition.parent for mode in ('rgb','rgbd') for transition in (root/mode/'trials').glob('*/steps/*/transition.json')]
    # observations.json is committed only after depth and camera files close.
    for manifest in root.glob('*/*/trials/*/observations.json'):
        folders.extend(Path(frame['rgb_path']).parent for frame in json.loads(manifest.read_text()))
    for folder in folders:
            # A committed transition guarantees these three depths are closed.
            for path in (folder/'depth.npy',folder/'input_depth_m.npy',folder/'before_evaluator/depth.npy',folder/'after_evaluator/depth.npy'):
                if not path.exists():continue
                source_bytes=path.read_bytes();digest=hashlib.sha256(source_bytes).hexdigest()
                target=archive/(digest+'.npz')
                array=np.load(io.BytesIO(source_bytes),allow_pickle=False)
                if not target.exists():
                    temporary=target.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream,byte_planes=array.view(np.uint8).reshape(-1,array.itemsize).T.copy(),
                            shape=np.array(array.shape),dtype=np.array(array.dtype.str))
                    restored=decode_archive(temporary)
                    if not np.array_equal(array.view(np.uint8),restored.view(np.uint8)) or hashlib.sha256(npy_bytes(restored)).hexdigest()!=digest:
                        raise RuntimeError(f'Depth archive failed bit-exact verification: {path}')
                    temporary.replace(target)
                pointer={'format':'byte-shuffled lossless depth archive v1','original_npy_sha256':digest,
                    'original_bytes':len(source_bytes),'shape':list(array.shape),'dtype':array.dtype.str,
                    'archive_relative_path':os.path.relpath(target,path.parent),
                    'archive_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                    'restoration':'depth_archive.load_depth(original_npy_path); restored np.save bytes match original sha256'}
                pointer_path=path.with_suffix(path.suffix+'.json');pointer_path.write_text(json.dumps(pointer,indent=2))
                saved+=len(source_bytes);count+=1;path.unlink()
    return {'newly_archived_depth_files':count,'removed_duplicate_raw_bytes':saved}
