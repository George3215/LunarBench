"""Refresh browser and audit the already running paired experiment; never rerun."""
import argparse
import json
import os
import time
import subprocess
import sys
from pathlib import Path
from depth_archive import compact


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def main(root):
    previous=None
    last_analysis=0.
    root.mkdir(parents=True,exist_ok=True)
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['watch_visible_start_trials.py','analyse_visible_start_trials.py','depth_archive.py','plot_visible_start_trials.py']:
        source=Path(__file__).with_name(name)
        (snapshot/name).write_bytes(source.read_bytes())
    (root/'depth_archive.py').write_bytes(Path(__file__).with_name('depth_archive.py').read_bytes())
    start=time.monotonic()
    while time.monotonic()-start<64*3600:
        error=root/'ERROR.json'
        if error.exists():
            save(root/'FINAL_STATUS.json',{'status':'engine_failed','details':json.loads(error.read_text())});return
        status=root/'RUN_STATUS.json'
        if status.exists():
            value=json.loads(status.read_text())
            process_file=root/'processes.json'
            processes=json.loads(process_file.read_text()) if process_file.exists() else {}
            if processes.get('watcher_pid')!=os.getpid():
                processes['watcher_pid']=os.getpid();save(process_file,processes)
            completed=value['completed']
            compaction=compact(root)
            save(root/'DEPTH_ARCHIVE_STATUS.json',compaction)
            ready=any(any((root/m/'trials').glob('*/approach_gate.json')) for m in ('rgb','rgbd'))
            if ready and (previous is None or completed!=previous or time.monotonic()-last_analysis>=120 or value['status']=='trials_complete_pending_analysis'):
                with (root/'latest_analysis.log').open('w') as log:
                    analysis=subprocess.run([sys.executable,str(snapshot/'analyse_visible_start_trials.py'),str(root)],stdout=log,stderr=subprocess.STDOUT)
                if analysis.returncode:
                    save(root/'ANALYSIS_STATUS.json',{'status':'analysis_failed','returncode':analysis.returncode,'details':'latest_analysis.log'})
                    raise SystemExit(analysis.returncode)
                metrics=json.loads((root/'run_metrics.json').read_text())
                previous=completed
                last_analysis=time.monotonic()
                save(root/'ANALYSIS_STATUS.json',{'status':'snapshot','metrics':metrics})
                if value['status']=='trials_complete_pending_analysis':
                    valid=metrics['completed']==48 and metrics['paired_completed']==24 and metrics['all_artifact_checks_pass'] and metrics['all_available_pair_checks_pass']
                    value['status']='complete' if valid else 'records_need_review';value['metrics']=metrics
                    save(status,value);save(root/'FINAL_STATUS.json',value)
                    return
        time.sleep(30)
    save(root/'FINAL_STATUS.json',{'status':'watcher_timeout','note':'No retries or expanded budgets.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
