"""Periodically archive committed depth frames and update the audited browser."""
import sys, time, json, os, traceback
from pathlib import Path
from analyse_all_terrain_context import analyse
from depth_archive import compact
root=Path(sys.argv[1]).resolve()
while True:
    try:
        if (root/'protocol.json').exists():
            print(json.dumps(compact(root)),flush=True)
            report=analyse(root)
            status=json.loads((root/'RUN_STATUS.json').read_text()) if (root/'RUN_STATUS.json').exists() else {}
            print('ANALYSIS',report['completed'],report['expected'],report['audit_pass'],flush=True)
            if status.get('status')=='trials_complete_pending_analysis':
                status['status']='complete' if report['all_complete'] else 'audit_failed'
                (root/'RUN_STATUS.json').write_text(json.dumps(status,indent=2));break
            process_file=root/'processes.json'
            if process_file.exists():
                engine_pid=json.loads(process_file.read_text())['engine_pid']
                try:os.kill(engine_pid,0)
                except ProcessLookupError:
                    status['status']='interrupted';status['completed']=report['completed']
                    (root/'RUN_STATUS.json').write_text(json.dumps(status,indent=2));break
        if (root/'ERROR.json').exists():break
    except Exception:
        traceback.print_exc()
    time.sleep(45)
