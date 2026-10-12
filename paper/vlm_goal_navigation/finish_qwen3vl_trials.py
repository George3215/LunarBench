"""Watch an already-running bounded trial suite and build its final report once."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def dump(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()
    status_file = root/'RUN_STATUS.json'
    # This watcher never starts or retries a trial and never changes its budget.
    # Maximum wait bounds an engine failure; actual action limits live in the runner.
    while time.monotonic()-started < 8*3600:
        aggregate_path = root/'aggregate.json'
        aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
        if aggregate.get('completed') == aggregate.get('expected') == 24:
            dump(status_file, {'status': 'analysing_final_records', 'aggregate': aggregate})
            env = dict(os.environ, MPLCONFIGDIR='/tmp/task3-matplotlib')
            analysis = root/'source_snapshot'/'analyse_qwen3vl_trials.py'
            with (root/'final_analysis.log').open('w') as log:
                result = subprocess.run(['python3', str(analysis), str(root)], env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                dump(status_file, {'status': 'analysis_failed', 'returncode': result.returncode,
                    'aggregate': aggregate, 'details': 'final_analysis.log'})
                raise SystemExit(result.returncode)
            metrics = json.loads((root/'run_metrics.json').read_text())
            valid = metrics['completed'] == 24 and metrics['all_artifact_checks_pass']
            dump(status_file, {'status': 'complete' if valid else 'records_need_review',
                'aggregate': aggregate, 'metrics': metrics,
                'report': 'REPORT.md', 'browser': 'TRIALS.html',
                'note': 'Original target heights retained; raw SR is not a pure VLM capability score. Physical reference results excluded from VLM SR.'})
            return
        log_path = root/'run.log'
        if log_path.exists():
            with log_path.open('rb') as f:
                f.seek(max(0, log_path.stat().st_size-12000))
                tail = f.read().decode(errors='replace')
            if 'Traceback (most recent call last)' in tail:
                dump(status_file, {'status': 'engine_failed', 'aggregate': aggregate,
                    'details': 'run.log; no automatic retries performed'})
                raise SystemExit(1)
        dump(status_file, {'status': 'running', 'aggregate': aggregate,
            'action_limit_per_trial': 3000, 'final_analysis_automatic': True,
            'report_note': 'Current REPORT/TRIALS files are an analysis snapshot; trace/state/image files continue growing.'})
        time.sleep(30)
    dump(status_file, {'status': 'watcher_timeout', 'details': 'Eight-hour monitoring bound reached; inspect run.log. No new trials or budget extensions were started.'})
    raise SystemExit(2)

if __name__ == '__main__':
    main()
