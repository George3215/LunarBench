"""Stateless, tools-disabled Responses policy. Only camera payloads cross the API."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import json
import math
import os
import time
import tomllib
from pathlib import Path

import requests
from PIL import Image

from context_navigation_policy import SYSTEM, parse
from vlm_benchmark import dump, sha

MODEL = 'gpt-6.1-sol'
AREA = 65536


class Sol61Policy:
    def __init__(self, workers=6):
        home = Path(os.environ.get('CODEX_HOME', '/home/lry/.codex'))
        config = tomllib.loads((home / 'config.toml').read_text())
        provider = config['model_providers'][config['model_provider']]
        self.endpoint = provider['base_url'].rstrip('/') + '/responses'
        auth = json.loads((home / 'auth.json').read_text())
        self.key = auth.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not self.key:
            raise RuntimeError('Configured API credential unavailable')
        self.workers = workers
        self.generation_calls = 0

    def payload(self, request):
        # Never serialize the full request, frame record, path, or evaluator.
        text = request['text']
        content = [{'type': 'input_text', 'text': text}]
        saved = [{'type': 'input_text', 'text': text}]
        for label, path in request['images']:
            if request['mode'] == 'rgb' and 'depth' in label.lower():
                raise ValueError('Depth forbidden in RGB condition')
            image = Image.open(path).convert('RGB')
            scale = min(1., math.sqrt(AREA / (image.width * image.height)))
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            blob = buffer.getvalue()
            content.extend([{'type': 'input_text', 'text': label},
                            {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(blob).decode(), 'detail': 'high'}])
            saved.extend([{'type': 'input_text', 'text': label},
                          {'type': 'input_image', 'sha256': hashlib.sha256(blob).hexdigest(),
                           'source_sha256': sha(Path(path)), 'size': list(image.size)}])
        payload = {'model': MODEL, 'instructions': SYSTEM,
                   'input': [{'role': 'user', 'content': content}],
                   'reasoning': {'effort': 'medium'}, 'max_output_tokens': 4096,
                   'tools': [], 'tool_choice': 'none', 'store': False,
                   'truncation': 'disabled'}
        saved_payload = {**payload, 'input': [{'role': 'user', 'content': saved}]}
        return payload, saved_payload

    def decide(self, item):
        request, folder, attempt = item
        folder.mkdir(parents=True, exist_ok=True)
        payload, saved = self.payload(request)
        # Exact wire payload contains camera bytes only, no credentials or file paths.
        dump(folder / 'network_input.json', payload)
        dump(folder / 'messages.json', saved)
        dump(folder / 'context.json', {
            'mode': request['mode'], 'horizon': request['horizon'],
            'input_frame_ids': request['frame_ids'], 'input_frame_count': len(request['frame_ids']),
            'input_image_count': len(request['images']), 'input_tokens': None,
            'all_recorded_frame_count': request['all_recorded_frame_count'],
            'past_assistant_outputs_in_context': False, 'past_action_history_in_context': False,
            'tools_enabled': False, 'truncated': False, 'attempt': attempt})
        start = time.perf_counter()
        response = requests.post(self.endpoint, headers={'Authorization': 'Bearer ' + self.key},
                                 json=payload, timeout=(20, 240))
        if not response.ok:
            dump(folder / 'service_error.json', {'http_status': response.status_code})
            raise RuntimeError(f'Model service HTTP {response.status_code}; no navigation result inferred')
        data = response.json()
        dump(folder / 'response.json', data)
        if data.get('model') != MODEL:
            raise RuntimeError(f'Model identity mismatch: {data.get("model")}')
        if any(o.get('type') != 'message' and o.get('type') != 'reasoning' for o in data.get('output', [])):
            raise RuntimeError('Unexpected tool output: trial halted')
        raw = '\n'.join(c.get('text', '') for o in data.get('output', [])
                        if o.get('type') == 'message' for c in o.get('content', []) if c.get('type') == 'output_text')
        parsed = parse(raw)
        unsupported = []
        if parsed:
            unsupported = [f for f in parsed['evidence_frames'] if f not in request['frame_ids']]
            sources = parsed['evidence_sources']
            if request['mode'] == 'rgb' and any('depth' in s for s in sources):
                unsupported.append('depth_in_rgb')
            if len(request['frame_ids']) == 1 and any(s.startswith('past_') for s in sources):
                unsupported.append('past_in_current')
            if unsupported:
                parsed = None
        usage = data.get('usage', {})
        context = json.loads((folder / 'context.json').read_text())
        context['input_tokens'] = usage.get('input_tokens')
        dump(folder / 'context.json', context)
        record = {'raw_output': raw, 'parsed': parsed, 'unsupported_evidence': unsupported,
                  'elapsed_s': time.perf_counter() - start, 'batch_amortized_s': time.perf_counter() - start,
                  'generation_skipped': False, 'model': data.get('model'), 'response_id': data.get('id'),
                  'usage': usage, 'hit_token_limit': data.get('status') == 'incomplete'}
        if data.get('status') != 'completed':
            record['termination'] = 'model_response_incomplete'
        dump(folder / 'output.json', record)
        (folder / 'raw_output.txt').write_text(raw)
        return record

    def decide_batch(self, items):
        self.generation_calls += len(items)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.decide, items))
