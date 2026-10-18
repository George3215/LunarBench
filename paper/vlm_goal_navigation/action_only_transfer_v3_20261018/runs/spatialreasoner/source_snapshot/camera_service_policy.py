"""Allowlisted camera-only client for local navigation models."""
import base64
import io
import math
import time

import requests
from PIL import Image

from context_navigation_policy import SYSTEM
from vlm_benchmark import dump, sha


class CameraServicePolicy:
    def __init__(self, endpoint, model, protocol='matrix'):
        self.endpoint = endpoint
        self.model = model
        self.protocol = protocol
        self.generation_calls = 0

    def decide_batch(self, items):
        results = []
        for request, folder, attempt in items:
            folder.mkdir(parents=True, exist_ok=True)
            images, labels, manifest = [], [], []
            for label, path in request['images']:
                image = Image.open(path).convert('RGB')
                if self.protocol == 'matrix':
                    scale = min(1., math.sqrt(65536/(image.width*image.height)))
                    image = image.resize((int(image.width*scale), int(image.height*scale)), Image.Resampling.BICUBIC)
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                images.append(base64.b64encode(buffer.getvalue()).decode())
                labels.append(label)
                manifest.append({'label': label, 'sha256': sha(path), 'size': list(image.size)})
            payload = {'images': images, 'labels': labels, 'text': request['text'],
                       'system': SYSTEM, 'protocol': self.protocol}
            dump(folder/'network_input.json', payload)
            dump(folder/'messages.json', {'system': SYSTEM, 'text': request['text'], 'images': manifest})
            start = time.perf_counter()
            response = requests.post(self.endpoint+'/predict', json=payload, timeout=600)
            data = response.json()
            dump(folder/'response.json', data)
            if not response.ok or data.get('error'):
                raise RuntimeError(f"Model inference failed; no navigation failure inferred: {data.get('error')}")
            if data.get('model') != self.model:
                raise RuntimeError('Local service model identity mismatch')
            dump(folder/'context.json', {
                'mode': request['mode'], 'horizon': request['horizon'], 'protocol': self.protocol,
                'input_frame_ids': request['frame_ids'], 'input_frame_count': len(request['frame_ids']),
                'input_image_count': len(images), 'input_tokens': data['input_tokens'],
                'context_limit':data['context_limit'],
                'architecture_context_limit':data.get('architecture_context_limit'),
                'configured_context_limit':data.get('configured_context_limit'),
                'all_recorded_frame_count': request['all_recorded_frame_count'],
                'actual_model_input_labels': data['actual_input_labels'],
                'actual_model_image_count': data['input_image_count'],
                'truncated': False, 'past_assistant_outputs_in_context': False,
                'past_action_history_in_context': False})
            raw = data['raw_output']
            record = {'raw_output': raw, 'parsed': {'action': data['action']} if data.get('action') else None,
                      'elapsed_s': time.perf_counter()-start, 'batch_amortized_s': time.perf_counter()-start,
                      'generation_skipped': data.get('termination') == 'context_limit',
                      'generated_token_count': data['generated_token_count'], 'model': self.model,
                      'termination': data.get('termination'), 'hit_token_limit': data.get('hit_token_limit', False)}
            self.generation_calls += int(not record['generation_skipped'])
            dump(folder/'output.json', record)
            (folder/'raw_output.txt').write_text(raw)
            results.append(record)
        return results
