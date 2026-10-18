"""Reuse frozen TASK3 physics/prompts; isolate architecture-specific inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tasks/task3_reobservability'))
import spatial_transfer_v4 as base
from action_grammar import ActionVocabulary
from model_output_adapter import initial_plan


class NativeQwenPolicy(base.Policy):
    def __init__(self, model_path, processor_path, max_tokens):
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import torch
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        if config.model_type == 'qwen2_5_vl':
            # Upstream preprocessor uses the old Qwen2_5_VLImageProcessor alias.
            # Transformers 5 requires the Qwen2 shared processor and a video
            # processor even for image-only calls. Preserve the checkpoint's
            # image configuration and chat template; no video input is used.
            from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(processor_path), local_files_only=True)
            self.processor = Qwen2_5_VLProcessor(
                image_processor=Qwen2VLImageProcessor.from_pretrained(str(processor_path), local_files_only=True),
                video_processor=Qwen2VLVideoProcessor(), tokenizer=tokenizer, chat_template=tokenizer.chat_template)
        else:
            self.processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': 65536}
        self.processor.image_processor.min_pixels = 65536
        self.processor.image_processor.max_pixels = 65536
        cls = Qwen2_5_VLForConditionalGeneration if config.model_type == 'qwen2_5_vl' else base.Qwen3VLForConditionalGeneration
        self.model = cls.from_pretrained(str(model_path), dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa').to('cuda:0').eval()
        self.model.generation_config.use_cache = True
        config = getattr(self.model.config, 'text_config', self.model.config)
        config.use_cache = True
        self.model_context_limit = int(config.max_position_embeddings)
        self.actor_max_tokens = max_tokens
        self.generation_calls = 0
        self.action_vocabulary = ActionVocabulary(self.processor.tokenizer)
        self.action_grammars = {}


class SpatialRGPTPolicy(base.Policy):
    def __init__(self, model_path, processor_path, max_tokens):
        self.actor_max_tokens = max_tokens
        self.generation_calls = 0
        self.log = (OUT / 'worker.log').open('a')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TRANSFORMERS_CACHE=str(REPO / '.local/cache/spatialrgpt'))
        self.worker = subprocess.Popen([str(REPO / '.local/envs/spatialrgpt/bin/python'),
            '-u', str(REPO / 'tools/task3_spatialrgpt_worker.py'), '--model', str(model_path)],
            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        ready = self.worker.stdout.readline()
        if not ready or not json.loads(ready).get('ready'):
            raise RuntimeError('SpatialRGPT worker failed to load; see worker.log')

    def decide_batch(self, requests):
        records = []
        for request, folder, attempt in requests:
            folder.mkdir(parents=True, exist_ok=True)
            _, messages = self.messages(request, attempt)
            base.dump(folder / 'messages.json', {'messages': messages, 'mode': request['mode'],
                'horizon': request['horizon'], 'policy_inputs': request['input_description']})
            payload = dict(system=base.cp.SYSTEM, text=messages[1]['content'][0]['text'],
                images=[(label, str(path)) for label, path in request['images']],
                stop_allowed=request.get('stop_allowed'), max_new_tokens=base.cp.OUTPUT_TOKEN_LIMIT)
            self.worker.stdin.write(json.dumps(payload) + '\n')
            self.worker.stdin.flush()
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError('SpatialRGPT worker exited; see worker.log')
            response = json.loads(line)
            if 'error' in response:
                raise RuntimeError('SpatialRGPT inference failed: ' + response['error'])
            self.generation_calls += 1
            network = response.pop('network_input')
            prompt = response.pop('rendered_prompt')
            context = dict(horizon=request['horizon'], mode=request['mode'],
                input_frame_ids=request['frame_ids'], input_frame_count=len(request['frame_ids']),
                input_image_count=len(request['images']), input_tokens=network['expanded_input_tokens'],
                all_recorded_frame_count=request['all_recorded_frame_count'],
                first_frame_in_context=request['first_frame_in_context'],
                history_span_sim_s=request['history_span_sim_s'], truncated=False,
                model_max_position_embeddings=8192, runtime_input_token_limit=16384,
                output_token_limit=base.cp.OUTPUT_TOKEN_LIMIT,
                past_assistant_outputs_in_context=False, past_action_history_in_context=False)
            network['context'] = context
            base.dump(folder / 'network_input.json', network)
            base.dump(folder / 'context.json', context)
            (folder / 'rendered_prompt.txt').write_text(prompt)
            record = dict(response, generation_call_id=self.generation_calls, batch_size=1,
                          batch_amortized_s=response['elapsed_s'])
            record['parsed'] = self.parse_output(record['raw_output']) if not record.get('termination') else None
            parsed = record['parsed']
            if parsed:
                record['unsupported_evidence_frame_ids'] = [f for f in parsed['evidence_frames'] if f not in request['frame_ids']]
                record['unsupported_depth_claim'] = False
                record['unsupported_past_claim'] = len(request['frame_ids']) == 1 and any(s.startswith('past_') for s in parsed['evidence_sources'])
                if parsed['action'] == 'STOP' and not request.get('stop_allowed'):
                    record.update(rejected_stop=True, parsed_before_stop_gate=parsed, parsed=None)
            base.dump(folder / 'output.json', record)
            (folder / 'raw_output.txt').write_text(record['raw_output'])
            records.append(record)
        return records

    def plan_batch(self, trials, max_tokens):
        base.cp.SYSTEM = base.PLAN_SYSTEM
        base.cp.OUTPUT_TOKEN_LIMIT = max_tokens
        requests = []
        for trial in trials:
            request = base.cp.request_for(trial.frames[:1], 'current', 'rgbd')
            request['text'] += '\nPlan from this initial observation only; no later frame is available.'
            requests.append((request, trial.folder / 'initial_plan', 1))
        for trial, (_, folder, _), record in zip(trials, requests, self.decide_batch(requests)):
            plan, fmt = initial_plan(record['raw_output'])
            trial.plan = plan
            trial.plan_valid = plan is not None
            record.update(plan_parsed=plan, plan_format=fmt)
            base.dump(folder / 'output.json', record)
        base.cp.SYSTEM = base.SYSTEM
        base.cp.OUTPUT_TOKEN_LIMIT = self.actor_max_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=['native_qwen', 'spatialrgpt'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    options, _ = parser.parse_known_args()
    OUT = options.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    manifest = dict(backend=options.backend, adapter_sha256=hashlib.sha256(source).hexdigest(),
        comparison_scope='cross-backbone transfer, not a single-factor spatial-training ablation',
        prompt_content='frozen v2 SYSTEM and PLAN_SYSTEM; native architecture chat template',
        input_modalities='same event4 RGB-D images, ideal self-odometry and own plan',
        spatialrgpt_variant='general multimodal images; no region proposal or GT mask; depth grayscale is an ordinary image; native region/depth-token variant NOT evaluated',
        batch_execution='native_qwen batches as v2; SpatialRGPT sequential requests, same action budget',
        no_silent_truncation=True)
    base.dump(OUT / 'ADAPTER_PROTOCOL.json', manifest)
    inherited_dump = base.dump
    def adapter_dump(path, value):
        if Path(path).name == 'protocol.json':
            value = dict(value, same_base_processor_for_both_models=False,
                frame_area_cap=65536 if options.backend == 'native_qwen' else None,
                native_visual_preprocessing='Qwen checkpoint processor, area cap 65536' if options.backend == 'native_qwen' else 'official SigLIP 384x384 image resize',
                adapter_protocol='ADAPTER_PROTOCOL.json')
        inherited_dump(path, value)
    base.dump = adapter_dump
    snapshots = OUT / 'adapter_source'
    snapshots.mkdir(exist_ok=True)
    for name in ('task3_candidate_driver.py', 'task3_spatialrgpt_worker.py'):
        file = REPO / 'tools' / name
        if file.exists():
            dest = snapshots / name
            if dest.exists() and dest.read_bytes() != file.read_bytes():
                raise RuntimeError('adapter source drift; use a fresh run')
            dest.write_bytes(file.read_bytes())
    index = sys.argv.index('--backend')
    del sys.argv[index:index+2]
    base.Policy = NativeQwenPolicy if options.backend == 'native_qwen' else SpatialRGPTPolicy
    try:
        base.main()
    finally:
        # Worker receives EOF when the physics driver exits, including errors.
        pass
