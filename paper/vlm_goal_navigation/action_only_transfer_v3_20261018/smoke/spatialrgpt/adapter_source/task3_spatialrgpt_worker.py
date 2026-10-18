"""Isolated official SpatialRGPT inference; JSON lines over stdin/stdout."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / '.local/vendor/SpatialRGPT'))
sys.path.insert(0, str(REPO / 'tasks/task3_reobservability'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    wire = sys.stdout
    # Official modules print load diagnostics; keep the RPC stream machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from PIL import Image
        from transformers import LogitsProcessorList
        from llava.model.language_model.llava_llama import LlavaLlamaModel, LlavaLlamaConfig
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from action_grammar import ActionGrammar, ActionVocabulary
        from task3_action_only import ClosedSetActionGrammar

        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = LlavaLlamaConfig.from_pretrained(str(args.model), local_files_only=True)
        config._name_or_path = str(args.model.resolve())
        config.resume_path = str(args.model.resolve())
        config.model_dtype = 'torch.bfloat16'
        model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, attn_implementation='sdpa',
                              model_max_length=8192).to('cuda:0').eval()
        # No hidden clipping of the multimodal sequence; check the actual size below.
        model.llm.config.tokenizer_model_max_length = None
        tokenizer = model.tokenizer
        vocabulary = ActionVocabulary(tokenizer)
        processor = model.get_vision_tower().image_processor
        stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        print(json.dumps({'ready': True, 'context_limit': 8192}), file=wire, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                conv = conv_templates['llama_3'].copy()
                conv.system = '<|start_header_id|>system<|end_header_id|>\n\n' + request['system']
                user = request['text']
                images = []
                for label, path in request['images']:
                    user += '\n' + label + '\n<image>\n'
                    images.append(Image.open(path).convert('RGB'))
                conv.append_message(conv.roles[0], user)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
                pixels = processor.preprocess(images, return_tensors='pt')['pixel_values']
                image_records = [{'label': label, 'image_path': path, 'image_sha256': sha(path),
                    'pixel_values_shape': list(pixels[i].shape),
                    'pixel_values_float32_sha256': hashlib.sha256(pixels[i].float().numpy().tobytes()).hexdigest()}
                    for i, (label, path) in enumerate(request['images'])]
                begin = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    _, _, attention, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
                        ids, None, torch.ones_like(ids, dtype=torch.bool), None, None,
                        pixels.to('cuda:0', dtype=torch.bfloat16), None, None)
                    length = embeds.shape[1]
                    network = dict(input_ids=ids[0].tolist(), attention_mask=[1]*ids.shape[1],
                        images=image_records, expanded_input_tokens=length,
                        packed_embeddings_shape=list(embeds.shape),
                        packed_embeddings_bf16_sha256=hashlib.sha256(embeds.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        batch_size=1, independent_sequence=0)
                    result = dict(raw_output='', raw_output_with_special_tokens='', generated_token_ids=[],
                        generated_token_count=0, hit_token_limit=False, generation_skipped=False,
                        rendered_prompt=prompt, network_input=network)
                    if length + request['max_new_tokens'] > 8192 or length > 16384:
                        result.update(termination='context_limit', generation_skipped=True)
                    else:
                        grammar = None
                        if request['stop_allowed'] is not None:
                            if request.get('action_interface') == 'action_only':
                                # Transformers 4.37 inputs_embeds decoding starts
                                # input_ids with one BOS; it is not an action token.
                                grammar = ClosedSetActionGrammar(tokenizer, [request['stop_allowed']], 1, stop_ids)
                            else:
                                grammar = ActionGrammar(vocabulary, [request['stop_allowed']], 0)
                        kwargs = dict(logits_processor=LogitsProcessorList([grammar])) if grammar else {}
                        outputs = model.llm.generate(inputs_embeds=embeds, attention_mask=attention,
                            max_new_tokens=request['max_new_tokens'], do_sample=False, use_cache=True,
                            eos_token_id=stop_ids, pad_token_id=tokenizer.eos_token_id, **kwargs)
                        generated = outputs[0].tolist()
                        result.update(raw_output=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            raw_output_with_special_tokens=tokenizer.decode(generated, skip_special_tokens=False),
                            generated_token_ids=generated, generated_token_count=len(generated),
                            hit_token_limit=len(generated) >= request['max_new_tokens'])
                        if grammar:
                            result['action_grammar'] = dict(masked_argmax_tokens=grammar.masked_argmax[0],
                                forbidden_stop_path_interceptions=grammar.stop_path_interceptions[0])
                torch.cuda.synchronize()
                result.update(elapsed_s=time.perf_counter()-begin,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                print(json.dumps(result), file=wire, flush=True)
                del embeds, ids, pixels
            except Exception as exc:
                traceback.print_exc()
                print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}), file=wire, flush=True)


if __name__ == '__main__':
    main()
