"""Constrain only the VLM's JSON action value to the sensor-legal action set.

No controller chooses an alternative action: legal token logits remain unchanged.
The rest of the generated assessment, rationale and plan remains unconstrained.
"""
import re

import torch
from transformers import LogitsProcessor

MOTION = ("FORWARD", "BACKWARD", "STRAFE_LEFT", "STRAFE_RIGHT", "TURN_LEFT", "TURN_RIGHT")
ACTION_TAIL = re.compile(r'"action"\s*:\s*("?[^"\\]*)\Z')
LABEL_TAIL = re.compile(r'^\s*action\s*:\s*([A-Z_]*)\Z', re.MULTILINE | re.IGNORECASE)


def valid_extension(partial, extension, actions, quoted=True):
    if not extension:
        return False
    value = (partial + extension).lstrip()
    if not value:
        return True  # JSON whitespace before the opening quote
    if not quoted:
        return any(action.startswith(value) or
                   (value.startswith(action) and len(value) > len(action) and value[len(action)].isspace())
                   for action in actions)
    if not value.startswith('"'):
        return False
    value = value[1:]
    return any(action.startswith(value) or value.startswith(action + '"') for action in actions)


class ActionVocabulary:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text = tokenizer.batch_decode([[i] for i in range(len(tokenizer))], skip_special_tokens=False)
        # For this frozen tokenizer the key/colon cannot be bypassed together
        # with a complete STOP in one token before ACTION_TAIL sees the field.
        if any(":" in text and "STOP" in text for text in self.text):
            raise RuntimeError("tokenizer requires a full JSON lexer for action-field entry")
        self.cache = {}

    def allowed(self, partial, stop_available, quoted=True):
        partial = partial.lstrip()
        key = (partial, stop_available, quoted)
        if key not in self.cache:
            actions = MOTION + (("STOP",) if stop_available else ())
            self.cache[key] = [i for i, text in enumerate(self.text) if valid_extension(partial, text, actions, quoted)]
        if not self.cache[key]:
            raise RuntimeError(f"empty legal-action grammar for prefix {partial!r}")
        return self.cache[key]


class ActionGrammar(LogitsProcessor):
    def __init__(self, vocabulary, stop_available, prompt_length):
        self.vocabulary = vocabulary
        self.stop_available = stop_available
        self.prompt_length = prompt_length
        self.masked_argmax = [0] * len(stop_available)
        self.stop_path_interceptions = [0] * len(stop_available)

    def __call__(self, input_ids, scores):
        if len(scores) != len(self.stop_available):
            raise RuntimeError("action grammar requires independent greedy sequences")
        for row, stop in enumerate(self.stop_available):
            text = self.vocabulary.tokenizer.decode(input_ids[row, self.prompt_length:].tolist(), skip_special_tokens=False)
            match = ACTION_TAIL.search(text)
            quoted = True
            if match is None:
                match = LABEL_TAIL.search(text)
                quoted = False
            if match is None:
                continue
            allowed = self.vocabulary.allowed(match.group(1), stop, quoted)
            top = int(scores[row].argmax())
            if top not in allowed:
                self.masked_argmax[row] += 1
                token = self.vocabulary.text[top] if top < len(self.vocabulary.text) else ""
                value = (match.group(1) + token).lstrip().removeprefix('"')
                if not stop and ("STOP".startswith(value) or value.startswith('STOP"')):
                    self.stop_path_interceptions[row] += 1
            # Mask illegal values; preserve all relative probabilities of legal
            # choices. No progress heuristic, target direction or fallback here.
            mask = torch.ones_like(scores[row], dtype=torch.bool)
            mask[allowed] = False
            scores[row].masked_fill_(mask, -torch.inf)
        return scores
