"""Closed-set action decoding; no planner, inferred action, or invented assessment."""
import torch
from transformers import LogitsProcessor

MOTION = ('FORWARD', 'BACKWARD', 'STRAFE_LEFT', 'STRAFE_RIGHT', 'TURN_LEFT', 'TURN_RIGHT')
ACTIONS = MOTION + ('STOP',)


def parse_action(raw):
    action = raw.strip()
    if action not in ACTIONS:
        return None
    # Empty evidence lists mean NOT REQUESTED, never a claim about observed frames.
    # They only satisfy the inherited input logger's list interface.
    return dict(action=action, evidence_frames=[], evidence_sources=[],
                auxiliary_assessments_not_requested=True)


class ClosedSetActionGrammar(LogitsProcessor):
    def __init__(self, tokenizer, stop_available, prompt_length, eos_token_ids):
        self.stop_available = stop_available
        self.prompt_length = prompt_length
        self.eos = tuple(eos_token_ids if isinstance(eos_token_ids, (list, tuple)) else [eos_token_ids])
        self.eos = tuple(x for x in self.eos if x is not None)
        if not self.eos:
            raise ValueError('A valid EOS token is required')
        self.paths = {action: tuple(tokenizer.encode(action, add_special_tokens=False)) for action in ACTIONS}
        for action, tokens in self.paths.items():
            if not tokens or tokenizer.decode(tokens, skip_special_tokens=True).strip() != action:
                raise ValueError(f'Action tokenizer round-trip failed: {action}')
        self.masked_argmax = [0] * len(stop_available)
        # Old stop-field interception counts do not apply to this whole-answer trie.
        self.stop_path_interceptions = [0] * len(stop_available)

    def allowed(self, generated, stop_available):
        if any(token in self.eos for token in generated):
            return self.eos  # another batch row may still be generating
        legal = ACTIONS if stop_available else MOTION
        allowed = set()
        for action in legal:
            path = self.paths[action]
            if tuple(generated) == path:
                allowed.update(self.eos)
            elif len(generated) < len(path) and tuple(generated) == path[:len(generated)]:
                allowed.add(path[len(generated)])
        if not allowed:
            raise RuntimeError(f'Invalid action trie prefix: {generated}')
        return tuple(allowed)

    def __call__(self, input_ids, scores):
        if len(scores) != len(self.stop_available):
            raise RuntimeError('Independent greedy sequences required')
        for row, stop in enumerate(self.stop_available):
            generated = input_ids[row, self.prompt_length:].tolist()
            allowed = self.allowed(generated, stop)
            if int(scores[row].argmax()) not in allowed:
                self.masked_argmax[row] += 1
            mask = torch.ones_like(scores[row], dtype=torch.bool)
            mask[list(allowed)] = False
            scores[row].masked_fill_(mask, -torch.inf)
        return scores
