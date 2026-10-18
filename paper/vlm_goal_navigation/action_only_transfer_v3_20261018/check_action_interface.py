import sys
sys.path.insert(0,'tools')
import torch
from transformers import AutoTokenizer
from task3_action_only import ClosedSetActionGrammar, MOTION, ACTIONS, parse_action
from pathlib import Path
results=[]
for name in ['qwen3vl8b','spatial_interactor8b','spatialreasoner','robobrain25_8b_nv']:
    tokenizer=AutoTokenizer.from_pretrained(str(Path('.local/models')/name),local_files_only=True)
    eos=tokenizer.eos_token_id
    for stop in [False,True]:
        grammar=ClosedSetActionGrammar(tokenizer,[stop],0,[eos])
        leaves=[]
        def visit(prefix):
            for token in grammar.allowed(prefix,stop):
                if token==eos:
                    leaves.append(tokenizer.decode(prefix,skip_special_tokens=True).strip())
                else:
                    assert len(prefix)<14
                    visit(prefix+[token])
        visit([])
        assert set(leaves)==set(ACTIONS if stop else MOTION),(name,leaves)
        scores=torch.arange(len(tokenizer),dtype=torch.float32).unsqueeze(0)
        expected=scores.clone()
        allowed=grammar.allowed([],stop)
        grammar(torch.empty((1,0),dtype=torch.long),scores)
        assert torch.equal(scores[0,list(allowed)],expected[0,list(allowed)])
        mask=torch.ones(len(tokenizer),dtype=torch.bool);mask[list(allowed)]=False
        assert torch.isneginf(scores[0,mask]).all()
    assert max(map(len,grammar.paths.values()))+1<=16
    results.append(dict(model=name,trie_leaves_verified=True,legal_logits_unchanged=True,max_action_tokens=max(map(len,grammar.paths.values()))+1))
assert parse_action('TURN_LEFT')['action']=='TURN_LEFT'
assert parse_action('I would TURN_LEFT') is None
assert parse_action('{"action":"FORWARD"}') is None
assert parse_action('FORWARD STOP') is None
assert parse_action('FORWARD')['evidence_frames']==[]
import json
print(json.dumps({'status':'passed','checks':results},indent=2))
