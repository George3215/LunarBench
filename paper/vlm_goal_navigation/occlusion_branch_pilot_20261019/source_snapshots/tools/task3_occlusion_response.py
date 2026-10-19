"""Deterministic semantic extraction, with explicitly reported syntax-only repair."""
import json


def valid(value,check_time=True):
    if not isinstance(value,dict):return False
    c=value.get('choice');f=value.get('forecasts')
    if not isinstance(c,str) or len(c)!=1 or c not in 'ABCD':return False
    if not isinstance(f,dict) or set(f)!=set('ABCD'):return False
    for p in f.values():
        if not isinstance(p,dict):return False
        prob=p.get('p');t=p.get('time_s')
        if isinstance(prob,bool) or not isinstance(prob,(float,int)) or not 0<=prob<=1:return False
        if check_time and t is not None and (isinstance(t,bool) or not isinstance(t,(float,int)) or not 0<t<=6):return False
    return True


def extract(raw,check_time=True):
    if '</think>' in raw:raw=raw.rsplit('</think>',1)[1]
    if '<answer>' in raw:raw=raw.split('<answer>',1)[1].split('</answer>',1)[0]
    decoder=json.JSONDecoder()
    for i,c in enumerate(raw):
        if c!='{':continue
        try:value,_=decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:continue
        if valid(value,check_time):return value,'strict_json'
    # Only append up to two missing final '}' characters. No values, keys, choices,
    # times or probabilities are inferred or changed; incomplete fields stay invalid.
    for i,c in enumerate(raw):
        if c!='{':continue
        tail=raw[i:].strip().removesuffix('```').strip()
        for n in [1,2]:
            try:value=json.loads(tail+'}'*n)
            except json.JSONDecodeError:continue
            if valid(value,check_time):return value,f'append_{n}_final_brace'
    return None,'invalid'
