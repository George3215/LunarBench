"""Common final-answer extraction; never infer an action from prose."""
import json
import re

PLAN_KEYS = ("goal_evidence", "tentative_route", "occlusion_hypothesis", "uncertainty", "revision_trigger")


def final_json(raw):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    # Action must be explicitly in a complete JSON object. No regex extraction
    # of 'move forward' and no fabricated visibility/assessment/schema fields.
    actions = [v for v in candidates if "action" in v]
    selected = actions[-1] if actions else candidates[-1] if candidates else None
    return json.dumps(selected) if selected is not None else ""


def decision_json(raw):
    """Accept JSON or explicit named fields, not an action inferred from prose."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    extracted = final_json(text)
    value = json.loads(extracted) if extracted else {}
    names = ('action', 'target_visible', 'target_location', 'proximity', 'confidence',
             'visual_assessment', 'decision_rationale', 'evidence_frames', 'evidence_sources',
             'spatial_belief', 'plan_update', 'expected_reappearance_m')
    patterns = {k.replace('_', ' '): k for k in names}
    expression = '|'.join(re.escape(k).replace(r'\ ', '[ _]') for k in patterns)
    matches = list(re.finditer(r'(?im)^\s*('+expression+r')\s*:\s*', text))
    for i, match in enumerate(matches):
        key = match.group(1).lower().replace(' ', '_')
        field = text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
        if key in value:
            if key == 'action' and field != value[key]:
                return ''  # conflicting explicit action declarations
            continue
        try:
            if key in ('target_visible', 'confidence', 'evidence_frames', 'expected_reappearance_m'):
                parsed = json.loads(field)
            elif key == 'evidence_sources':
                parsed = json.loads(field) if field.startswith('[') else [s.strip() for s in field.split(',')]
            else:
                parsed = field
        except ValueError:
            return ''
        value[key] = parsed
    return json.dumps(value) if 'action' in value else ''


def initial_plan(raw):
    extracted = final_json(raw)
    try:
        plan = json.loads(extracted)
    except ValueError:
        plan = None
    if isinstance(plan, dict) and all(isinstance(plan.get(k), str) and plan[k].strip() for k in PLAN_KEYS):
        return plan, "json"
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    pattern = r"(?m)^\s*(" + "|".join(PLAN_KEYS) + r")\s*:\s*"
    labels = list(re.finditer(pattern, text))
    if [m.group(1) for m in labels] != list(PLAN_KEYS):
        return None, "invalid"
    plan = {m.group(1): text[m.end():labels[i+1].start() if i+1 < len(labels) else len(text)].strip()
            for i, m in enumerate(labels)}
    return (plan, "labeled_fields") if all(plan.values()) else (None, "invalid")
