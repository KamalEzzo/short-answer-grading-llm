"""Few-shot run for one commercial model on one split (validation or test).

Usage (from the repository root): python scripts/inference/fewshot_run.py <model> <validation|test>
Needs data/Data.xlsx, data/fewshot_examples.json (made by fewshot_draw_examples.py) and the
environment variables OPENAI_API_KEY / ANTHROPIC_API_KEY.
The zero-shot prompt, clean_text and parse_response are loaded word for word from notebook 08.
Resumes from progress.json if interrupted. Never overwrites a finished run.
"""
import os, re, sys, json, time, datetime, platform
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from metrics import qwk
NB08 = os.path.join(ROOT, 'notebooks', '08_Commercial_Models_Inference.ipynb')
DATA = os.path.join(ROOT, 'data', 'Data.xlsx')
SPLITS = pd.read_csv(os.path.join(ROOT, 'data', 'splits.csv'))
VAL = [int(q[1:]) for q in SPLITS.question[SPLITS.split == 'validation']]
S = ['Clarity_Score', 'Terminology_Score', 'Coverage_Score', 'Accuracy_Score']
MAX = {'Clarity_Score': 2, 'Terminology_Score': 2, 'Coverage_Score': 2, 'Accuracy_Score': 4}
LABELS = {'On-Topic', 'Off-Topic', 'No Answer'}
MAX_TOKENS, SEED, WORKERS = 400, 42, 4

# --- zero-shot code (February 2026, notebook 08), word for word -------------------------------------------
cells = [''.join(c['source']) for c in json.load(open(NB08))['cells'] if c['cell_type'] == 'code']
feb = {'re': re, 'json': json, 'pd': pd}
exec(next(c for c in cells if 'PROMPT = """' in c), feb)     # PROMPT, TEST_QUESTIONS, ...
exec(next(c for c in cells if 'def parse_response' in c), feb)  # clean_text, parse_response
PROMPT, TEST, clean_text = feb['PROMPT'], feb['TEST_QUESTIONS'], feb['clean_text']
assert sorted(TEST) == [int(q[1:]) for q in SPLITS.question[SPLITS.split == 'test']], 'test questions differ from data/splits.csv'


# --- Few-shot block: inserted after the role line, nothing else changes ---------
def fields(question, model_answer, student_answer):
    """Same text preparation as the zero-shot run."""
    return str(question).strip(), clean_text(model_answer), str(student_answer).strip()


def examples_block():
    ex = json.load(open(os.path.join(ROOT, 'data', 'fewshot_examples.json')))
    parts = ['**EXAMPLES:** Five answers to other questions, graded by expert examiners.']
    for n, e in enumerate(ex, 1):
        q, m, a = fields(e['question'], e['model_answer'], e['student_answer'])
        parts.append(f'Example {n}\nQuestion: {q}\nModel answer: {m}\nStudent answer: {a}\n'
                     f'Output:\n{json.dumps(e["output"], indent=2, ensure_ascii=False)}')
    return '\n\n'.join(parts)


role, rest = PROMPT.split('\n\n', 1)
assert role.startswith('You are an expert AI Teaching Assistant') and rest.startswith('**QUESTION:**')
FEWSHOT_PROMPT = role + '\n\n' + examples_block().replace('{', '{{').replace('}', '}}') + '\n\n' + rest


# --- Reading a reply (zero-shot parser + the failure rules) ----------------------
def read_reply(raw, stop_reason):
    """Returns (scores dict, status). status: ok / fallback / failed:<why>; clipped scores counted in 'clipped'."""
    zero = {'off_topic': 'On-Topic', 'Off_Topic_Reason': '', **{k: 0 for k in S}, 'clipped': 0}
    if raw is None: return zero, 'failed:api_error'
    if stop_reason == 'refusal': return zero, 'failed:refusal'
    if stop_reason in ('max_tokens', 'length'): return zero, 'failed:truncated'
    status = 'ok'
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    try:
        parsed = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:                                # zero-shot fallback: regex
        status, parsed = 'fallback', {}
        for k in ['off_topic', 'Off_Topic_Reason']:
            r = re.search(rf'"{k}"\s*:\s*"([^"]+)"', raw)
            if r: parsed[k] = r.group(1)
        for k in S:
            r = re.search(rf'"?{k}"?\s*:\s*([^,}}\n"]*)', raw)     # whole value up to , }} newline or quote
            if r: parsed[k] = int(r.group(1).strip()) if re.fullmatch(r'\d+', r.group(1).strip()) else r.group(1)
    if not all(k in parsed for k in ['off_topic', 'Off_Topic_Reason'] + S):
        return zero, 'failed:unparseable' if status == 'fallback' and not parsed else 'failed:missing_field'
    if not isinstance(parsed['off_topic'], str) or parsed['off_topic'] not in LABELS: return zero, 'failed:invalid_label'
    if not all(isinstance(parsed[k], int) and not isinstance(parsed[k], bool) for k in S):
        return zero, 'failed:non_integer'
    out = {'off_topic': parsed['off_topic'], 'Off_Topic_Reason': parsed['Off_Topic_Reason'], 'clipped': 0}
    for k in S:
        v = min(max(parsed[k], 0), MAX[k])
        out['clipped'] += v != parsed[k]
        out[k] = v
    return out, status


# --- API calls -----------------------------------------------------------------
def key(name):
    return os.environ[name]


def make_caller(model):
    if model.startswith('claude'):
        from anthropic import Anthropic
        c = Anthropic(api_key=key('ANTHROPIC_API_KEY'))
        def call(msg):
            r = c.messages.create(model=model, max_tokens=MAX_TOKENS, temperature=0,
                                  messages=[{'role': 'user', 'content': msg}])
            text = ''.join(b.text for b in r.content if b.type == 'text')
            return text, r.stop_reason, r.model, None, r.usage.model_dump()
    else:
        from openai import OpenAI
        c = OpenAI(api_key=key('OPENAI_API_KEY'))
        def call(msg):
            p = {'model': model, 'messages': [{'role': 'user', 'content': msg}], 'temperature': 0, 'seed': SEED}
            p['max_completion_tokens' if 'gpt-5' in model else 'max_tokens'] = MAX_TOKENS
            r = c.chat.completions.create(**p)
            ch = r.choices[0]
            stop = 'refusal' if getattr(ch.message, 'refusal', None) else ch.finish_reason
            return (ch.message.content or ch.message.refusal or '', stop, r.model, r.system_fingerprint,
                    r.usage.model_dump())
    return call


def grade(call, row):
    q, m, a = fields(row['Question'], row['Model Answer'], row['Answer Text'])
    msg = FEWSHOT_PROMPT.format(question=q, model_answer=m, student_answer=a)
    rec = {'student_id': row['ID'], 'raw_response': None, 'stop_reason': None, 'response_model': None,
           'fingerprint': None, 'usage': None, 'in_tokens': 0, 'out_tokens': 0, 'attempts': 0, 'api_error': ''}
    for attempt in range(3):                          # retries for API errors only, as in the zero-shot run
        rec['attempts'] = attempt + 1
        try:
            (rec['raw_response'], rec['stop_reason'], rec['response_model'], rec['fingerprint'],
             usage) = call(msg)
            rec['usage'] = json.dumps(usage)
            rec['in_tokens'] = usage.get('input_tokens', usage.get('prompt_tokens', 0))
            rec['out_tokens'] = usage.get('output_tokens', usage.get('completion_tokens', 0))
            break
        except Exception as e:
            rec['api_error'] = f'{type(e).__name__}: {e}'[:300]
            if attempt < 2: time.sleep(5)
    try:
        llm, status = read_reply(rec['raw_response'], rec['stop_reason'])
    except Exception as e:                            # safety net: never lose a paid reply
        llm, status = read_reply(None, None)[0], f'failed:reader_error {type(e).__name__}'
    h = {k: int(row[k]) for k in S}
    rec.update({'status': status, 'clipped': llm['clipped'],
                'Human_OffTopic': str(row['off_topic']), 'LLM_OffTopic': llm['off_topic'],
                'LLM_Reason': llm['Off_Topic_Reason']})
    for k in S:
        n = k.replace('_Score', '')
        rec[f'Human_{n}'], rec[f'LLM_{n}'] = h[k], llm[k]
    rec['Human_Final'] = sum(h.values()) / 2
    rec['LLM_Final'] = sum(llm[k] for k in S) / 2
    return rec


def main(model, split):
    df = pd.read_excel(DATA)
    df['Q_num'] = df['ID'].apply(lambda x: int(str(x).split(' ')[0].replace('Q', '')))
    qs = {'validation': VAL, 'test': TEST}[split]
    df = df[df.Q_num.isin(qs)].reset_index(drop=True)
    assert len(df) == 510 and df.Q_num.nunique() == 17

    out = os.path.join(ROOT, 'results', 'commercial_fewshot', model.replace('.', '_'), split)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'results.csv')):
        sys.exit(f'{out}/results.csv exists: finished run, not overwritten.')
    with open(os.path.join(ROOT, 'data', f"prompt_example_{model.replace('.', '_')}_{split}.txt"), 'w') as f:   # first filled prompt (contains student text: kept in data/, not published)
        f.write(FEWSHOT_PROMPT.format(**dict(zip(['question', 'model_answer', 'student_answer'],
                                                 fields(*df.loc[0, ['Question', 'Model Answer', 'Answer Text']])))))

    prog = os.path.join(out, 'progress.jsonl')             # one line per finished reply, written at once
    done = []
    if os.path.exists(prog):                                  # drop a half-written last line, if any
        lines = open(prog).read().split('\n')
        for l in lines:
            try: done.append(json.loads(l))
            except json.JSONDecodeError: pass
        with open(prog + '.tmp', 'w') as t:                   # rewrite safely: temp file, then swap
            t.write(''.join(json.dumps(d) + '\n' for d in done)); t.flush(); os.fsync(t.fileno())
        os.replace(prog + '.tmp', prog)
    todo = [r for _, r in df.iterrows() if r['ID'] not in {d['student_id'] for d in done}]
    print(f'{model} {split}: {len(done)} done, {len(todo)} to go', flush=True)
    call, t0 = make_caller(model), time.time()
    with ThreadPoolExecutor(WORKERS) as ex, open(prog, 'a') as f:
        for fut in as_completed([ex.submit(grade, call, r) for r in todo]):
            rec = fut.result()
            f.write(json.dumps(rec) + '\n'); f.flush(); os.fsync(f.fileno())
            done.append(rec)
            if len(done) % 25 == 0 or len(done) == 510:
                d = pd.DataFrame(done)
                h, l = (d[[c for c in d if c.startswith(w) and c.split('_')[1] in
                           ('Clarity', 'Terminology', 'Coverage', 'Accuracy')]].sum(axis=1) for w in ('Human_', 'LLM_'))
                print(f'  {len(done)}/510  failed {d.status.str.startswith("failed").sum()}  '
                      f'±1 {((l - h).abs() / 2 <= 1).mean() * 100:.1f}%  QWK {qwk(h, l):.3f}  '
                      f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    res = pd.DataFrame(done).set_index('student_id').loc[df.ID].reset_index()   # original order
    res.to_csv(os.path.join(out, 'results.csv'), index=False)
    import anthropic, openai
    log = {'model_requested': model, 'split': split, 'run_date': datetime.datetime.now().isoformat(timespec='seconds'),
           'response_models': res.response_model.value_counts().to_dict(),
           'fingerprints': res.fingerprint.dropna().value_counts().to_dict(),
           'status_counts': res.status.value_counts().to_dict(), 'clipped_scores': int(res.clipped.sum()),
           'stop_reasons': res.stop_reason.astype(str).value_counts().to_dict(),
           'in_tokens': int(res.in_tokens.sum()), 'out_tokens': int(res.out_tokens.sum()),
           'cost': 'full usage saved per reply (usage column); charge taken from the provider console for this key and date',
           'settings': {'temperature': 0, 'max_tokens': MAX_TOKENS, 'seed (OpenAI only)': SEED, 'workers': WORKERS},
           'versions': {'python': platform.python_version(), 'anthropic': anthropic.__version__,
                        'openai': openai.__version__, 'pandas': pd.__version__},
           'minutes': round((time.time() - t0) / 60, 1)}
    json.dump(log, open(os.path.join(out, 'run_log.json'), 'w'), indent=2)
    os.remove(prog)
    print(json.dumps(log, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
