import json, sys, importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('bm', 'tools/benchmark_nas_samples.py')
bm = importlib.util.module_from_spec(spec); sys.modules['bm'] = bm; spec.loader.exec_module(bm)
sys.path.insert(0, '01-FrameAnalyzer/risk_hunter')
lfs = importlib.util.spec_from_file_location('lf', '01-FrameAnalyzer/risk_hunter/log_first_detector.py')
lf = importlib.util.module_from_spec(lfs); sys.modules['lf'] = lf; lfs.loader.exec_module(lf)
rls = importlib.util.spec_from_file_location('rl', 'tools/log_signal_rules.py')
rl = importlib.util.module_from_spec(rls); sys.modules['rl'] = rl; rls.loader.exec_module(rl)

root = Path('spec/data/nas_samples')
RUN = 'spec/output/nas_vlm_practical_adaptive_20260702_140225'
errs = json.load(open(f'{RUN}/report_errors.json', encoding='utf-8'))['wrong_cases']
fn_cases = {e['case'].replace('\\', '/') for e in errs if e['bucket'] == 'fn'}

rows = []
for stage in ('stage1', 'stage2', 'stage4'):
    sd = root / stage
    if not sd.exists():
        continue
    seen = set()
    for case in sorted(sd.iterdir()):
        if not case.is_dir():
            continue
        dirs = {case}
        for extra in case.rglob('groundtruth.json'):
            dirs.add(extra.parent)
        for cd in sorted(dirs):
            if cd in seen:
                continue
            seen.add(cd)
            gt_path = bm._choose_groundtruth(cd)
            if not gt_path:
                continue
            try:
                gt = bm._read_json_lenient(gt_path)
                logs, src = bm._load_case_logs(cd, lf)
            except Exception as exc:
                continue
            if not logs:
                continue
            expected = bm._expected_positive(gt)
            sens = bm._sensitive_files_from_groundtruth(gt)
            for p in bm._sensitive_files_from_logs(logs, lf):
                if p.lower() not in {s.lower() for s in sens}:
                    sens.append(p)
            if not sens and expected:
                for p in bm._fallback_sensitive_files_from_logs(logs, lf):
                    sens.append(p)
            sig = rl.extract_deterministic_signals(logs, sens, lf.is_sensitive_name)
            rows.append({'case': str(cd.relative_to(root)).replace('\\', '/'),
                         'expected': expected, 'rules': sig['rules'],
                         'evidence': {k: v[:2] for k, v in sig['evidence'].items()}})

pos = [r for r in rows if r['expected']]
neg = [r for r in rows if not r['expected']]
print('cases:', len(rows), 'pos:', len(pos), 'neg:', len(neg))
rule_names = sorted({x for r in rows for x in r['rules']})
print(f"{'rule':22s} {'pos':>4s} {'neg':>4s}")
for rn in rule_names:
    ph = sum(1 for r in pos if rn in r['rules'])
    nh = sum(1 for r in neg if rn in r['rules'])
    print(f"{rn:22s} {ph:4d} {nh:4d}")
print('union pos:', sum(1 for r in pos if r['rules']), '/', len(pos))
print('union neg (FP!):', sum(1 for r in neg if r['rules']))
for r in neg:
    if r['rules']:
        print('  NEG-HIT', r['case'], r['rules'])
        for k, v in r['evidence'].items():
            print('     ', k, json.dumps(v, ensure_ascii=False)[:220])
print()
print('current-FN coverage:', sum(1 for r in rows if r['case'] in fn_cases and r['rules']), '/', len(fn_cases))
for r in rows:
    if r['case'] in fn_cases:
        print(('  FIXED ' if r['rules'] else '  still ') + r['case'], r['rules'])
json.dump(rows, open('spec/output/_rule_validation.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
