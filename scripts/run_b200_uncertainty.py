"""Run the frozen development-set baseline replication; invokes paid judging."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import common
common.load_env()


def main():
    for condition in json.loads((ROOT / 'runs/b200_uncertainty/conditions.json').read_text()):
        name = condition['name']
        output = ROOT / 'runs' / name / 'eval/samples.jsonl'
        command = [sys.executable, 'scripts/steer_sample.py', '--name', name, '--adapter', 't_finance',
                   '--questions', 'eval/prompt_pool_gate.yaml', '--samples', '5', '--batch-seqs', '30',
                   '--seed', str(condition['seed'])]
        if condition['intervention'] == 'leak':
            command += ['--vector', 'runs/exp5/dirs/t_finance.npz', '--key', 'leak', '--layers',
                        '12,16,20,24,28', '--mode', 'ablate']
        if not output.exists():
            print('Generating', name, flush=True)
            with (ROOT / 'logs' / f'{name}_sample.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if len(rows) != 480 or len({(r['id'], r['paraphrase_idx'], r['sample_idx']) for r in rows}) != 480:
            raise ValueError(f'{name}: incomplete generation')
        print('Judging', name, flush=True)
        with (ROOT / 'logs' / f'{name}_judge.log').open('a') as log:
            subprocess.run([sys.executable, 'scripts/judge.py', '--name', name, '--questions',
                            'eval/prompt_pool_gate.yaml', '--provider', 'anthropic', '--strict-output',
                            '--concurrency', '8'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed', name, flush=True)


    # Judge the identical original pilot answers with the second model, preserving Haiku rows.
    for condition in ('baseline', 'leak'):
        name = 'b200_crossjudge_' + condition
        source = ROOT / 'runs' / ('b200_finance_' + condition) / 'eval/samples.jsonl'
        output = ROOT / 'runs' / name / 'eval/samples.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.read_bytes() != source.read_bytes():
            raise ValueError(f'{name}: cross-judge samples differ from the original pilot')
        if not output.exists():
            output.write_bytes(source.read_bytes())
        print('Cross-judging', name, flush=True)
        with (ROOT / 'logs' / f'{name}_judge.log').open('a') as log:
            subprocess.run([sys.executable, 'scripts/judge.py', '--name', name, '--questions',
                            'eval/prompt_pool_gate.yaml', '--provider', 'openai', '--strict-output',
                            '--concurrency', '8'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Completed', name, flush=True)


if __name__ == '__main__':
    main()
