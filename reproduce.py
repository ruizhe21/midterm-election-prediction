"""Reproduce the two submission CSVs; intermediate calculations stay temporary."""
from pathlib import Path
import argparse, shutil, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refit', action='store_true', help='Refit the same frozen models on the included historical training data.')
    parser.add_argument('--output-dir', default='predictions', help='Destination for the two final CSVs.')
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    with tempfile.TemporaryDirectory(prefix='congress_forecast_') as temporary:
        work = Path(temporary)
        snapshot = work / 'model_snapshot'
        shutil.copytree(ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT / 'inputs', snapshot / 'inputs')
        shutil.copytree(ROOT / 'models', snapshot / 'models')
        shutil.copy2(ROOT / 'model_spec.json', snapshot / 'model_spec.json')
        (snapshot / 'results').mkdir()
        for name in ['ballot_candidates.csv', 'ballot_name_crosswalk.csv']:
            shutil.copy2(ROOT / 'inputs' / name, snapshot / 'results' / name)
        commands = []
        if args.refit:
            commands.append(['model.py'])
        commands += [['build_draft.py'], ['identity_sensitivity.py'],
                     ['build_draft.py', '--extend-domain'], ['build_review_draft.py']]
        for command in commands:
            result = subprocess.run([sys.executable, str(snapshot / 'src' / command[0]), *command[1:]],
                                    cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(command[0] + '\n' + result.stdout + result.stderr)
        shutil.copytree(ROOT / 'inputs' / 'official', work / 'inputs' / 'official')
        shutil.copy2(ROOT / 'inputs' / 'completion_policy.json', work / 'inputs' / 'completion_policy.json')
        (work / 'src').mkdir()
        shutil.copy2(ROOT / 'src' / 'complete_ballot.py', work / 'src' / 'complete_ballot.py')
        subprocess.run([sys.executable, str(work / 'src' / 'complete_ballot.py')], cwd=work, check=True)
        for name, template in [('full_congress_FINAL.csv', 'full_congress_template.csv'),
                               ('senate_FINAL.csv', 'senate_template.csv')]:
            subprocess.run([sys.executable, str(snapshot / 'src' / 'validate_submission.py'),
                            str(work / name), str(ROOT / 'inputs' / template)], check=True)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['full_congress_FINAL.csv', 'senate_FINAL.csv']:
            shutil.copy2(work / name, destination / name)
    print('Saved only the two final prediction CSVs to ' + str(destination))

if __name__ == '__main__':
    main()
