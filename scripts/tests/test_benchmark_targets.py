"""Name migration tests, including both checkout builds with a fake Cargo."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
import benchmark_targets as TARGETS


def metadata(target, feature, package='aerie', kind='bench'):
    return {'packages': [{'name': package, 'targets': [{'name': target, 'kind': [kind]}],
                          'features': {name: [] for name in feature.split(',')}}]}


class TargetSelectionTests(unittest.TestCase):
    def test_target_and_feature_are_independent_per_checkout(self):
        for target in ('falcon', 'falcon-v1'):
            for feature in ('falcon-bench', 'falcon-v1-bench', 'parallel,honest-vectors'):
                with self.subTest(target=target, feature=feature):
                    self.assertEqual(TARGETS.select_falcon_benchmark(metadata(target, feature)),
                                     (target, feature))

    def test_prefers_stable_names_if_both_generations_are_declared(self):
        value = metadata('falcon-v1', 'falcon-v1-bench,falcon-bench')
        value['packages'][0]['targets'].append({'name': 'falcon', 'kind': ['bench']})
        self.assertEqual(TARGETS.select_falcon_benchmark(value), ('falcon', 'falcon-bench'))

    def test_rejects_unknown_package_target_kind_and_feature(self):
        invalid = [metadata('unrelated', 'falcon-bench'), metadata('falcon', 'unrelated'),
                   metadata('falcon', 'parallel'), metadata('falcon', 'falcon-bench', kind='example'),
                   metadata('falcon', 'falcon-bench', package='unrelated'), {'packages': []}]
        duplicated = metadata('falcon', 'falcon-bench')
        duplicated['packages'] *= 2
        invalid.append(duplicated)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TARGETS.select_falcon_benchmark(value)

    def test_selector_cli_emits_only_two_allowlisted_fields(self):
        script = Path(TARGETS.__file__)
        result = subprocess.run([sys.executable, str(script), 'falcon'],
                                input=json.dumps(metadata('falcon-v1', 'falcon-bench')),
                                text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout, 'falcon-v1\tfalcon-bench\n')


@unittest.skipUnless(shutil.which('bash') and shutil.which('jq'), 'workflow requires bash and jq')
class WorkflowCheckoutBuildTests(unittest.TestCase):
    def test_workflow_builds_old_base_and_new_head_from_separate_metadata(self):
        repo = Path(__file__).resolve().parents[2]
        workflow = (repo / '.github/workflows/falcon-bench.yml').read_text()
        start = workflow.index('          build_bench() {')
        end = workflow.index('\n          build_bench \\', start)
        function = 'selector="$AERIE_BENCH_SELECTOR"\n' + textwrap.dedent(workflow[start:end])
        with tempfile.TemporaryDirectory(prefix='falcon names ') as tmp:
            directory = Path(tmp).resolve()
            commands = directory / 'commands.jsonl'
            bin_dir = directory / 'bin'
            bin_dir.mkdir()
            cargo = bin_dir / 'cargo'
            cargo.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''\
                import json, os, pathlib, sys
                args = sys.argv[1:]
                cwd = pathlib.Path.cwd()
                with pathlib.Path(os.environ['FAKE_CARGO_COMMANDS']).open('a') as log:
                    log.write(json.dumps({'cwd': str(cwd), 'args': args}) + '\\n')
                if args[0] == 'metadata':
                    print((cwd / 'metadata.json').read_text())
                elif args[0] == 'build':
                    target = args[args.index('--bench') + 1]
                    binary = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / target
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_text(os.environ['AERIE_BENCH_BUILD_REVISION'])
                    binary.chmod(0o755)
                    print(json.dumps({'reason': 'compiler-artifact',
                          'target': {'name': target, 'kind': ['bench']},
                          'executable': str(binary)}))
                else:
                    raise SystemExit('Unexpected Cargo command')
                '''))
            cargo.chmod(0o755)
            sources = []
            selections = [('falcon-v1', 'falcon-v1-bench'), ('falcon', 'falcon-bench'),
                          ('falcon-v1', 'falcon-bench'), ('falcon', 'falcon-v1-bench')]
            for index, (target, feature) in enumerate(selections):
                source = directory / f'checkout-{index}'
                source.mkdir()
                (source / 'metadata.json').write_text(json.dumps(metadata(target, feature)))
                sources.append(source)
            shell = 'set -euo pipefail\n' + function + '\n' + '''\
                for source_dir in "$@"; do
                  build_bench "$source_dir" "$source_dir/frozen" "$source_dir/target" \\
                    "$(basename "$source_dir")" "$source_dir/build.jsonl"
                done
                '''
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       AERIE_BENCH_SELECTOR=str(repo / 'scripts/benchmark_targets.py'),
                       FAKE_CARGO_COMMANDS=str(commands))
            subprocess.run(['bash', '-c', shell, 'bench-test', *map(str, sources)],
                           env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in commands.read_text().splitlines()]
            for source, (target, feature) in zip(sources, selections):
                checkout_calls = [call['args'] for call in calls if call['cwd'] == str(source)]
                self.assertEqual(sum(call[0] == 'metadata' for call in checkout_calls), 1)
                build = next(call for call in checkout_calls if call[0] == 'build')
                self.assertEqual(build[build.index('--bench') + 1], target)
                self.assertEqual(build[build.index('--features') + 1], feature)
                self.assertEqual((source / 'frozen').read_text(), source.name)


if __name__ == '__main__':
    unittest.main()
