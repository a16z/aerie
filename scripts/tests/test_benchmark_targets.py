"""Falcon target selection and isolated checkout build tests."""
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
import falcon_bench_report as REPORT


def metadata(target, feature, package='aerie-bench', kind='bench', workspace_root='/workspace',
             harness=None):
    harness = harness or f'crates/{package}/benches/falcon.rs'
    return {'workspace_root': workspace_root,
            'packages': [{'name': package, 'targets': [{'name': target, 'kind': [kind],
                          'src_path': str(Path(workspace_root) / harness)}],
                          'features': {name: [] for name in feature.split(',')}}]}


class TargetSelectionTests(unittest.TestCase):
    def test_selects_falcon_target_feature_package_and_harness(self):
        for side, package in [('base', 'aerie'), ('base', 'aerie-bench'), ('head', 'aerie-bench')]:
            with self.subTest(side=side, package=package):
                self.assertEqual(TARGETS.select_falcon_benchmark(
                    metadata('falcon', 'falcon-bench', package=package), side=side),
                    ('falcon', 'falcon-bench', package, f'crates/{package}/benches/falcon.rs'))

    def test_head_rejects_old_layout(self):
        value = metadata('falcon', 'falcon-bench', package='aerie')
        for kwargs in [{}, {'side': 'head'}]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, '^Expected exactly one Cargo package: aerie-bench$'):
                TARGETS.select_falcon_benchmark(value, **kwargs)

    def test_prefers_aerie_bench_when_both_packages_exist(self):
        value = metadata('falcon', 'falcon-bench', package='aerie')
        value['packages'] += metadata('falcon', 'falcon-bench')['packages']
        for side in ['base', 'head']:
            with self.subTest(side=side):
                self.assertEqual(TARGETS.select_falcon_benchmark(value, side=side),
                                 ('falcon', 'falcon-bench', 'aerie-bench',
                                  'crates/aerie-bench/benches/falcon.rs'))

    def test_harness_comes_from_selected_target_metadata(self):
        self.assertEqual(TARGETS.select_falcon_benchmark(
            metadata('falcon', 'falcon-bench', harness='custom/falcon.rs')),
            ('falcon', 'falcon-bench', 'aerie-bench', 'custom/falcon.rs'))

    def test_rejects_missing_target_or_feature_in_selected_package(self):
        for package in ['aerie-bench', 'aerie']:
            for target, feature, message in [
                ('unrelated', 'falcon-bench', 'No falcon benchmark target in '),
                ('falcon', 'unrelated', 'No falcon-bench feature in '),
            ]:
                with self.subTest(package=package, target=target, feature=feature):
                    with self.assertRaisesRegex(ValueError, '^' + message + package + '$'):
                        TARGETS.select_falcon_benchmark(metadata(target, feature, package=package),
                                                        side='base')

    def test_does_not_fall_back_when_aerie_bench_exists_but_is_invalid(self):
        value = metadata('falcon', 'falcon-bench', package='aerie')
        value['packages'] += metadata('unrelated', 'falcon-bench')['packages']
        for side in ['base', 'head']:
            with self.subTest(side=side), self.assertRaisesRegex(
                    ValueError, '^No falcon benchmark target in aerie-bench$'):
                TARGETS.select_falcon_benchmark(value, side=side)

    def test_rejects_unknown_package_target_kind_and_feature(self):
        invalid = [metadata('unrelated', 'falcon-bench'), metadata('falcon', 'unrelated'),
                   metadata('falcon', 'parallel'), metadata('falcon', 'falcon-bench', kind='example'),
                   metadata('falcon', 'falcon-bench', package='unrelated'),
                   metadata('falcon', 'falcon-bench', package='aerie', kind='example'),
                   {'packages': []}]
        for package in ['aerie-bench', 'aerie']:
            duplicated = metadata('falcon', 'falcon-bench', package=package)
            duplicated['packages'] *= 2
            invalid.append(duplicated)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TARGETS.select_falcon_benchmark(value, side='base')

    def test_selector_cli_emits_four_fields(self):
        script = Path(TARGETS.__file__)
        for side, package in [('base', 'aerie'), ('base', 'aerie-bench'), ('head', 'aerie-bench')]:
            with self.subTest(side=side, package=package):
                result = subprocess.run([sys.executable, str(script), 'falcon', '--side', side],
                                        input=json.dumps(metadata('falcon', 'falcon-bench',
                                                                  package=package)),
                                        text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout, 'falcon\tfalcon-bench\t' + package + '\t' +
                                 f'crates/{package}/benches/falcon.rs\n')

    def test_selector_cli_rejects_head_old_layout(self):
        result = subprocess.run([sys.executable, TARGETS.__file__, 'falcon', '--side', 'head'],
                                input=json.dumps(metadata('falcon', 'falcon-bench', package='aerie')),
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('Expected exactly one Cargo package: aerie-bench', result.stderr)


@unittest.skipUnless(shutil.which('bash') and shutil.which('jq'), 'workflow requires bash and jq')
class WorkflowCheckoutBuildTests(unittest.TestCase):
    def test_workflow_builds_base_and_head_in_isolated_directories(self):
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
            selections = [('falcon', 'falcon-bench', 'aerie'),
                          ('falcon', 'falcon-bench', 'aerie-bench')]
            for index, (target, feature, package) in enumerate(selections):
                source = directory / f'checkout-{index}'
                source.mkdir()
                (source / 'metadata.json').write_text(json.dumps(metadata(
                    target, feature, package=package, workspace_root=str(source))))
                sources.append(source)
            shell = 'set -euo pipefail\n' + function + '\n' + '''\
                for source_dir in "$@"; do
                  if [ "$(basename "$source_dir")" = checkout-0 ]; then
                    AERIE_BENCH_SIDE=base
                  else
                    AERIE_BENCH_SIDE=head
                  fi
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
            for source, (target, feature, package) in zip(sources, selections):
                checkout_calls = [call['args'] for call in calls if call['cwd'] == str(source)]
                self.assertEqual(sum(call[0] == 'metadata' for call in checkout_calls), 1)
                build = next(call for call in checkout_calls if call[0] == 'build')
                self.assertEqual(build[build.index('-p') + 1], package)
                self.assertEqual(build[build.index('--bench') + 1], target)
                self.assertEqual(build[build.index('--features') + 1], feature)
                self.assertEqual((source / 'frozen').read_text(), source.name)


class WorkflowSchemaTests(unittest.TestCase):
    def schema_shell(self):
        repo = Path(__file__).resolve().parents[2]
        workflow = (repo / '.github/workflows/falcon-bench.yml').read_text()
        start = workflow.index('          set -euo pipefail', workflow.index('      - name: Check benchmark schema'))
        end = workflow.index('\n      - name: Compute binary cache key', start)
        return 'export PATH="$(pwd)/bin:$PATH"\n' + textwrap.dedent(workflow[start:end])

    def write_reporter(self, directory, schema, package='aerie-bench', harness=None):
        reporter = directory / '.head-selector/scripts/falcon_bench_report.py'
        reporter.parent.mkdir(parents=True)
        reporter.write_text(f'BENCHMARK_OUTPUT_SCHEMA = {schema!r}\nprint(BENCHMARK_OUTPUT_SCHEMA)\n')

        shutil.copyfile(TARGETS.__file__, reporter.parent / 'benchmark_targets.py')
        (directory / 'metadata.json').write_text(json.dumps(metadata(
            'falcon', 'falcon-bench', package=package, workspace_root=str(directory), harness=harness)))
        bin_dir = directory / 'bin'
        bin_dir.mkdir()
        cargo = bin_dir / 'cargo'
        cargo.write_text('#!/bin/sh\ncat metadata.json\n')
        cargo.chmod(0o755)

    def test_schema_gate_skips_unsupported_base_and_requires_current_schema_head(self):
        shell = self.schema_shell()
        schema = REPORT.BENCHMARK_OUTPUT_SCHEMA
        valid_source = f'const BENCHMARK_SCHEMA: u32 = {schema};\n'
        sources = [None, '', 'const BENCHMARK_SCHEMA: u32 = invalid;\n',
                   valid_source * 2, f'const BENCHMARK_SCHEMA: u32 = {int(schema) - 1};\n',
                   valid_source]
        for side, package in [(side, package) for side in ['base', 'head']
                              for package in ['aerie-bench', 'aerie']]:
            for source_text in sources:
                with self.subTest(side=side, package=package, source=source_text), \
                        tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    source = directory / f'crates/{package}/benches/falcon.rs'
                    source.parent.mkdir(parents=True)
                    if source_text is not None:
                        source.write_text(source_text)
                    self.write_reporter(directory, schema, package=package)
                    output = directory / 'outputs'
                    output.touch()
                    env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                               AERIE_BENCH_SIDE=side, AERIE_BENCH_REVISION='test-revision')
                    result = subprocess.run(['bash', '-c', shell], cwd=directory, env=env,
                                            text=True, capture_output=True)
                    if side == 'head' and package == 'aerie':
                        # The head selector rejects the legacy package outright.
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertIn('benchmark target selection failed', result.stderr)
                        continue
                    supported = source_text == valid_source and (side == 'base' or package == 'aerie-bench')
                    exit_code = 0 if supported or side == 'base' else 1
                    self.assertEqual(result.returncode, exit_code, result.stderr)
                    self.assertIn(f'expected={schema}\n', output.read_text())
                    skip_file = directory / 'falcon-bench-bin/comparison-skipped.txt'
                    if side == 'base' and not supported:
                        self.assertIn('supported=false', output.read_text())
                        self.assertIn('no base comparison', skip_file.read_text().lower())
                        self.assertIn(f'does not emit benchmark schema {schema}', skip_file.read_text())
                    elif supported:
                        self.assertIn('supported=true', output.read_text())
                        self.assertFalse(skip_file.exists())
                    else:
                        self.assertIn(f'Head benchmark must emit benchmark schema {schema}', result.stderr)

    def test_schema_gate_follows_metadata_when_both_harness_files_exist(self):
        schema = REPORT.BENCHMARK_OUTPUT_SCHEMA
        valid_source = f'const BENCHMARK_SCHEMA: u32 = {schema};\n'
        cases = [('aerie-bench', valid_source, '', True),
                 ('aerie-bench', '', valid_source, False),
                 ('aerie', '', valid_source, True),
                 ('aerie', valid_source, '', False)]
        for selected, source_text, legacy_text, supported in cases:
            with self.subTest(selected=selected, source=source_text), \
                    tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                for package, text in [('aerie-bench', source_text), ('aerie', legacy_text)]:
                    source = directory / f'crates/{package}/benches/falcon.rs'
                    source.parent.mkdir(parents=True)
                    source.write_text(text)
                self.write_reporter(directory, schema, package=selected)
                if selected == 'aerie-bench':
                    value = json.loads((directory / 'metadata.json').read_text())
                    value['packages'] += metadata('falcon', 'falcon-bench', package='aerie',
                                                  workspace_root=str(directory))['packages']
                    (directory / 'metadata.json').write_text(json.dumps(value))
                output = directory / 'outputs'
                output.touch()
                env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                           AERIE_BENCH_SIDE='base', AERIE_BENCH_REVISION='test-revision')
                result = subprocess.run(['bash', '-c', self.schema_shell()], cwd=directory,
                                        env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'supported={str(supported).lower()}\n', output.read_text())

    def test_schema_gate_reads_custom_target_source_path(self):
        schema = REPORT.BENCHMARK_OUTPUT_SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'custom/falcon.rs'
            source.parent.mkdir(parents=True)
            source.write_text(f'const BENCHMARK_SCHEMA: u32 = {schema};\n')
            self.write_reporter(directory, schema, harness='custom/falcon.rs')
            output = directory / 'outputs'
            output.touch()
            env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                       AERIE_BENCH_SIDE='head', AERIE_BENCH_REVISION='test-revision')
            result = subprocess.run(['bash', '-c', self.schema_shell()], cwd=directory,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('supported=true\n', output.read_text())

    def test_schema_gate_skips_invalid_base_selection_despite_valid_legacy_harness(self):
        schema = REPORT.BENCHMARK_OUTPUT_SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'crates/aerie/benches/falcon.rs'
            source.parent.mkdir(parents=True)
            source.write_text(f'const BENCHMARK_SCHEMA: u32 = {schema};\n')
            self.write_reporter(directory, schema, package='aerie')
            value = json.loads((directory / 'metadata.json').read_text())
            value['packages'] += metadata('unrelated', 'falcon-bench',
                                          workspace_root=str(directory))['packages']
            (directory / 'metadata.json').write_text(json.dumps(value))
            output = directory / 'outputs'
            output.touch()
            env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                       AERIE_BENCH_SIDE='base', AERIE_BENCH_REVISION='test-revision')
            result = subprocess.run(['bash', '-c', self.schema_shell()], cwd=directory,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('supported=false\n', output.read_text())
            self.assertTrue((directory / 'falcon-bench-bin/comparison-skipped.txt').exists())

    def test_schema_gate_rejects_missing_or_malformed_head_reporter_output(self):
        for side in ['base', 'head']:
            for schema in ['', 'invalid', '-1', REPORT.BENCHMARK_OUTPUT_SCHEMA + '\ninvalid']:
                with self.subTest(side=side, schema=schema), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    self.write_reporter(directory, schema)
                    output = directory / 'outputs'
                    output.touch()
                    env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                               AERIE_BENCH_SIDE=side, AERIE_BENCH_REVISION='test-revision')
                    result = subprocess.run(['bash', '-c', self.schema_shell()], cwd=directory,
                                            env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn('Head benchmark reporter must emit a numeric benchmark schema',
                                  result.stderr)
                    self.assertEqual(output.read_text(), '')

    def test_binary_cache_key_uses_schema_gate_output(self):
        repo = Path(__file__).resolve().parents[2]
        workflow = (repo / '.github/workflows/falcon-bench.yml').read_text()
        start = workflow.index('          set -euo pipefail', workflow.index('      - name: Compute binary cache key'))
        end = workflow.index('\n      # A revision', start)
        cache_shell = textwrap.dedent(workflow[start:end])
        keys = []
        for schema in [REPORT.BENCHMARK_OUTPUT_SCHEMA, str(int(REPORT.BENCHMARK_OUTPUT_SCHEMA) + 1)]:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source = directory / 'crates/aerie-bench/benches/falcon.rs'
                source.parent.mkdir(parents=True)
                source.write_text(f'const BENCHMARK_SCHEMA: u32 = {schema};\n')
                self.write_reporter(directory, schema)
                bin_dir = directory / 'bin'
                rustc = bin_dir / 'rustc'
                rustc.write_text('#!/bin/sh\nprintf "rustc fixture\\n"\n')
                rustc.chmod(0o755)
                output = directory / 'outputs'
                output.touch()
                env = dict(os.environ, RUNNER_TEMP=str(directory), GITHUB_OUTPUT=str(output),
                           AERIE_BENCH_SIDE='head', AERIE_BENCH_REVISION='test-revision',
                           RUSTFLAGS='', PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
                subprocess.run(['bash', '-c', self.schema_shell()], cwd=directory, env=env,
                               text=True, capture_output=True, check=True)
                outputs = dict(line.split('=', 1) for line in output.read_text().splitlines())
                shell = cache_shell.replace('${{ steps.schema.outputs.expected }}', outputs['expected'])
                subprocess.run(['bash', '-c', shell], cwd=directory, env=env,
                               text=True, capture_output=True, check=True)
                outputs = dict(line.split('=', 1) for line in output.read_text().splitlines())
                self.assertTrue(outputs['binary'].startswith(f'falcon-bench-bin-schema{schema}-test-revision-'))
                keys.append(outputs['binary'])
        self.assertNotEqual(*keys)

    def test_fixture_cache_key_includes_both_harness_paths(self):
        repo = Path(__file__).resolve().parents[2]
        workflow = (repo / '.github/workflows/falcon-bench.yml').read_text()
        key = next(line for line in workflow.splitlines() if 'key: falcon-bench-fixtures-' in line)
        for package in ['aerie-bench', 'aerie']:
            self.assertIn(f"'crates/{package}/benches/falcon.rs'", key)


if __name__ == '__main__':
    unittest.main()
