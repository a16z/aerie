#!/usr/bin/env python3
"""Resolve the Falcon Cargo benchmark target, feature, package and harness."""
import argparse
import json
from pathlib import Path
import sys


def package_metadata(metadata, package):
    matches = [item for item in metadata['packages'] if item['name'] == package]
    if len(matches) != 1:
        raise ValueError('Expected exactly one Cargo package: ' + package)
    return matches[0]


def select_falcon_benchmark(metadata, side='head'):
    # A base revision may keep the harness in the aerie package.
    name = 'aerie-bench'
    if side == 'base' and not any(item['name'] == name for item in metadata['packages']):
        name = 'aerie'
    package = package_metadata(metadata, name)
    targets = [target for target in package['targets']
               if target['name'] == 'falcon' and 'bench' in target['kind']]
    if len(targets) != 1:
        raise ValueError('No falcon benchmark target in ' + name)
    if 'falcon-bench' not in package['features']:
        raise ValueError('No falcon-bench feature in ' + name)
    harness = Path(targets[0]['src_path']).relative_to(metadata['workspace_root']).as_posix()
    return 'falcon', 'falcon-bench', name, harness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['falcon'])
    parser.add_argument('--side', choices=['base', 'head'], default='head')
    args = parser.parse_args()
    selection = select_falcon_benchmark(json.load(sys.stdin), side=args.side)
    print('\t'.join(selection))


if __name__ == '__main__':
    main()
