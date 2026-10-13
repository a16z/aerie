#!/usr/bin/env python3
"""Resolve current and archived Cargo benchmark names without changing protocols."""
import argparse
import json
import sys

def package_metadata(metadata, package):
    matches = [item for item in metadata['packages'] if item['name'] == package]
    if len(matches) != 1:
        raise ValueError('Expected exactly one Cargo package: ' + package)
    return matches[0]


def select_target(package, kind, names):
    targets = {target['name'] for target in package['targets'] if kind in target['kind']}
    for name in names:
        if name in targets:
            return name
    raise ValueError('No supported ' + kind + ' target in ' + package['name'])


def select_falcon_benchmark(metadata):
    """Choose target and feature independently for each head/base checkout."""
    package = package_metadata(metadata, 'aerie')
    target = select_target(package, 'bench', ('falcon', 'falcon-v1'))
    features = package['features']
    for feature in ('falcon-bench', 'falcon-v1-bench'):
        if feature in features:
            return target, feature
    if {'parallel', 'honest-vectors'}.issubset(features):
        return target, 'parallel,honest-vectors'
    raise ValueError('No supported Falcon benchmark feature set')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['falcon'])
    parser.parse_args()
    target, features = select_falcon_benchmark(json.load(sys.stdin))
    print(target + '\t' + features)


if __name__ == '__main__':
    main()
