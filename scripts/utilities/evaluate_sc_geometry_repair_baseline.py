#!/usr/bin/env python3
"""Evaluate the materialized 46k EMA on the fixed 308-protein repair panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibration-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = Path(args.calibration_dir).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    import train_sc_adaptation as driver
    from pxdesign_train.runner.sc_stream import atomic_json, seed_all, sha256_file
    from pxdesign_train.runner.trainer import PXDesignTrainer

    options = driver.parser().parse_args([
        '--accepted-checkpoint', args.checkpoint, '--phase', 'sc_geometry_repair',
        '--donor-weights', 'ema', '--repair-arm', 'B', '--output-dir', str(output),
        '--calibration-path', str(calibration / 'calibration.yaml'),
        '--source-index', str(calibration / 'train.csv.gz'),
        '--eval-source-index', str(calibration / 'validation.csv.gz'),
        '--final-test-index', str(calibration / 'final_test.csv.gz'),
        '--eval-samples', '491', '--num-workers', '4'])
    config, recipe = driver.resolve(options)
    seed_all(config.seed)
    components = driver.build_data(config, recipe, output)
    trainer = PXDesignTrainer(config, components, device=torch.device(args.device),
        checkpoint_dir=str(output / 'checkpoints'))
    if trainer.step != 0:
        raise ValueError('Donor baseline must be evaluated before any repair update')
    metrics = trainer.evaluate()
    validation_path = output / 'validation-step0.json'
    validation = json.loads(validation_path.read_text())
    validation['weights'] = 'materialized_46k_ema'
    atomic_json(validation_path, validation)
    report = dict(schema='sc_geometry_repair_donor_baseline_v1', source_step=46000,
        source_checkpoint=str(Path(args.checkpoint).resolve()),
        source_checkpoint_sha256=sha256_file(args.checkpoint), weights='ema',
        calibration_sha256=sha256_file(calibration / 'calibration.yaml'),
        validation_sha256=sha256_file(validation_path), metrics=metrics,
        proteins=trainer.last_eval_per_protein)
    atomic_json(output / 'donor_baseline.json', report)
    print(json.dumps({key: report[key] for key in report if key not in ('metrics', 'proteins')}, indent=2))


if __name__ == '__main__':
    main()
