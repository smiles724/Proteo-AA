#!/usr/bin/env python3
"""Evaluate an accepted repair checkpoint once on the reserved final-test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    acceptance_path = Path(args.acceptance).resolve()
    acceptance = json.loads(acceptance_path.read_text())
    if (acceptance.get('schema') != 'sc_geometry_repair_acceptance_v1'
            or not acceptance.get('approved')):
        raise ValueError('Final-test evaluation requires an approved repair selection')

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = Path(acceptance['selected']['checkpoint']).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    import train_sc_adaptation as driver
    import train_protenix_monomer as base
    from pxdesign_train.runner.sc_stream import CoordinatePanel, atomic_json, seed_all, sha256_file
    from pxdesign_train.runner.trainer import PXDesignTrainer

    if sha256_file(selected) != acceptance['selected']['checkpoint_sha256']:
        raise ValueError('Selected repair checkpoint changed after acceptance')
    options = driver.parser().parse_args([
        '--resume-checkpoint', str(selected), '--output-dir', str(output),
        '--num-workers', str(args.num_workers)])
    config, recipe = driver.resolve(options)
    if recipe['phase'] != 'sc_geometry_repair':
        raise ValueError('Selected checkpoint is not a geometry-repair checkpoint')
    seed_all(config.seed)
    components = driver.build_data(config, recipe, output)

    final_manifest = Path(recipe['final_test_index']).resolve()
    eval_args = driver.legacy_arguments(recipe, output / 'final_panel')
    eval_args.eval_source_index = str(final_manifest)
    eval_args.eval_filtered_index = str(output / 'cache' / 'final_test.csv.gz')
    eval_args.eval_samples = 1000000
    eval_args.eval_num_workers = args.num_workers
    eval_args.rebuild_eval_index = True
    loader, count, filtered = base.build_eval_dataloader(eval_args, output / 'final_panel')
    if loader is None or count != 128:
        raise ValueError(f'Expected the frozen 128-protein final test, got {count}')
    components.eval_dataloader = None
    components.named_eval_dataloaders = {
        'native/final_test': CoordinatePanel(loader, 'native', seed=2000003)}

    trainer = PXDesignTrainer(config, components, device=torch.device(args.device),
        checkpoint_dir=str(output / 'checkpoints'))
    if trainer.step != int(acceptance['selected']['step']):
        raise ValueError('Loaded repair step differs from the accepted step')
    metrics = trainer.evaluate()
    if len(trainer.last_eval_per_protein) != 128:
        raise ValueError('Final-test evaluator did not score all 128 proteins')
    report = dict(schema='sc_geometry_repair_final_test_v1', completed=True,
        selected_step=trainer.step, selected_checkpoint=str(selected),
        selected_checkpoint_sha256=sha256_file(selected),
        acceptance_path=str(acceptance_path), acceptance_sha256=sha256_file(acceptance_path),
        final_test_manifest=str(final_manifest),
        final_test_manifest_sha256=sha256_file(final_manifest),
        filtered_final_test_sha256=sha256_file(filtered), count=count,
        input_seed=2000003, selection_metrics_used=False,
        metrics=metrics, proteins=trainer.last_eval_per_protein)
    target = output / 'final_test.json'
    atomic_json(target, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('metrics', 'proteins')}, indent=2))


if __name__ == '__main__':
    main()
