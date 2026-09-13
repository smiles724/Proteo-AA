#!/usr/bin/env python3
"""Screen the preregistered D/E sweep or select its post-extension checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_sc_geometry_repair import (  # noqa: E402
    PREFIX, TERMS, covalent_failure, load_steps, metric, pooled_metric, sha256)


PREREG_SCHEMA = 'sc_geometry_repair_strength_sweep_v2_preregistration'
SCREEN_SCHEMA = 'sc_geometry_repair_strength_sweep_v2_screen'


def load_preregistration(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get('schema') != PREREG_SCHEMA or not value.get('registered_before_training'):
        raise ValueError('Expected the preregistered v2 geometry-strength gate')
    return path, value


def fixed_context(prereg):
    control_run, control_steps = load_steps(prereg['control']['run'])
    if sha256(control_run / 'data_audit.json') != prereg['control']['data_audit_sha256']:
        raise ValueError('Arm B data identity changed after preregistration')
    expected_steps = {500, 1000, 1500, 2000}
    if set(control_steps) != expected_steps:
        raise ValueError('Arm B must provide saved 500/1000/1500/2000 controls')
    for step, digest in prereg['control']['validation_sha256'].items():
        if sha256(control_steps[int(step)][0]) != digest:
            raise ValueError(f'Arm B step {step} validation changed after preregistration')
    baseline_path = Path(prereg['donor_baseline']['path']).resolve()
    if sha256(baseline_path) != prereg['donor_baseline']['sha256']:
        raise ValueError('Donor baseline changed after preregistration')
    baseline = json.loads(baseline_path.read_text())
    if (baseline.get('schema') != 'sc_geometry_repair_donor_baseline_v1'
            or baseline.get('source_checkpoint_sha256') != prereg['source_checkpoint_sha256']
            or baseline.get('weights') != 'ema' or len(baseline.get('proteins', ())) != 308):
        raise ValueError('Invalid 46k EMA donor baseline')
    weights_path = Path(prereg['base_gradient_weights']['path']).resolve()
    if sha256(weights_path) != prereg['base_gradient_weights']['sha256']:
        raise ValueError('Base gradient weights changed after preregistration')
    donor_ids = tuple((row.get('source'), row['sample_id'], row['index'])
                      for row in baseline['proteins'])
    return control_run, control_steps, baseline['metrics'], donor_ids


def validate_arm(run, arm, prereg, control_run, *, extended=False):
    run, steps = load_steps(run)
    config = json.loads((run / 'resolved_config.json').read_text())
    recipe = config['training']['sc_adaptation_recipe']
    expected = prereg['arms'][arm]['weights']
    if recipe['repair_arm'] != arm or not config['stage4']['symmetry_aware_coordinates']:
        raise ValueError(f'Arm {arm} is not the preregistered symmetry-aware arm')
    for name, value in expected.items():
        if not math.isclose(float(config['stage4']['weight_' + name]), value, rel_tol=0., abs_tol=1e-15):
            raise ValueError(f'Arm {arm} weight differs for {name}')
    fixed = prereg['fixed_recipe']
    checks = {
        'seed': config['seed'],
        'sc_lr': config['stage4']['sc_lr'],
        'warmup_steps': config['training']['warmup_steps'],
        'geometry_ramp_steps': config['stage4']['geometry_ramp_steps'],
        'accumulation': config['training']['iters_to_accumulate'],
        'eval_interval': config['training']['eval_interval'],
        'checkpoint_interval': config['training']['checkpoint_interval'],
    }
    for name, actual in checks.items():
        if actual != fixed[name]:
            raise ValueError(f'Arm {arm} differs from preregistered {name}')
    if float(config['stage4']['weight_physical']) != 0.:
        raise ValueError('The v2 sweep must not enable clash/physical loss')
    expected_budget = prereg['extension']['max_steps'] if extended else fixed['max_steps']
    if int(config['training']['max_steps']) != expected_budget:
        raise ValueError(f'Arm {arm} has the wrong step budget')
    control_identity = json.loads((control_run / 'data_audit.json').read_text())['identity']
    arm_identity = json.loads((run / 'data_audit.json').read_text())['identity']
    if arm_identity != control_identity:
        raise ValueError(f'Arm {arm} data, seeds, or validation panel differ from Arm B')
    return run, steps


def assess(metrics, ids, control_metrics, control_ids, donor_metrics, donor_ids, prereg):
    if ids != control_ids or ids != donor_ids:
        raise ValueError('Validation samples or stochastic inputs differ from the controls')
    failure, classes = covalent_failure(metrics)
    control_failure, control_classes = covalent_failure(control_metrics)
    reduction = (control_failure - failure) / max(control_failure, 1e-12)
    class_conditions = {name: dict(passed=classes[name]['rate'] < control_classes[name]['rate'],
        B=control_classes[name]['rate'], candidate=classes[name]['rate'],
        delta=classes[name]['rate']-control_classes[name]['rate']) for name in TERMS}
    chi1 = pooled_metric(metrics, PREFIX + 'all/chi1_accuracy_20deg')
    control_chi1 = pooled_metric(control_metrics, PREFIX + 'all/chi1_accuracy_20deg')
    joint = pooled_metric(metrics, PREFIX + 'all/chi1_chi2_accuracy_20deg')
    control_joint = pooled_metric(control_metrics, PREFIX + 'all/chi1_chi2_accuracy_20deg')
    symmetry = metric(metrics, PREFIX + 'sc_symmetry_rmsd')
    donor_symmetry = metric(donor_metrics, PREFIX + 'sc_symmetry_rmsd')
    max_symmetry = donor_symmetry + float(prereg['screen_gate'][
        'symmetry_rmsd_per_protein_mean_max_delta_vs_donor_angstrom'])
    passed = (reduction >= float(prereg['screen_gate']['aggregate_failure_reduction_vs_matched_B_min'])
              and all(row['passed'] for row in class_conditions.values())
              and chi1 >= control_chi1 and joint >= control_joint
              and symmetry <= max_symmetry)
    return dict(passed=passed, aggregate_failure_rate=failure,
        control_failure_rate=control_failure, aggregate_reduction_vs_B=reduction,
        classes=classes, control_classes=control_classes,
        class_conditions=class_conditions,
        chi1_accuracy_20deg=dict(B=control_chi1, candidate=chi1, passed=chi1 >= control_chi1),
        joint_chi1_chi2_accuracy_20deg=dict(B=control_joint, candidate=joint,
            passed=joint >= control_joint),
        symmetry_rmsd=dict(donor=donor_symmetry, candidate=symmetry,
            maximum=max_symmetry, passed=symmetry <= max_symmetry))


def checkpoint_for(run, step):
    path = run / 'checkpoints' / f'step{step}.pt'
    if not path.is_file():
        raise ValueError(f'Missing checkpoint for validation step {step}: {path}')
    return path


def screen(args):
    prereg_path, prereg = load_preregistration(args.preregistration)
    control_run, controls, donor_metrics, donor_ids = fixed_context(prereg)
    candidates = []
    arm_runs = {}
    for arm, supplied in (('D', args.arm_d), ('E', args.arm_e)):
        run, steps = validate_arm(supplied, arm, prereg, control_run)
        arm_runs[arm] = run
        if set(steps) != {500, 1000, 1500, 2000}:
            raise ValueError(f'Arm {arm} did not complete all preregistered validations')
        for step in sorted(steps):
            path, metrics, ids = steps[step]
            control_path, control_metrics, control_ids = controls[step]
            row = assess(metrics, ids, control_metrics, control_ids,
                         donor_metrics, donor_ids, prereg)
            candidates.append(dict(arm=arm, step=step, **row,
                validation_file=str(path), control_validation_file=str(control_path)))
    passing = [row for row in candidates if row['passed']]
    selected = min(passing, key=lambda row: (row['aggregate_failure_rate'], -row['step'], row['arm'])) if passing else None
    output = dict(schema=SCREEN_SCHEMA, passed=selected is not None,
        preregistration=dict(path=str(prereg_path), sha256=sha256(prereg_path)),
        runs={key:str(value) for key,value in arm_runs.items()}, candidates=candidates)
    if selected:
        checkpoint = checkpoint_for(arm_runs[selected['arm']], selected['step'])
        output['selected'] = dict(arm=selected['arm'], step=selected['step'],
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            validation_sha256=sha256(selected['validation_file']))
    write_output(args.output, output)
    raise SystemExit(0 if output['passed'] else 2)


def final_select(args):
    prereg_path, prereg = load_preregistration(args.preregistration)
    control_run, controls, donor_metrics, donor_ids = fixed_context(prereg)
    screen_path = Path(args.screen).resolve()
    screen_value = json.loads(screen_path.read_text())
    if (screen_value.get('schema') != SCREEN_SCHEMA or not screen_value.get('passed')
            or screen_value.get('preregistration', {}).get('sha256') != sha256(prereg_path)):
        raise ValueError('Extension selection requires a passing v2 screen for this preregistration')
    arm = screen_value['selected']['arm']
    original_run, original = validate_arm(screen_value['runs'][arm], arm, prereg, control_run)
    extension_run, extension = validate_arm(args.extension, arm, prereg, control_run, extended=True)
    extension_config = json.loads((extension_run / 'resolved_config.json').read_text())
    if (Path(extension_config['training']['resume_checkpoint']).resolve()
            != Path(screen_value['selected']['checkpoint']).resolve()):
        raise ValueError('The 5k extension did not resume the screened checkpoint')
    if 5000 not in extension:
        raise ValueError('The selected arm extension did not reach step 5000 validation')
    candidates_by_step = dict(original)
    sources = {step: original_run for step in original}
    for step, value in extension.items():
        if step in original and (value[1] != original[step][1] or value[2] != original[step][2]):
            raise ValueError(f'Extension replay differs from the original arm at step {step}')
        candidates_by_step[step] = value
        sources[step] = extension_run
    candidates = []
    for step in sorted(candidates_by_step):
        path, metrics, ids = candidates_by_step[step]
        control_step = step if step <= 2000 else 2000
        control_path, control_metrics, control_ids = controls[control_step]
        row = assess(metrics, ids, control_metrics, control_ids,
                     donor_metrics, donor_ids, prereg)
        candidates.append(dict(arm=arm, step=step, control_step=control_step, **row,
            validation_file=str(path), control_validation_file=str(control_path)))
    passing = [row for row in candidates if row['passed']]
    selected = min(passing, key=lambda row: (row['aggregate_failure_rate'], -row['step'])) if passing else None
    output = dict(schema='sc_geometry_repair_acceptance_v2', approved=selected is not None,
        preregistration=dict(path=str(prereg_path), sha256=sha256(prereg_path)),
        screen=dict(path=str(screen_path), sha256=sha256(screen_path)),
        selected_arm=arm, extension_run=str(extension_run), candidates=candidates,
        criteria=prereg['screen_gate'], control_policy=prereg['extension']['control_policy'],
        final_test_used=False)
    if selected:
        checkpoint = checkpoint_for(sources[selected['step']], selected['step'])
        output['selected'] = dict(arm=arm, step=selected['step'], checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            validation_sha256=sha256(selected['validation_file']))
    write_output(args.output, output)
    raise SystemExit(0 if output['approved'] else 2)


def write_output(path, value):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    first = subparsers.add_parser('screen')
    first.add_argument('--preregistration', required=True)
    first.add_argument('--arm-b', required=True)
    first.add_argument('--arm-d', required=True)
    first.add_argument('--arm-e', required=True)
    first.add_argument('--donor-baseline', required=True)
    first.add_argument('--output', required=True)
    first.set_defaults(action=screen)
    final = subparsers.add_parser('final')
    final.add_argument('--preregistration', required=True)
    final.add_argument('--screen', required=True)
    final.add_argument('--arm-b', required=True)
    final.add_argument('--donor-baseline', required=True)
    final.add_argument('--extension', required=True)
    final.add_argument('--output', required=True)
    final.set_defaults(action=final_select)
    args = parser.parse_args()
    # CLI paths must agree with the immutable preregistration; explicit arguments
    # make scheduler logs self-contained without allowing a different control.
    _, prereg = load_preregistration(args.preregistration)
    if Path(args.arm_b).resolve() != Path(prereg['control']['run']).resolve():
        raise ValueError('--arm-b differs from preregistered control')
    if Path(args.donor_baseline).resolve() != Path(prereg['donor_baseline']['path']).resolve():
        raise ValueError('--donor-baseline differs from preregistered baseline')
    args.action(args)


if __name__ == '__main__':
    main()
