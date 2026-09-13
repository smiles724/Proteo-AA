#!/usr/bin/env python3
"""Select a repair checkpoint only when covalent failures fall and torsions hold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TERMS = ('bond_sc', 'bond_attach', 'angle_sc', 'angle_attach')
PREFIX = 'native/monomer_retention/'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_steps(run):
    run = Path(run).resolve()
    rows = {}
    for path in run.glob('validation-step*.json'):
        value = json.loads(path.read_text())
        step = int(value['step'])
        if len(value.get('proteins', ())) != 308:
            raise ValueError(f'{path}: expected the fixed 308-protein validation panel')
        sample_ids = tuple((row.get('source'), row['sample_id'], row['index']) for row in value['proteins'])
        rows[step] = (path, value['metrics'], sample_ids)
    if not rows:
        raise ValueError(f'{run}: no validation-step*.json files')
    return run, rows


def metric(metrics, name):
    if name in metrics:
        return float(metrics[name])
    if name + '_pooled' in metrics:
        return float(metrics[name + '_pooled'])
    raise ValueError(f'Missing validation metric {name}')


def pooled_metric(metrics, name):
    """Use the explicitly pooled metric when both reductions are logged."""
    if name + '_pooled' in metrics:
        return float(metrics[name + '_pooled'])
    return metric(metrics, name)


def covalent_failure(metrics):
    failed = eligible = 0.
    by_class = {}
    for term in TERMS:
        base = PREFIX + f'geometry/model/{term}/'
        count = metric(metrics, base + 'outliers_3rms_count')
        total = metric(metrics, base + 'eligible_constraints')
        if total <= 0:
            raise ValueError(f'No eligible {term} constraints')
        failed += count
        eligible += total
        by_class[term] = dict(failed=count, eligible=total, rate=count / total)
    return failed / eligible, by_class


def classwise_acceptance(donor, control, candidate, max_regression):
    """Require internal bonds to improve and bound every other class regression."""
    conditions = {}
    for term in TERMS:
        candidate_rate = candidate[term]['rate']
        donor_rate = donor[term]['rate']
        control_rate = control[term]['rate']
        if term == 'bond_sc':
            passed = candidate_rate < donor_rate and candidate_rate < control_rate
        else:
            passed = (candidate_rate <= donor_rate + max_regression
                      and candidate_rate <= control_rate + max_regression)
        conditions[term] = dict(passed=passed, donor=donor_rate, B=control_rate,
            C=candidate_rate, delta_vs_donor=candidate_rate-donor_rate,
            delta_vs_B=candidate_rate-control_rate)
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-a', required=True)
    parser.add_argument('--arm-b', required=True)
    parser.add_argument('--arm-c', required=True)
    parser.add_argument('--donor-baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-covalent-relative-reduction', type=float, default=.25)
    parser.add_argument('--max-class-absolute-regression', type=float, default=.02)
    parser.add_argument('--max-chi1-drop', type=float, default=.02)
    parser.add_argument('--max-joint-chi12-drop', type=float, default=.02)
    args = parser.parse_args()
    arm_a, a = load_steps(args.arm_a)
    arm_b, b = load_steps(args.arm_b)
    arm_c, c = load_steps(args.arm_c)
    baseline_path = Path(args.donor_baseline).resolve()
    baseline = json.loads(baseline_path.read_text())
    if (baseline.get('schema') != 'sc_geometry_repair_donor_baseline_v1'
            or baseline.get('source_step') != 46000 or baseline.get('weights') != 'ema'):
        raise ValueError('Expected a fixed 46k EMA donor baseline')
    if len(baseline.get('proteins', ())) != 308:
        raise ValueError('Donor baseline must contain the fixed 308-protein panel')
    donor_metrics = baseline['metrics']
    donor_ids = tuple((row.get('source'), row['sample_id'], row['index']) for row in baseline['proteins'])
    donor_failure, donor_classes = covalent_failure(donor_metrics)
    candidates = []
    for step in sorted(set(a) & set(b) & set(c)):
        apath, am, aids = a[step]
        bpath, bm, bids = b[step]
        cpath, cm, cids = c[step]
        if aids != bids or aids != cids or aids != donor_ids:
            raise ValueError(f'Step {step}: stochastic validation inputs differ across arms')
        a_failure, a_classes = covalent_failure(am)
        b_failure, b_classes = covalent_failure(bm)
        c_failure, c_classes = covalent_failure(cm)
        reduction = (b_failure - c_failure) / max(b_failure, 1e-12)
        b_chi1 = pooled_metric(bm, PREFIX + 'all/chi1_accuracy_20deg')
        c_chi1 = pooled_metric(cm, PREFIX + 'all/chi1_accuracy_20deg')
        b_joint = pooled_metric(bm, PREFIX + 'all/chi1_chi2_accuracy_20deg')
        c_joint = pooled_metric(cm, PREFIX + 'all/chi1_chi2_accuracy_20deg')
        class_conditions = classwise_acceptance(
            donor_classes, b_classes, c_classes, args.max_class_absolute_regression)
        passed = (reduction >= args.min_covalent_relative_reduction
                  and all(row['passed'] for row in class_conditions.values())
                  and c_chi1 >= b_chi1 - args.max_chi1_drop
                  and c_joint >= b_joint - args.max_joint_chi12_drop)
        candidates.append(dict(step=step, passed=passed,
            covalent_failure_rate=dict(donor_46k_ema=donor_failure,A=a_failure, B=b_failure, C=c_failure),
            covalent_relative_reduction_C_vs_B=reduction,
            classes=dict(donor_46k_ema=donor_classes,A=a_classes, B=b_classes, C=c_classes),
            class_conditions=class_conditions,
            chi1_accuracy_20deg=dict(B=b_chi1, C=c_chi1),
            joint_chi1_chi2_accuracy_20deg=dict(B=b_joint, C=c_joint),
            symmetry_rmsd=dict(A=metric(am, PREFIX+'sc_symmetry_rmsd'),
                               B=metric(bm, PREFIX+'sc_symmetry_rmsd'),
                               C=metric(cm, PREFIX+'sc_symmetry_rmsd')),
            ordinary_rmsd=dict(A=metric(am, PREFIX+'sc_gt_rmsd'),
                               B=metric(bm, PREFIX+'sc_gt_rmsd'),
                               C=metric(cm, PREFIX+'sc_gt_rmsd')),
            validation_files=dict(A=str(apath), B=str(bpath), C=str(cpath))))
    passing = [row for row in candidates if row['passed']]
    selected = min(passing, key=lambda row: (row['covalent_failure_rate']['C'], -row['step'])) if passing else None
    output = dict(schema='sc_geometry_repair_acceptance_v1', approved=selected is not None,
        criteria=dict(min_covalent_relative_reduction=args.min_covalent_relative_reduction,
            bond_sc='C must improve strictly against both 46k EMA and arm B',
            max_class_absolute_regression=args.max_class_absolute_regression,
            max_chi1_drop=args.max_chi1_drop,max_joint_chi12_drop=args.max_joint_chi12_drop,
            control='arm B and 46k EMA on identical fixed stochastic inputs',final_test_used=False),
        donor_baseline=dict(path=str(baseline_path),sha256=sha256(baseline_path),
            source_checkpoint_sha256=baseline['source_checkpoint_sha256']),
        runs=dict(A=str(arm_a), B=str(arm_b), C=str(arm_c)), candidates=candidates)
    if selected:
        checkpoint = arm_c / 'checkpoints' / f"step{selected['step']}.pt"
        if not checkpoint.is_file():
            raise ValueError(f'Selected validation has no checkpoint: {checkpoint}')
        output['selected'] = dict(step=selected['step'], checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint), validation_sha256=sha256(selected['validation_files']['C']))
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output['approved'] else 2)


if __name__ == '__main__':
    main()
