#!/usr/bin/env python3
"""Apply the unchanged v2 screen gate to preregistered Arm F."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from select_sc_geometry_repair_v2 import (
    SCREEN_SCHEMA,
    assess,
    checkpoint_for,
    fixed_context,
    load_preregistration,
    sha256,
    validate_arm,
    write_output,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', required=True)
    parser.add_argument('--arm-b', required=True)
    parser.add_argument('--arm-f', required=True)
    parser.add_argument('--donor-baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    prereg_path, prereg = load_preregistration(args.preregistration)
    if set(prereg.get('arms', ())) != {'F'}:
        raise ValueError('Arm F preregistration must contain exactly Arm F')
    if Path(args.arm_b).resolve() != Path(prereg['control']['run']).resolve():
        raise ValueError('--arm-b differs from preregistered control')
    if Path(args.donor_baseline).resolve() != Path(prereg['donor_baseline']['path']).resolve():
        raise ValueError('--donor-baseline differs from preregistered baseline')

    control_run, controls, donor_metrics, donor_ids = fixed_context(prereg)
    run, steps = validate_arm(args.arm_f, 'F', prereg, control_run)
    if set(steps) != {500, 1000, 1500, 2000}:
        raise ValueError('Arm F did not complete all preregistered validations')
    candidates = []
    for step in sorted(steps):
        path, metrics, ids = steps[step]
        control_path, control_metrics, control_ids = controls[step]
        row = assess(metrics, ids, control_metrics, control_ids,
                     donor_metrics, donor_ids, prereg)
        candidates.append(dict(arm='F', step=step, **row,
            validation_file=str(path), control_validation_file=str(control_path)))
    passing = [row for row in candidates if row['passed']]
    selected = min(passing, key=lambda row: (row['aggregate_failure_rate'], -row['step'])) if passing else None
    output = dict(schema=SCREEN_SCHEMA, passed=selected is not None,
        preregistration=dict(path=str(prereg_path), sha256=sha256(prereg_path)),
        runs={'F': str(run)}, candidates=candidates)
    if selected:
        checkpoint = checkpoint_for(run, selected['step'])
        output['selected'] = dict(arm='F', step=selected['step'],
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            validation_sha256=sha256(selected['validation_file']))
    write_output(args.output, output)
    raise SystemExit(0 if output['passed'] else 2)


if __name__ == '__main__':
    main()
