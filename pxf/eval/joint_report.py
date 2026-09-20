"""Aggregation, paired contrasts and the verdict for the joint evaluator.

Three things here are easy to get quietly wrong, so each is explicit.

*Direction.* Every paired table reports "positive means the candidate is
better". That means subtracting in opposite orders for an RMSD and for an
lDDT, so the direction of each metric is declared in a table rather than
guessed from its name -- ``geom_valid_frame_rate`` and ``geom_c_n_bad_rate``
are both rates and point opposite ways.

*The estimand.* The unit is the protein. Replicates collapse inside an event,
events collapse inside a (target, sigma), sigmas are weighted equally, and only
then is a target a single number. Resampling rows instead would treat one
structure measured at four sigmas as four independent observations and report
an interval roughly half as wide as it should be.

*What a number is not.* A nonsignificant difference is not "no regression", a
comparison missing its baseline has no verdict, and an intersection left over
after one candidate failed is not a complete run.
"""

import json
import math
import statistics
from pathlib import Path

# Which way is better. Declared, never inferred: two metrics ending in "_rate"
# point in opposite directions, and a wrong sign silently reverses a verdict.
LOWER_IS_BETTER = {
    "bb_ca_rmsd",
    "bb_backbone_rmsd",
    "bb_rmsd_unaligned",
    "geom_n_ca_bad_rate",
    "geom_ca_c_bad_rate",
    "geom_c_o_bad_rate",
    "geom_c_n_bad_rate",
    "sc_local_rmsd",
    "sc_global_rmsd_bb_aligned",
    "sc_chemistry_clash_rate",
    "sc_chemistry_bad_bond_rate",
}
HIGHER_IS_BETTER = {
    "bb_lddt",
    "bb_lddt_ca",
    "bb_lddt_bb",
    "bb_tm",
    "bb_tm_score",
    "geom_valid_frame_rate",
    "sc_local_rotamer_recovery",
    "sc_local_chi_recovery",
    "sc_environment_lddt",
}

# The practical gate this project already uses. A threshold for deciding to
# spend more compute, not a definition of statistical significance.
BB_GATE_ABSOLUTE = 0.05      # Angstroms
BB_GATE_RELATIVE = 0.03      # fraction of the matched B0 RMSD
# Packing-safety margins, carried forward from the plan as explicit
# configurable values rather than inherited from the old evaluator's settings.
SAFETY_MARGINS = {
    "sc_local_rmsd": 0.03,                 # Angstroms, candidate may be worse by
    "sc_local_rotamer_recovery": 0.01,     # fraction (1 percentage point)
    "sc_chemistry_bad_bond_rate": 0.005,   # fraction (0.5 percentage points)
}
BOOTSTRAP_RESAMPLES = 10000

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INCOMPLETE = "incomplete"
STATUS_NOT_REQUESTED = "not_requested"


class ReportError(RuntimeError):
    """A comparison that cannot be formed as asked."""


def metric_direction(metric):
    if metric in LOWER_IS_BETTER:
        return -1
    if metric in HIGHER_IS_BETTER:
        return +1
    raise ReportError(
        f"metric {metric!r} has no declared direction; add it to "
        "LOWER_IS_BETTER or HIGHER_IS_BETTER rather than letting the sign be "
        "inferred from the name"
    )


def signed_improvement(metric, candidate, reference):
    """Positive means ``candidate`` is better, whichever way the metric runs."""
    if candidate is None or reference is None:
        return None
    return metric_direction(metric) * (float(candidate) - float(reference))


# ---- aggregation -------------------------------------------------------------


def _mean(values):
    present = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(present) if present else None


def protein_values(rows, metric, *, sigmas=None):
    """One value per (model label, sample), via the declared estimand.

    Collapses replicates inside an event, then events inside a (target, sigma),
    then weights the requested sigmas equally. A sigma the panel did not
    produce for a given target makes that target incomplete for this metric and
    it is dropped, rather than silently reweighting the ones that are present.
    """
    by_event = {}
    for row in rows:
        if row.get("failure"):
            continue
        key = (row["model"], row["sample_id"], row["sigma_key"])
        by_event.setdefault(key, []).append(row.get(metric))

    wanted = None if sigmas is None else [str(s) for s in sigmas]
    by_target = {}
    for (model, sample, sigma), values in by_event.items():
        if wanted is not None and sigma not in wanted:
            continue
        collapsed = _mean(values)
        if collapsed is None:
            continue
        by_target.setdefault((model, sample), {})[sigma] = collapsed

    out = {}
    for (model, sample), per_sigma in by_target.items():
        if wanted is not None and set(per_sigma) != set(wanted):
            continue  # incomplete coverage for this target; excluded and counted
        out[(model, sample)] = _mean(per_sigma.values())
    return out


def coverage(rows, *, sigmas=None):
    """Per-model event counts and failure counts, for the coverage table."""
    out = {}
    for row in rows:
        entry = out.setdefault(
            row["model"], dict(events=0, failures=0, samples=set(), sigmas=set())
        )
        entry["events"] += 1
        if row.get("failure"):
            entry["failures"] += 1
        entry["samples"].add(row["sample_id"])
        entry["sigmas"].add(row["sigma_key"])
    for entry in out.values():
        entry["n_samples"] = len(entry.pop("samples"))
        entry["sigmas"] = sorted(entry["sigmas"])
    return out


# ---- paired contrasts --------------------------------------------------------


def _resample_indices(n, n_resamples, seed):
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(n, (int(n_resamples), n), generator=generator).tolist()


def paired_contrast(
    rows,
    *,
    candidate,
    reference,
    metric,
    sigmas=None,
    clusters=None,
    n_resamples=BOOTSTRAP_RESAMPLES,
    seed=0,
    alpha=0.05,
):
    """Protein-balanced paired improvement of ``candidate`` over ``reference``.

    Resamples whole proteins, or whole clusters when labels are supplied -- a
    cluster draw takes every member target from both arms together, so homology
    between targets cannot masquerade as independent evidence. Without cluster
    labels the interval is protein-level and is labelled as possibly optimistic
    rather than quietly presented as if homology had been handled.
    """
    values = protein_values(rows, metric, sigmas=sigmas)
    samples = sorted(
        {sample for (model, sample) in values if model in (candidate, reference)}
    )
    paired = []
    for sample in samples:
        a = values.get((candidate, sample))
        b = values.get((reference, sample))
        delta = signed_improvement(metric, a, b)
        if delta is not None:
            paired.append((sample, delta))

    if not paired:
        return dict(
            candidate=candidate,
            reference=reference,
            metric=metric,
            status=STATUS_INCOMPLETE,
            reason="no target has a paired value for both arms",
            n_targets=0,
            mean=None,
            ci_low=None,
            ci_high=None,
        )

    names = [s for s, _ in paired]
    deltas = [d for _, d in paired]
    point = statistics.fmean(deltas)

    if clusters:
        groups = {}
        for name, delta in paired:
            groups.setdefault(clusters.get(name, name), []).append(delta)
        units = list(groups.values())
        unit_kind = "cluster"
    else:
        units = [[d] for d in deltas]
        unit_kind = "protein"

    draws = _resample_indices(len(units), n_resamples, seed)
    means = []
    for draw in draws:
        pooled = []
        for index in draw:
            pooled.extend(units[index])
        # Recompute the declared estimate inside the draw, not a mean of means.
        means.append(statistics.fmean(pooled))
    means.sort()
    low = means[int((alpha / 2) * len(means))]
    high = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]

    return dict(
        candidate=candidate,
        reference=reference,
        metric=metric,
        estimand="protein-balanced mean of within-protein improvement",
        sign_convention="positive means the candidate is better",
        mean=point,
        ci_low=low,
        ci_high=high,
        alpha=alpha,
        n_targets=len(names),
        n_units=len(units),
        unit=unit_kind,
        homology_caveat=None if clusters else "no cluster labels; interval may be optimistic",
        n_resamples=int(n_resamples),
        bootstrap_seed=int(seed),
        sigmas=None if sigmas is None else list(sigmas),
        status=STATUS_PASS,
    )


# ---- verdict -----------------------------------------------------------------


def backbone_gate(contrast, reference_rmsd, *, absolute=BB_GATE_ABSOLUTE,
                  relative=BB_GATE_RELATIVE):
    """The project's practical gate on backbone improvement.

    Requires a positive improvement whose interval excludes zero and whose
    point estimate clears max(absolute, relative x matched baseline). Stated as
    a practical threshold: clearing it is a decision to spend more compute, not
    a claim about significance.
    """
    if contrast.get("mean") is None:
        return dict(status=STATUS_INCOMPLETE, reason=contrast.get("reason", "no contrast"))
    if reference_rmsd is None:
        return dict(
            status=STATUS_INCOMPLETE,
            reason="no matched baseline RMSD, so the relative part of the "
                   "threshold cannot be evaluated",
        )
    threshold = max(absolute, relative * float(reference_rmsd))
    mean, low, high = contrast["mean"], contrast["ci_low"], contrast["ci_high"]
    excludes_zero = (low > 0) or (high < 0)
    passed = mean > 0 and excludes_zero and mean >= threshold
    return dict(
        status=STATUS_PASS if passed else STATUS_FAIL,
        threshold=threshold,
        threshold_parts=dict(absolute=absolute, relative=relative,
                             reference_rmsd=float(reference_rmsd)),
        mean=mean,
        ci=[low, high],
        ci_excludes_zero=excludes_zero,
        reason=(
            "improvement clears the gate with an interval excluding zero"
            if passed
            else "improvement does not clear the gate or its interval includes zero"
        ),
    )


def safety_status(contrasts, *, margins=None, mode="backbone"):
    """Packing-safety noninferiority, per metric.

    In backbone mode there are no packing metrics at all, so the status is
    ``incomplete`` -- not ``pass``. Absence of a measurement is not evidence of
    safety.
    """
    if mode != "full":
        return dict(
            status=STATUS_INCOMPLETE,
            reason="backbone mode computes no packing metrics; packing safety is "
                   "unmeasured, which is not the same as unharmed",
        )
    margins = dict(margins or SAFETY_MARGINS)
    checks = {}
    worst = STATUS_PASS
    for metric, margin in margins.items():
        contrast = contrasts.get(metric)
        if contrast is None or contrast.get("mean") is None:
            checks[metric] = dict(status=STATUS_INCOMPLETE, reason="not measured")
            worst = STATUS_INCOMPLETE
            continue
        # Noninferiority: the candidate may be worse by at most `margin`, and
        # the bound must establish it, not merely fail to refute it.
        bound = contrast["ci_low"]
        ok = bound is not None and bound > -abs(margin)
        checks[metric] = dict(
            status=STATUS_PASS if ok else STATUS_FAIL,
            margin=-abs(margin),
            ci_low=bound,
            mean=contrast["mean"],
        )
        if not ok:
            worst = STATUS_FAIL
    return dict(status=worst, checks=checks)


def verdict(*, bb_gate, safety, have_baseline, complete_coverage):
    """Combine the gate, safety and coverage into one honest status."""
    if not have_baseline:
        return dict(
            status=STATUS_INCOMPLETE,
            reason="no B0 in the comparison; metrics can be computed against R0 "
                   "but the benefit of side-chain supervision has no verdict "
                   "without the matched backbone-only baseline",
        )
    if not complete_coverage:
        return dict(
            status=STATUS_INCOMPLETE,
            reason="the predeclared event set is not complete; an automatic gate "
                   "requires every event, not the intersection left after "
                   "candidate-specific failures",
        )
    if bb_gate.get("status") == STATUS_INCOMPLETE:
        return dict(status=STATUS_INCOMPLETE, reason=bb_gate.get("reason"))
    if bb_gate.get("status") == STATUS_FAIL:
        return dict(status=STATUS_FAIL, reason=bb_gate.get("reason"))
    if safety.get("status") != STATUS_PASS:
        return dict(
            status=STATUS_INCOMPLETE if safety.get("status") == STATUS_INCOMPLETE
            else STATUS_FAIL,
            reason=f"backbone gate passed but packing safety is "
                   f"{safety.get('status')}: {safety.get('reason', 'see checks')}",
        )
    return dict(status=STATUS_PASS, reason="backbone gate and packing safety both pass")


# ---- output ------------------------------------------------------------------


def _json_safe(value):
    """Strict JSON: undefined becomes null, never a NaN or Infinity token."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path, payload):
    """Atomic write, so a crashed run cannot leave a half-parsed artefact."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(_json_safe(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    return path


def append_rows(path, rows):
    """Append JSONL rows atomically at a shard boundary."""
    path = Path(path)
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
    return path


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge_rows(existing, incoming):
    """Merge shards, refusing duplicates that disagree."""
    def key(row):
        return (row["model"], row["panel_id"], row["sample_id"],
                row["sigma_key"], row["backbone_replicate"])

    merged = {key(r): r for r in existing}
    for row in incoming:
        k = key(row)
        if k in merged and merged[k] != row:
            raise ReportError(
                f"conflicting duplicate row for {k}; two shards disagree about "
                "the same event, so neither can be trusted"
            )
        merged[k] = row
    return list(merged.values())


def render_markdown(summary):
    """A readable report that states its own gaps."""
    lines = ["# Joint refinement evaluation", ""]
    lines.append(f"- mode: `{summary.get('mode')}`")
    lines.append(f"- panel: `{summary.get('panel_id')}` "
                 f"({summary.get('n_targets')} targets)")
    lines.append(f"- sigmas: {', '.join(summary.get('sigma_keys', []))}")
    lines.append(f"- weights: `{summary.get('weights')}`")
    lines.append("")

    final = summary.get("verdict", {})
    lines += ["## Verdict", "",
              f"**{final.get('status', 'unknown')}** -- {final.get('reason', '')}", ""]

    coverage_table = summary.get("coverage", {})
    if coverage_table:
        lines += ["## Coverage", "",
                  "| model | events | failures | targets |", "|---|---|---|---|"]
        for model, entry in sorted(coverage_table.items()):
            lines.append(f"| {model} | {entry['events']} | {entry['failures']} "
                         f"| {entry['n_samples']} |")
        lines.append("")

    contrasts = summary.get("paired", [])
    if contrasts:
        lines += ["## Paired comparisons", "",
                  "Positive means the candidate is better.", "",
                  "| candidate | reference | metric | mean | 95% CI | n |",
                  "|---|---|---|---|---|---|"]
        for c in contrasts:
            if c.get("mean") is None:
                lines.append(f"| {c['candidate']} | {c['reference']} | {c['metric']} "
                             f"| - | - | 0 |")
                continue
            lines.append(
                f"| {c['candidate']} | {c['reference']} | {c['metric']} "
                f"| {c['mean']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
                f"| {c['n_targets']} |"
            )
        lines.append("")

    missing = summary.get("missing_groups") or []
    if missing:
        lines += ["## Not measured", "",
                  "These groups are declared by the plan and not computed by this "
                  "run, so no claim is made about them:", ""]
        lines += [f"- `{name}`" for name in missing]
        lines.append("")
    return "\n".join(lines)
