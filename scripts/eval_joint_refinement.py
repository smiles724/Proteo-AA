#!/usr/bin/env python
"""Score joint-refinement checkpoints on a held-out panel.

A falling training loss is not the verdict. This asks whether backbone
training with side-chain supervision improves held-out *backbone* accuracy
over matched backbone-only fine-tuning, on events every arm sees identically.

Two modes are planned; one is implemented.

``backbone``  one denoise per (target, sigma, replicate) per model, backbone
              metrics only. No side-chain rollout, and therefore no claim
              whatever about packing quality.
``full``      the same prediction plus fresh packing from a common frozen
              FaMPNN, with conformation, placement and chemistry metrics.
              NOT IMPLEMENTED -- it is refused rather than silently degraded
              to backbone mode, because "the packing metrics are missing" and
              "the packing is fine" must never look the same in a report.

At inference every current arm is one backbone denoise with the side-chain
branch off. BF is not handed native local conformations, BS runs no detached
branch, B1/B2 get no teacher-forced side-chain state: those words describe how
the weights were trained, not what happens here. Arm identity controls
interpretation, never the forward path.

Usage:

    python scripts/eval_joint_refinement.py \\
        --panel heldout.jsonl \\
        --donor .../pxdesign_v0.1.0.pt \\
        --checkpoint B0=.../B0/checkpoints/final.pt \\
        --checkpoint B1=.../B1/checkpoints/final.pt \\
        --include-donor --weights ema --mode backbone \\
        --sigmas 0.105 0.314 0.847 1.939 \\
        --replicates 1 --seed 17 --out eval-dir

    # build a panel manifest from a directory of held-out CIFs
    python scripts/eval_joint_refinement.py --build-panel-from .../cif_val \\
        --panel heldout.jsonl --min-length 64 --max-length 256
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pxf.eval import joint_metrics, joint_report  # noqa: E402
from pxf.joint import evaluation as joint_eval  # noqa: E402

PROTOCOL_VERSION = "joint-eval/1"
DEFAULT_SIGMAS = (0.105, 0.314, 0.847, 1.939)
# Reported separately from the primary aggregate: a clean-end probe answers a
# different question and folding it in would dilute the refinement window.
SAFETY_SIGMAS = (0.010, 0.082)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--panel", help="held-out panel manifest (JSONL)")
    p.add_argument("--structures", dest="panel", help="alias for --panel")
    p.add_argument("--build-panel-from", help="directory of CIFs to write a panel from")
    p.add_argument("--min-length", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--panel-id", default=None, help="defaults to the manifest digest")

    p.add_argument("--donor")
    p.add_argument("--checkpoint", action="append", default=[],
                   metavar="LABEL=PATH", help="repeatable; LABEL must be unique")
    p.add_argument("--expect-arm", action="append", default=[],
                   metavar="LABEL=ARM", help="independent label check")
    p.add_argument("--include-donor", action="store_true",
                   help="score the untrained donor as R0")
    p.add_argument("--weights", choices=joint_eval.WEIGHT_CHOICES, default="ema")
    p.add_argument("--config", help="YAML with the same settings")

    p.add_argument("--mode", choices=("backbone", "full"), default="backbone")
    p.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--seed", type=int, default=17,
                   help="evaluation randomness; NOT the training seed")
    p.add_argument("--device", default=None)
    p.add_argument("--cpu-threads", type=int, default=None)

    p.add_argument("--bootstrap-resamples", type=int,
                   default=joint_report.BOOTSTRAP_RESAMPLES)
    p.add_argument("--bootstrap-seed", type=int, default=20260920)
    p.add_argument("--baseline", default="B0", help="reference arm for the gate")
    p.add_argument("--candidate", default=None,
                   help="arm the verdict gates on; required when several are present")

    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--report-only", action="store_true",
                   help="re-aggregate existing rows without predicting")
    p.add_argument("--out", help="output directory")

    # full mode only
    p.add_argument("--fampnn-checkpoint", default="0.0")
    p.add_argument("--pack-steps", type=int, default=50)
    p.add_argument("--packing-replicates", type=int, default=1)
    return p.parse_args(argv)


def load_config(path):
    if not path:
        return {}
    import yaml

    with open(path) as stream:
        return yaml.safe_load(stream) or {}


# ---- panel -------------------------------------------------------------------


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_panel(directory, out_path, *, min_length, max_length):
    """Write a panel manifest from a directory of CIFs.

    Records the content digest of every file, so a panel is pinned to bytes
    rather than to a path. Lengths are recorded as *accepted*, after parsing,
    not promised in advance.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("*.cif"))
    if not paths:
        raise SystemExit(f"no CIFs under {directory}")
    rows, seen = [], set()
    for path in paths:
        sample_id = path.stem
        if sample_id in seen:
            raise SystemExit(
                f"duplicate sample_id {sample_id!r}; file stems are not unique "
                "across directories, so the manifest needs explicit ids"
            )
        seen.add(sample_id)
        rows.append(dict(
            sample_id=sample_id,
            path=str(path.resolve()),
            sha256=file_sha256(path),
            cluster_id=None,
            split="heldout",
            min_length=min_length,
            max_length=max_length,
        ))
    out_path = Path(out_path)
    with out_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} panel rows -> {out_path}")
    return out_path


def read_panel(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("panel manifest has duplicate sample_id values")
    for row in rows:
        actual = file_sha256(row["path"])
        if row.get("sha256") and actual != row["sha256"]:
            raise SystemExit(
                f"{row['sample_id']}: file digest {actual[:12]} does not match the "
                f"manifest's {row['sha256'][:12]}; the panel is not the one recorded"
            )
    return rows


def panel_digest(rows):
    payload = json.dumps(
        [[r["sample_id"], r.get("sha256")] for r in sorted(rows, key=lambda r: r["sample_id"])],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---- static examples ---------------------------------------------------------


def prepare_examples(panel_rows, fampnn, *, device, min_length, max_length):
    """Featurize once, before any model is loaded.

    Input construction is separated from target scoring: the batch carries the
    native side chains only as *targets*, and the backbone path never reads
    them. Failures here are model-independent exclusions and are recorded as
    such, so they cannot later be mistaken for one arm's failure.
    """
    from fampnn.data.data import load_feats_from_pdb, process_single_pdb

    from pxf.backbone.driver import featurize_structures, to_featurized
    from pxf.joint import data as joint_data

    prepared, excluded = [], []
    for row in panel_rows:
        path = row["path"]
        try:
            sample_id, source = featurize_structures([path], crop_size=1024)[0]
            structure = to_featurized(sample_id, source[0]).to(device)
            length = int(structure.num_tokens)
            if not min_length <= length <= max_length:
                excluded.append(dict(sample_id=row["sample_id"],
                                     reason=f"length {length}", scope="panel"))
                continue
            native = process_single_pdb(load_feats_from_pdb(path))
            batch = joint_data.build_joint_batch(fampnn, structure, native)
        except Exception as error:
            excluded.append(dict(sample_id=row["sample_id"],
                                 reason=f"{type(error).__name__}: {error}",
                                 scope="panel"))
            continue
        prepared.append(dict(row=row, batch=batch, length=length))
    return prepared, excluded


# ---- scoring -----------------------------------------------------------------


def native_and_mask(batch, supplied):
    """Native atom37 and the mask of atoms present on both sides."""
    native = batch.native_batch["x"]
    if native.dim() == 4:
        native = native[0]
    missing = batch.native_batch["missing_atom_mask"]
    if missing.dim() == 3:
        missing = missing[0]
    present = (1.0 - missing).to(native.device)
    supplied = supplied[0] if supplied.dim() == 3 else supplied
    return native, (present * supplied.to(present.device))


def score_event(loaded, batch, panel_row, sigma, replicate, *, panel_id, seed):
    """Predict and score one event for one model."""
    conditioning = joint_eval.conditioning_for(loaded, batch)
    eps = joint_eval.replay_backbone_noise(
        batch.backbone_target.shape, seed, panel_id,
        panel_row["sample_id"], sigma, replicate,
    ).to(batch.backbone_target.device)

    started = time.time()
    forward, dense, supplied = joint_eval.predict_backbone(
        loaded, conditioning, batch, sigma=sigma, backbone_noise=eps
    )
    if torch.cuda.is_available() and batch.backbone_target.device.type == "cuda":
        torch.cuda.synchronize()
    denoise_seconds = time.time() - started

    pred37 = dense[0] if dense.dim() == 4 else dense
    # densify_prediction keeps a batch axis on both; the metrics are per
    # structure, so drop it on the same side for coordinates and masks.
    supplied2 = supplied[0] if supplied.dim() == 3 else supplied
    native37, mask37 = native_and_mask(batch, supplied)

    row = dict(
        protocol=PROTOCOL_VERSION,
        model=loaded.label,
        arm=loaded.arm,
        weights=loaded.weights,
        model_identity=loaded.content_identity(),
        panel_id=panel_id,
        sample_id=panel_row["sample_id"],
        cluster_id=panel_row.get("cluster_id"),
        sigma=float(sigma),
        sigma_key=joint_eval.sigma_key(sigma),
        backbone_replicate=int(replicate),
        length=int(batch.length),
        eval_seed=int(seed),
        noise_digest=joint_eval.noise_digest(eps),
        denoise_seconds=denoise_seconds,
    )

    ok, reason = joint_metrics.prediction_is_scorable(pred37, supplied2)
    if not ok:
        row.update(joint_metrics.model_failure(reason))
        return row

    started = time.time()
    row.update(joint_metrics.backbone_group(pred37, native37, mask37))
    row.update(joint_metrics.backbone_geometry(
        pred37, mask37,
        residue_index=batch.native_batch.get("residue_index"),
        chain_index=batch.native_batch.get("chain_index"),
    ))
    row["metric_seconds"] = time.time() - started
    row["failure"] = False
    return row


# ---- orchestration -----------------------------------------------------------


def resolve_models(args):
    pairs = []
    for item in args.checkpoint:
        if "=" not in item:
            raise SystemExit(f"--checkpoint needs LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        pairs.append((label, path))
    labels = [label for label, _ in pairs]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate --checkpoint labels; each must be unique")
    expect = {}
    for item in args.expect_arm:
        label, arm = item.split("=", 1)
        expect[label] = arm
    return pairs, expect


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    for key, value in config.items():
        if getattr(args, key, None) in (None, [], ()) and value is not None:
            setattr(args, key, value)

    if args.build_panel_from:
        if not args.panel:
            raise SystemExit("--build-panel-from needs --panel to write to")
        build_panel(args.build_panel_from, args.panel,
                    min_length=args.min_length, max_length=args.max_length)
        return 0

    if args.mode == "full":
        raise SystemExit(
            "--mode full is not implemented. It needs the common inference "
            "packer and the conformation/placement/environment/chemistry "
            "groups (milestones 3-4 of the plan). Backbone mode is refused "
            "from standing in for it: a report that showed backbone numbers "
            "under a 'full' heading would imply packing had been checked. "
            "Use --mode backbone, and read its packing-safety status as "
            "'incomplete', which is what it will say."
        )

    for required in ("panel", "donor", "out"):
        if not getattr(args, required):
            raise SystemExit(f"--{required} is required")
    if args.cpu_threads:
        torch.set_num_threads(int(args.cpu_threads))

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "backbone_rows.jsonl"
    failures_path = out / "failures.jsonl"

    panel_rows = read_panel(args.panel)
    panel_id = args.panel_id or panel_digest(panel_rows)
    sigma_keys = joint_eval.check_distinct_sigmas(args.sigmas)

    existing = joint_report.read_rows(rows_path)
    done = {
        (r["model"], r["panel_id"], r["sample_id"], r["sigma_key"], r["backbone_replicate"])
        for r in existing
    }

    if args.report_only:
        rows = existing
    else:
        rows = existing + run_predictions(
            args, panel_rows, panel_id, done, rows_path, failures_path
        )

    # Model-independent exclusions shrink the predeclared set; candidate-
    # specific failures do not. Only the first kind may leave a gate passable.
    panel_excluded = {
        f["sample_id"]
        for f in joint_report.read_rows(failures_path)
        if f.get("scope") == "panel"
    }
    summary = build_summary(args, rows, panel_rows, panel_id, sigma_keys,
                            panel_excluded=panel_excluded)
    joint_report.write_json(out / "summary.json", summary)
    joint_report.write_json(out / "paired.json", summary.get("paired", []))
    (out / "report.md").write_text(joint_report.render_markdown(summary))
    print(joint_report.render_markdown(summary))
    print(f"wrote {out}/summary.json, paired.json, report.md")
    return 0


def run_predictions(args, panel_rows, panel_id, done, rows_path, failures_path):
    from fampnn.model.sd_model import SeqDenoiser

    from pxf.device import select_device
    from pxf.provenance import fampnn_checkpoint

    device = select_device(args.device)

    # FaMPNN is loaded to BUILD the batch -- build_joint_batch needs its
    # supervision masks -- and for nothing else in backbone mode. It never
    # sees a prediction here and never contributes a number.
    weights = torch.load(fampnn_checkpoint(args.fampnn_checkpoint),
                         map_location="cpu", weights_only=False)
    fampnn = SeqDenoiser(weights["model_cfg"])
    fampnn.load_state_dict(weights["state_dict"], strict=True)
    fampnn.to(device).eval().requires_grad_(False)

    donor_model, _configs, donor_record, snapshot = joint_eval.load_donor(
        args.donor, device=device
    )

    prepared, excluded = prepare_examples(
        panel_rows, fampnn, device=device,
        min_length=args.min_length, max_length=args.max_length,
    )
    if excluded:
        joint_report.append_rows(failures_path, excluded)
    if not prepared:
        raise SystemExit("no panel structure could be prepared")

    # Shard over targets. Sharding must not change a seed: every seed is a
    # hash of the event key, which does not contain the shard.
    if args.shard_count > 1:
        prepared = [e for i, e in enumerate(prepared) if i % args.shard_count == args.shard_index]

    pairs, expect = resolve_models(args)
    model_specs = []
    if args.include_donor:
        model_specs.append((joint_eval.DONOR_LABEL, None))
    model_specs += pairs

    produced = []
    for label, path in model_specs:
        if path is None:
            loaded = joint_eval.load_donor_as_model(donor_model, snapshot, donor_record,
                                                    label=label)
        else:
            loaded = joint_eval.load_joint_checkpoint(
                path, donor_model=donor_model, donor_snapshot=snapshot,
                donor_record=donor_record, weights=args.weights,
                expected_arm=expect.get(label), label=label,
            )
        print(f"[{label}] arm={loaded.arm} weights={loaded.weights} "
              f"step={loaded.step} ckpt={loaded.checkpoint_sha256[:12]}")

        batch_rows = []
        for entry in prepared:
            for sigma in args.sigmas:
                for replicate in range(int(args.replicates)):
                    key = (label, panel_id, entry["row"]["sample_id"],
                           joint_eval.sigma_key(sigma), replicate)
                    if key in done:
                        continue
                    row = score_event(loaded, entry["batch"], entry["row"], sigma,
                                      replicate, panel_id=panel_id, seed=args.seed)
                    batch_rows.append(row)
        joint_report.append_rows(rows_path, batch_rows)
        produced += batch_rows
        print(f"[{label}] {len(batch_rows)} events")
    return produced


def build_summary(args, rows, panel_rows, panel_id, sigma_keys, *, panel_excluded=()):
    scored = [r for r in rows if not r.get("failure")]
    models = sorted({r["model"] for r in rows})
    clusters = {r["sample_id"]: r.get("cluster_id") for r in rows if r.get("cluster_id")}

    primary = [k for k in sigma_keys
               if k not in {joint_eval.sigma_key(s) for s in SAFETY_SIGMAS}]

    contrasts = []
    baseline = args.baseline
    have_baseline = baseline in models
    for candidate in models:
        if candidate == baseline or not have_baseline:
            continue
        for metric in ("bb_ca_rmsd", "bb_backbone_rmsd"):
            contrasts.append(joint_report.paired_contrast(
                scored, candidate=candidate, reference=baseline, metric=metric,
                sigmas=primary, clusters=clusters or None,
                n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
            ))

    reference_values = joint_report.protein_values(scored, "bb_ca_rmsd", sigmas=primary)
    baseline_rmsd = None
    if have_baseline:
        values = [v for (m, _s), v in reference_values.items() if m == baseline and v is not None]
        baseline_rmsd = sum(values) / len(values) if values else None

    # One gate per candidate. Picking whichever arm happened to be first would
    # let the verdict depend on argument order, and R0 is a reference point --
    # the untrained donor -- not a candidate for the side-chain question.
    gates = {
        c["candidate"]: joint_report.backbone_gate(c, baseline_rmsd)
        for c in contrasts
        if c["metric"] == "bb_ca_rmsd" and c["candidate"] != joint_eval.DONOR_LABEL
    }
    candidate = args.candidate
    if candidate is None:
        automatic = sorted(gates)
        candidate = automatic[0] if len(automatic) == 1 else None
    if candidate is None:
        gate = dict(
            status=joint_report.STATUS_INCOMPLETE,
            reason=(
                f"no candidate declared and {len(gates)} are present "
                f"({sorted(gates)}); pass --candidate so the verdict does not "
                "depend on argument order"
            ),
        )
    elif candidate not in gates:
        gate = dict(status=joint_report.STATUS_INCOMPLETE,
                    reason=f"candidate {candidate!r} has no paired contrast")
    else:
        gate = gates[candidate]
    safety = joint_report.safety_status({}, mode=args.mode)

    cov = joint_report.coverage(rows)
    panel_excluded = set(panel_excluded)
    eligible = [r for r in panel_rows if r["sample_id"] not in panel_excluded]
    expected_events = len(eligible) * len(sigma_keys) * int(args.replicates)
    complete = all(entry["events"] >= expected_events and entry["failures"] == 0
                   for entry in cov.values()) if cov else False

    return dict(
        protocol=PROTOCOL_VERSION,
        mode=args.mode,
        panel_id=panel_id,
        n_targets=len(panel_rows),
        n_eligible_targets=len(eligible),
        panel_excluded=sorted(panel_excluded),
        sigma_keys=sigma_keys,
        primary_sigma_keys=primary,
        weights=args.weights,
        eval_seed=args.seed,
        bootstrap=dict(resamples=args.bootstrap_resamples, seed=args.bootstrap_seed),
        models=models,
        baseline=baseline,
        coverage=cov,
        expected_events_per_model=expected_events,
        complete_coverage=complete,
        baseline_ca_rmsd=baseline_rmsd,
        paired=contrasts,
        candidate=candidate,
        backbone_gate=gate,
        backbone_gates=gates,
        packing_safety=safety,
        missing_groups=list(joint_metrics.NOT_IMPLEMENTED_GROUPS),
        verdict=joint_report.verdict(
            bb_gate=gate, safety=safety,
            have_baseline=have_baseline, complete_coverage=complete,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
