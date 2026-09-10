"""Export the actual held-out items, including any crop-retry substitution."""
import csv
import json
from pathlib import Path
import torch


def export_validation(components, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = {}
    for dataset in components.train_dataset.datasets:
        clusters = getattr(dataset.provider, "cluster_ids", None)
        if clusters is not None:
            source = "pinder" if "pinder" in dataset.source_name else "protenix"
            training.setdefault(source, set()).update(map(str, clusters))
    rows, seen = [], set()
    for name, loader in (components.named_eval_dataloaders or {}).items():
        if not name.startswith("binder_"):
            continue
        source = name.removeprefix("binder_")
        if source not in training:
            raise ValueError(f"No training cluster inventory for {source}")
        for batch in loader:
            cluster, sample = batch["cluster_id"], batch["sample_id"]
            if not cluster or cluster in training[source]:
                raise ValueError(f"Validation cluster is missing or appears in training: {sample}")
            if (source, sample) in seen:
                continue
            seen.add((source, sample))
            path = f"{name}_{len(rows):05d}.pt"
            torch.save(batch, output / path)
            rows.append(dict(path=path, source=source, sample_id=sample, cluster_id=cluster, split="val"))
    if not rows:
        raise ValueError("No held-out binder batches to export; enable binder validation")
    with (output / "manifest.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "training_clusters.json").write_text(json.dumps({k: sorted(v) for k,v in training.items()}, indent=2))
    print(f"Exported {len(rows)} strict held-out binder batches to {output}", flush=True)
