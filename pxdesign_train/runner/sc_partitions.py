"""Partition before derivatives; conservative cross-source PDB identity exclusion.

This does not claim homology exclusion or independence from donor pretraining.
Exact sequence identifiers are audited when both manifests actually contain them.
"""
import json
from pathlib import Path


def prepare_partitions(monomer_train, monomer_validation, pinder_manifest, cache):
    import pandas as pd
    from .sc_stream import fingerprint
    identity = fingerprint([monomer_train, monomer_validation, pinder_manifest], dict(audit="pdb_identity_v1"))
    import hashlib
    tag = hashlib.sha256(json.dumps(sorted(identity["files"].values())).encode()).hexdigest()[:16]
    mono_out, pinder_out = Path(cache)/f"train-{tag}.csv.gz", Path(cache)/f"pinder-{tag}.parquet"
    train = pd.read_csv(monomer_train)
    val = pd.read_csv(monomer_validation)
    columns = {"pinder_id", "pdb_path", "converted_binder_chain", "source_split", "cluster_id",
               "num_tokens", "binder_tokens", "sequence_sha256", "sequence", "uniprot_id"}
    if str(pinder_manifest).endswith(".parquet"):
        import pyarrow.parquet as pq
        available = set(pq.read_schema(pinder_manifest).names)
        pinder = pd.read_parquet(pinder_manifest, columns=sorted(columns & available))
    else:
        pinder = pd.read_csv(pinder_manifest, usecols=lambda name: name in columns)
    for frame, required in ((train, ["pdb_id"]), (val, ["pdb_id"]), (pinder, ["pinder_id", "source_split", "cluster_id"])):
        for key in required:
            if key not in frame:
                raise ValueError(f"Partition audit requires {key}")
    mpdb = train.pdb_id.astype(str).str.lower()
    vpdb = set(val.pdb_id.astype(str).str.lower())
    ppdb = pinder.pinder_id.astype(str).str[:4].str.lower()
    pheldout = ~pinder.source_split.astype(str).eq("train")
    forbidden = vpdb | set(ppdb[pheldout])
    mono_keep = ~mpdb.isin(forbidden)
    pinder_keep = pheldout | ~ppdb.isin(forbidden)
    # Shared identifiers only; source-specific cluster names are not comparable.
    sequence_audits = {}
    for column in ("sequence_sha256", "sequence", "uniprot_id"):
        if all(column in frame for frame in (train, val, pinder)):
            heldout = set(val[column].dropna().astype(str)) | set(pinder.loc[pheldout, column].dropna().astype(str))
            heldout.discard("")
            mono_hits = train[column].astype(str).isin(heldout)
            pinder_hits = pinder[column].astype(str).isin(heldout) & ~pheldout
            mono_keep &= ~mono_hits
            pinder_keep &= ~pinder_hits
            sequence_audits[column] = dict(monomer_hits=int(mono_hits.sum()), pinder_hits=int(pinder_hits.sum()))
    if not mono_keep.any() or not (pinder_keep & ~pheldout).any():
        raise ValueError("Overlap exclusion removed all training items")
    # Gzip mtime=0 makes fingerprints independent of the run/output directory.
    train.loc[mono_keep].to_csv(mono_out, index=False, compression=dict(method="gzip", mtime=0))
    pinder.loc[pinder_keep].to_parquet(pinder_out, index=False)
    audit = dict(method="cross_source_pdb_identity_v1", input_hashes=sorted(identity["files"].values()),
        excluded_monomers=int((~mono_keep).sum()), excluded_pinder=int((~pinder_keep).sum()),
        sequence_audits=sequence_audits,
        sequence_audit_status="shared identifiers checked" if sequence_audits else "no shared sequence identifier columns; homology audit still required",
        homology_independence=False, donor_pretraining_independence=False)
    (Path(cache)/f"overlap-{tag}.json").write_text(json.dumps(audit, indent=2))
    return mono_out, pinder_out, audit
