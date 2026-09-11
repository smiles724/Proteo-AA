"""Sequence backends and explicit structural visibility contracts."""
# Every AA backend that runs the co-design cycle rather than the residue-type
# MLP. Adding one means adding it HERE, not hunting eleven equality checks: the
# trainer gates optimizer groups, phase re-application, checkpoint identity and
# donor validation on this, and a missed site fails silently as "trains like a
# Stage III run" rather than as an error. Kept in this package, not in stage4,
# so importing it costs nothing -- the trainer asks the question on paths that
# must stay cheap for the MLP backend.
CODESIGN_BACKENDS = ("fampnn", "ligandmpnn")


def uses_codesign(model) -> bool:
    return str(getattr(model, "aa_backend", "mlp")) in CODESIGN_BACKENDS
