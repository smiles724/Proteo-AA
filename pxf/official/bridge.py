"""Bridge the official feature dict to the coupling side's representation.

The local path got its ``Topology``/``aatype``/design mask from
``pxdesign_train``'s featurizer via ``pxf.backbone.driver.to_featurized``. The
official runtime produces neither that object nor those keys, so the same
facts are recovered from what it does produce: the biotite ``AtomArray`` the
dataloader returns alongside the features, and the features themselves.

Design tokens are identified by PXDesign's own marker rather than by a
fraction or a chain index: ``DesignFeaturizer`` renames designed residues to
``xpb`` and treats exactly ``res_name != "xpb"`` as the condition, so that
predicate is the definition rather than an inference from it.
"""

import numpy as np
import torch

DESIGN_RESNAME = "xpb"


def _token_representative(atom_array, feature_dict, n_tokens):
    """First flat-atom index belonging to each token."""
    a2t = feature_dict["atom_to_token_idx"].reshape(-1).long().cpu().numpy()
    if len(a2t) != len(atom_array):
        raise ValueError(
            f"atom_to_token_idx has {len(a2t)} entries for an AtomArray of "
            f"{len(atom_array)}; the flat atom axes disagree"
        )
    first = np.full(n_tokens, -1, dtype=np.int64)
    for atom_index, token in enumerate(a2t):
        if 0 <= token < n_tokens and first[token] < 0:
            first[token] = atom_index
    if (first < 0).any():
        missing = int((first < 0).sum())
        raise ValueError(f"{missing} token(s) have no atom in the flat axis")
    return first


def design_mask(atom_array, feature_dict, n_tokens):
    """Per-token design mask, from PXDesign's ``xpb`` marker."""
    rep = _token_representative(atom_array, feature_dict, n_tokens)
    res_names = np.asarray(atom_array.res_name)[rep]
    return torch.from_numpy(res_names == DESIGN_RESNAME)


def topology(atom_array, feature_dict, n_tokens):
    """A :class:`pxf.couple.controller.Topology` for the official features."""
    from pxf.couple.controller import Topology

    # res_names is PER ATOM, not per token: `bridge.design_mask_from_res_names`
    # iterates it against `atom_to_token_idx` and reduces to tokens itself.
    # `pxf.backbone.driver` builds the same thing by expanding its per-token
    # list back over the atoms. Taking the AtomArray's names directly also
    # keeps PXDesign's `xpb` marker, which the driver's aatype-derived names
    # lose to "UNK".
    res_names = list(np.asarray(atom_array.res_name))
    if len(res_names) != len(atom_array):
        raise ValueError("res_names must be per atom")
    residue_index = feature_dict["residue_index"].reshape(-1)[:n_tokens].long()
    chain_index = feature_dict["asym_id"].reshape(-1)[:n_tokens].long()
    return Topology(
        atom_names=list(np.asarray(atom_array.atom_name)),
        atom_to_token_idx=feature_dict["atom_to_token_idx"].reshape(-1).long(),
        num_tokens=int(n_tokens),
        res_names=res_names,
        residue_index=residue_index,
        chain_index=chain_index,
    )


def token_aatype(atom_array, feature_dict, n_tokens):
    """Per-token aatype, 20 (unknown) on design tokens.

    Only the backbone slots are read downstream -- ``shared_preparation``
    masks to N/CA/C/O immediately -- and those are the same four atom37 slots
    for every residue type, so the design region's unknown identity does not
    reach the coordinates. Returning 20 rather than a silent glycine keeps
    that explicit.
    """
    from fampnn.data import residue_constants as rc

    rep = _token_representative(atom_array, feature_dict, n_tokens)
    res_names = np.asarray(atom_array.res_name)[rep]
    out = torch.full((n_tokens,), 20, dtype=torch.long)
    for i, name in enumerate(res_names):
        one = rc.restype_3to1.get(str(name).upper())
        if one is None:
            continue
        index = rc.restype_order.get(one)
        if index is not None:
            out[i] = int(index)
    return out


class OfficialStructure:
    """The subset of ``FeaturizedStructure`` that the coupling code reads."""

    def __init__(self, atom_array, feature_dict, n_tokens, device=None):
        self.feature_dict = feature_dict
        self.num_tokens = int(n_tokens)
        self.topology = topology(atom_array, feature_dict, n_tokens)
        self.design_mask = design_mask(atom_array, feature_dict, n_tokens)
        self.aatype = token_aatype(atom_array, feature_dict, n_tokens)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.topology = self.topology.to(device)
        self.design_mask = self.design_mask.to(device)
        self.aatype = self.aatype.to(device)
        return self
