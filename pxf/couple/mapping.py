"""Which PXDesign token each FaMPNN residue is, built from identifiers.

FaMPNN is handed **only the generated chain**, so its residue axis is shorter
than PXDesign's token axis and the two are not the same object. A residual
computed per FaMPNN residue therefore has to be scattered, and the scatter has
to be built from chain and residue identifiers rather than from the assumption
that the design region is the final contiguous slice of the token axis.

That assumption is not safe here. The featurizer orders tokens by the parsed
structure, the converter can relabel chains, and the binder is whichever chain
is *smaller* -- which is often chain A, i.e. the **front** of the axis. Measured
on the prepared panel: the generated chain is the first chain for **13 of 31**
targets. A contiguous-tail scatter would have written the residual onto the fixed
target for 42% of the panel, silently, which is the one thing the fixed-target
policy exists to prevent.

So the mapping is a dictionary lookup on ``(asym_id, residue_index)`` and every
structural assumption about it is asserted rather than trusted: the mapping is
injective, it lands only on design tokens, and it covers exactly the residues
FaMPNN was given.

**The mask goes on after the projections.** ``A_SB``'s output projection has a
bias, and a readout whose input is zeroed still emits ``W2 SiLU(W1[0; e(sigma)])
+ b`` -- non-zero on every token including the target's. Masking the *input* or
the readout would leave that bias to be written onto fixed coordinates, so
:func:`scatter_design_residual` masks the finished residual.
"""

from dataclasses import dataclass

import torch


@dataclass
class TokenMapping:
    """``gen_to_px``: FaMPNN residue -> PXDesign token, plus what it was built from."""

    gen_to_px: torch.Tensor  # [L_generated] long
    design_mask: torch.Tensor  # [N_tokens] bool
    n_tokens: int
    keys: list  # the (asym_id, residue_index) pairs, in FaMPNN order

    def __post_init__(self):
        index = self.gen_to_px
        if index.dim() != 1:
            raise ValueError(f"gen_to_px must be 1-D, got {tuple(index.shape)}")
        if index.numel() != index.unique().numel():
            raise ValueError(
                "gen_to_px is not injective: two FaMPNN residues map to one "
                "PXDesign token, so the scatter would drop one of them"
            )
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= self.n_tokens):
            raise ValueError(
                f"gen_to_px indexes outside the token axis [0, {self.n_tokens})"
            )
        if not bool(self.design_mask[index].all()):
            stray = index[~self.design_mask[index]]
            raise ValueError(
                f"{stray.numel()} mapped token(s) are not design tokens (e.g. "
                f"{stray[:5].tolist()}); the residual would land on the target"
            )
        if int(self.design_mask.sum()) != index.numel():
            raise ValueError(
                f"{int(self.design_mask.sum())} design tokens but "
                f"{index.numel()} mapped residues; the generated chain and the "
                "design region disagree"
            )

    @property
    def contiguous_tail(self):
        """Whether the design region happens to be the final slice.

        Recorded, never relied on: it is true for some targets and false for
        others, and code that assumes it fails silently on the others.
        """
        index = self.gen_to_px
        if index.numel() == 0:
            return False
        expected = torch.arange(
            self.n_tokens - index.numel(), self.n_tokens, device=index.device
        )
        return bool(torch.equal(index.sort().values, expected))

    def identity(self):
        return dict(
            generated_residues=int(self.gen_to_px.numel()),
            n_tokens=int(self.n_tokens),
            design_tokens=int(self.design_mask.sum()),
            first_token=int(self.gen_to_px.min()) if self.gen_to_px.numel() else None,
            last_token=int(self.gen_to_px.max()) if self.gen_to_px.numel() else None,
            contiguous_tail=self.contiguous_tail,
        )


def token_keys(feature_dict, *, n_tokens=None):
    """``[(asym_id, residue_index)]`` per PXDesign token, in token order."""
    asym = feature_dict["asym_id"].reshape(-1).long().tolist()
    residue = feature_dict["residue_index"].reshape(-1).long().tolist()
    limit = n_tokens if n_tokens is not None else len(asym)
    return [(int(a), int(r)) for a, r in zip(asym[:limit], residue[:limit])]


def build_mapping(feature_dict, design_mask, *, generated_keys=None, n_tokens=None):
    """Build ``gen_to_px`` by identifier lookup.

    ``generated_keys`` is the ``(asym_id, residue_index)`` sequence in the order
    FaMPNN received them. Omitted, it defaults to the design tokens in token
    order -- the common case, where the generated chain is extracted straight
    from the token axis. Supplying it is what makes a reordered extraction
    correct rather than accidentally correct.
    """
    design_mask = design_mask.reshape(-1).bool()
    total = int(n_tokens if n_tokens is not None else design_mask.numel())
    keys = token_keys(feature_dict, n_tokens=total)
    if len(keys) != total:
        raise ValueError(f"{len(keys)} identifier(s) for {total} tokens")
    lookup = {}
    for position, key in enumerate(keys):
        if key in lookup:
            raise ValueError(
                f"duplicate token identifier {key}: (asym_id, residue_index) "
                "does not identify a token uniquely, so the mapping is ambiguous"
            )
        lookup[key] = position
    if generated_keys is None:
        generated_keys = [keys[i] for i in torch.nonzero(design_mask).reshape(-1).tolist()]
    missing = [k for k in generated_keys if k not in lookup]
    if missing:
        raise ValueError(
            f"{len(missing)} generated residue(s) have no matching token, e.g. "
            f"{missing[:3]}; the chain/residue identifiers disagree"
        )
    index = torch.tensor(
        [lookup[k] for k in generated_keys], dtype=torch.long, device=design_mask.device
    )
    return TokenMapping(
        gen_to_px=index,
        design_mask=design_mask,
        n_tokens=total,
        keys=list(generated_keys),
    )


def scatter_design_residual(delta_gen, mapping, *, width=None, reference=None):
    """Scatter a per-generated-residue residual onto the token axis, masked.

    Applied **after** all adapter projections and biases: the projection has a
    bias, so a zeroed input still produces a non-zero residual on every token,
    and masking anything earlier would leave that bias to be written onto the
    target's fixed coordinates.
    """
    if delta_gen.dim() == 3:
        if delta_gen.shape[0] != 1:
            raise ValueError(
                f"expected one example, got batch {delta_gen.shape[0]}; the "
                "mapping is per structure"
            )
        delta_gen = delta_gen[0]
    if delta_gen.shape[0] != mapping.gen_to_px.numel():
        raise ValueError(
            f"residual has {delta_gen.shape[0]} rows for "
            f"{mapping.gen_to_px.numel()} generated residues"
        )
    channels = int(width if width is not None else delta_gen.shape[-1])
    if delta_gen.shape[-1] != channels:
        raise ValueError(f"residual width {delta_gen.shape[-1]} != {channels}")

    delta = delta_gen.new_zeros(mapping.n_tokens, channels)
    delta.index_copy_(0, mapping.gen_to_px.to(delta.device), delta_gen.to(delta.dtype))
    mask = mapping.design_mask.to(delta.device)
    delta = delta.masked_fill(~mask[:, None], 0)

    leaked = int(torch.count_nonzero(delta[~mask]))
    if leaked:
        raise AssertionError(
            f"{leaked} non-zero residual entr(ies) on target tokens after "
            "masking; the fixed-target policy would be violated at the source"
        )
    if reference is not None:
        expected = reference.shape[-2:] if reference.dim() >= 2 else None
        if expected is not None and tuple(delta.shape) != tuple(expected):
            raise ValueError(
                f"scattered residual {tuple(delta.shape)} does not match the "
                f"token features {tuple(expected)}"
            )
    return delta[None]


def target_atom_mask(feature_dict, design_mask, *, n_atom=None):
    """``[N_atom]`` true for atoms of the *fixed target*, false for generated.

    Derived from ``atom_to_token_idx`` rather than from an atom-range guess, for
    the same reason as the token mapping: the design region is not reliably at
    either end.
    """
    tokens = feature_dict["atom_to_token_idx"].reshape(-1).long()
    if n_atom is not None:
        tokens = tokens[:n_atom]
    design = design_mask.reshape(-1).bool().to(tokens.device)
    return ~design[tokens]
