"""Shared preparation, arm feedback and scoring for the generation stress test.

Three pieces, separated because each has a rule that is easy to violate
silently.

:func:`shared_preparation` builds **one** sequence and packing per
``(target, seed, event)`` and every arm receives it. If each arm designed its own
sequence, the arms would differ by the sequence as well as by the residual, and
nothing downstream could separate the two.

:func:`arm_feedback` returns a callback that produces the residual for exactly
one solver invocation. The residual is scattered onto the token axis and masked
to the design region *after* the adapter's projections, so the output bias
cannot reach the fixed target.

:func:`score_arm` reports three families separately, because they answer
different questions and a gain in one does not imply a gain in another --
FaMPNN is target-blind, so better isolated packing carries no promise of better
target compatibility.
"""

from pathlib import Path

import torch

from pxf import atom37

# A heavy-atom pair closer than this is an overlap rather than close packing.
CLASH_RADIUS = 2.6
CONTACT_RADIUS = 5.0
CA_IDEAL = 3.8
CA_TOLERANCE = 0.3


def px_driver_densify(flat, structure, px_driver):
    """Flat PXDesign atoms -> dense atom37, for the whole complex."""
    from pxf.couple.converter import PXFaRepresentationConverter

    converter = PXFaRepresentationConverter()
    topology = structure.topology
    return converter.px_backbone_to_fampnn(
        flat,
        topology.atom_names,
        topology.atom_to_token_idx,
        topology.num_tokens,
        res_names=topology.res_names,
        residue_index=topology.residue_index,
        chain_index=topology.chain_index,
        aatype=structure.aatype,
    ).coords_af2[0]


def shared_preparation(
    *, controller, bound, state, structure, token_map, sc_stream, pack_steps, temperature
):
    """One sequence and one packed structure, shared by every arm.

    The clean-estimate forward pass here is a **separate** evaluation of the
    denoiser at the saved state. It does not advance the solver: every arm still
    takes exactly one solver step per schedule step, which the injection
    counters assert.

    Runs on the side-chain stream, so the sequence sampling and the packing
    cannot consume the backbone's randomness.
    """
    from pxf.couple import visibility as vis
    from pxf.couple.fampnn_iface import encode

    with torch.no_grad():
        # The no-feedback clean estimate at the saved state.
        clean, _a = bound(state.x_noisy, state.sigma, feedback=None)
        dense = px_driver_densify(clean, structure, controller)
        generated = dense[token_map.gen_to_px.to(dense.device)][None]

        backbone = list(atom37.BACKBONE_SLOTS)
        given = torch.zeros(
            1, generated.shape[1], atom37.NUM_ATOM37, device=generated.device
        )
        given[..., backbone] = 1.0
        backbone_only = generated * given[..., None]
        residue_index = torch.arange(generated.shape[1], device=generated.device)[None]
        chain_index = torch.zeros_like(residue_index)

        with sc_stream.active():
            # Sequence design on the generated chain alone: the native
            # partner's identity and its side chains are both absent.
            from pxf.couple import codesign

            s_hat, sidechains, aux = codesign.codesign_native(
                controller.fampnn,
                backbone_only,
                residue_index=residue_index,
                chain_index=chain_index,
                pack_steps=pack_steps,
                temperature=temperature,
            )
        aatype = s_hat.reshape(1, -1).long()
        coords = backbone_only.clone()
        coords[..., list(atom37.SIDECHAIN_SLOTS), :] = sidechains
        seq_mask = torch.ones(coords.shape[:2], device=coords.device)
        availability = vis.predicted_availability(
            aatype, seq_mask, given, coords, sidechains=sidechains
        )
        # Fresh full-atom encode of the completed structure.
        _logits, h_packed, _features = encode(
            controller.fampnn,
            coords,
            aatype,
            atom_availability=availability.available,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )
        _logits2, h_base, _f2 = encode(
            controller.fampnn,
            backbone_only,
            aatype,
            atom_availability=vis.predicted_availability(
                aatype, seq_mask, given, backbone_only
            ).available,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )
        packed = vis.PackedStructure(
            h_packed=h_packed,
            coords37=coords,
            aatype=aatype,
            seq_mask=seq_mask,
            visibility=availability,
            psce=aux.get("psce"),
            h_base=h_base,
        )
    return dict(
        packed=packed,
        aatype=aatype,
        sidechains=sidechains,
        clean=clean,
        generated=generated,
        given=given,
        residue_index=residue_index,
        chain_index=chain_index,
    )


def arm_feedback(arm, *, trained, adapters, controller, prep, token_map, torsions, seed):
    """A one-shot residual callback for this arm, or ``None`` for the baseline."""
    from dataclasses import replace as dc_replace

    from pxf.couple.mapping import scatter_design_residual

    if arm == "baseline":
        return None, None
    label = "full" if arm == "scrambled" else arm
    if label not in trained:
        return None, None
    module = trained[label]["module"]

    def feedback(state):
        adapters.sc_to_bb = module
        packed = prep["packed"]
        if arm == "scrambled":
            deltas = torsions.random_chi_deltas(
                packed.aatype,
                60.0 * 3.141592653589793 / 180.0,
                generator=torch.Generator().manual_seed(seed + 4242),
            )
            moved = torsions.perturb_chi(
                packed.coords37, packed.aatype, deltas, available=packed.available
            )
            # Re-encoded, not substituted: keeping h_packed would leave the
            # encoder's own view of the side chains unperturbed.
            from pxf.couple import visibility as vis
            from pxf.couple.fampnn_iface import encode

            availability = vis.predicted_availability(
                packed.aatype,
                packed.seq_mask,
                prep["given"],
                moved,
                sidechains=moved[..., list(atom37.SIDECHAIN_SLOTS), :],
            )
            _l, h_packed, _f = encode(
                controller.fampnn,
                moved,
                packed.aatype,
                atom_availability=availability.available,
                seq_mask=packed.seq_mask,
                residue_index=prep["residue_index"],
                chain_index=prep["chain_index"],
            )
            packed = dc_replace(
                packed, coords37=moved, h_packed=h_packed, visibility=availability
            )
        delta_gen, _stats = adapters.delta_a(packed, state.sigma)
        if delta_gen is None:
            return None
        if not torch.is_tensor(delta_gen):
            raise ValueError(
                "the generation stress test scatters the residual onto the "
                "design region of the token axis, which only the late "
                "decoder-input adapter produces. An early conditioner's payload "
                "addresses s_single and z_pair, where 'the design region' is a "
                "different set of indices on a different axis; masking it here "
                "would silently apply the wrong restriction"
            )
        # Scatter and mask AFTER the projections and their bias.
        return scatter_design_residual(delta_gen, token_map)

    return feedback, module


def _pairwise_min(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return float("nan"), 0, 0
    d = torch.cdist(a.float(), b.float())
    return (
        float(d.min()),
        int((d < CLASH_RADIUS).sum()),
        int((d < CONTACT_RADIUS).sum()),
    )


def score_arm(
    *,
    arm,
    x0,
    baseline_x0,
    structure,
    token_map,
    fixed_target,
    prep,
    controller,
    sc_stream,
    pack_steps,
    out_dir,
    target,
    seed,
    event,
    sequence_hash,
    seconds,
    denoiser_calls,
    injections,
    target_max_displacement,
    px_driver,
):
    """Score one arm's final structure. Three families, reported separately."""
    from pxf.couple import visibility as vis

    sidechain = list(atom37.SIDECHAIN_SLOTS)
    dense = px_driver_densify(x0, structure, controller)
    generated = dense[token_map.gen_to_px.to(dense.device)][None]

    # Repack under the SHARED sequence with paired packing randomness, so the
    # side chains differ only through the backbone they sit on.
    aatype = prep["aatype"]
    given = prep["given"]
    backbone_only = generated * given[..., None]
    seq_mask = torch.ones(backbone_only.shape[:2], device=backbone_only.device)
    with sc_stream.active():
        torch.manual_seed(abs(hash((target, seed, event, sequence_hash))) % (2**31))
        sidechains, _aux = controller.repack_on(
            backbone_only,
            aatype,
            seq_mask=seq_mask,
            residue_index=prep["residue_index"],
            chain_index=prep["chain_index"],
            num_steps=pack_steps,
        )
    full = backbone_only.clone()
    full[..., sidechain, :] = sidechains
    available = vis.predicted_availability(
        aatype, seq_mask, given, backbone_only, sidechains=sidechains
    ).available

    # --- isolated-chain quality ---
    ca = full[0, :, atom37.ATOM37.index("CA"), :]
    spacing = (ca[1:] - ca[:-1]).norm(dim=-1)
    bad_bond = float(((spacing - CA_IDEAL).abs() > CA_TOLERANCE).float().mean())
    atoms = full[0][available[0].bool()]
    residue_of = torch.arange(full.shape[1], device=full.device)[:, None].expand(
        full.shape[1], atom37.NUM_ATOM37
    )[available[0].bool()]
    intra = torch.cdist(atoms.float(), atoms.float())
    different = residue_of[:, None] != residue_of[None, :]
    # Exclude adjacent residues: their contacts are covalent geometry.
    apart = (residue_of[:, None] - residue_of[None, :]).abs() > 1
    intra_clashes = int(((intra < CLASH_RADIUS) & different & apart).sum() // 2)

    # --- target compatibility ---
    target_tokens = torch.nonzero(~token_map.design_mask).reshape(-1)
    target_dense = dense[target_tokens.to(dense.device)]
    target_atoms = target_dense.reshape(-1, 3)
    finite = torch.isfinite(target_atoms).all(-1) & (target_atoms.abs().sum(-1) > 0)
    min_dist, iface_clashes, iface_contacts = _pairwise_min(atoms, target_atoms[finite])

    # --- intervention effect ---
    keep = torch.ones(x0.reshape(-1, 3).shape[0], dtype=torch.bool, device=x0.device)
    drift = float(
        (x0.reshape(-1, 3)[keep] - baseline_x0.reshape(-1, 3)[keep])
        .float()
        .pow(2)
        .sum(-1)
        .mean()
        .sqrt()
    )

    name = f"{target}_seed{seed}_step{event[0]}_{arm}"
    path = Path(out_dir) / f"{name}.pt"
    torch.save(
        dict(
            generated_full_atom=full.detach().cpu(),
            available=available.detach().cpu(),
            aatype=aatype.detach().cpu(),
            target_dense=target_dense.detach().cpu(),
            complex_flat=x0.detach().cpu(),
        ),
        path,
    )
    return dict(
        target_id=target,
        generation_seed=seed,
        event_step=event[0],
        event_substage=event[1],
        arm=arm,
        sequence_hash=sequence_hash,
        final_structure_path=str(path),
        injection_count=injections,
        target_max_displacement=float(target_max_displacement or 0.0),
        denoiser_calls=denoiser_calls,
        seconds=seconds,
        generated_residues=int(generated.shape[1]),
        bad_ca_bond_fraction=bad_bond,
        ca_spacing_mean=float(spacing.mean()),
        intra_clashes=intra_clashes,
        interface_min_distance=min_dist,
        interface_clashes=iface_clashes,
        interface_contacts=iface_contacts,
        drift_from_baseline=drift,
        scored_atoms=int(available.sum()),
    )


def report(rows):
    """Print the three families separately. No native-RMSD column exists."""
    arms = sorted(
        {r["arm"] for r in rows},
        key=lambda a: (
            ("baseline", "bb_only", "full", "scrambled").index(a)
            if a in ("baseline", "bb_only", "full", "scrambled")
            else 9
        ),
    )

    def mean(arm, key):
        values = [
            r[key] for r in rows if r["arm"] == arm and isinstance(r.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else float("nan")

    print("\n=== generation stress test: one feedback event ===")
    print(
        f"  {len(rows)} row(s), {len({r['target_id'] for r in rows})} target(s), "
        f"{len({r['sequence_hash'] for r in rows})} unique sequence(s)\n"
    )
    families = (
        (
            "isolated-chain quality",
            ["bad_ca_bond_fraction", "ca_spacing_mean", "intra_clashes"],
        ),
        (
            "target compatibility",
            ["interface_min_distance", "interface_clashes", "interface_contacts"],
        ),
        (
            "intervention effect",
            ["drift_from_baseline", "target_max_displacement", "denoiser_calls", "seconds"],
        ),
    )
    for title, keys in families:
        print(f"  [{title}]")
        print(f"    {'metric':28s}" + "".join(f"{a:>13s}" for a in arms))
        for key in keys:
            cells = "".join(f"{mean(a, key):13.4f}" for a in arms)
            print(f"    {key:28s}{cells}")
        print()
    print(
        "  Refolding is emitted per UNIQUE sequence, so its pLDDT is shared\n"
        "  across arms and is not an arm-specific metric. No column scores the\n"
        "  generated chain against the native partner: there is no native for a\n"
        "  newly generated backbone.\n"
    )
