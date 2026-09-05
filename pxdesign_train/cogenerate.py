"""
Joint sequence-structure co-generation.

The sampler has two phases. First, the ordinary EDM trajectory generates an
initial backbone B_0. Then an explicit co-evolution loop repeatedly applies
S(B_k) followed by B_refine(B_k, feedback_k) to obtain B_(k+1). The refinement
loop carries the latest predicted backbone directly: it neither re-noises it nor
takes an Euler step between refinement rounds.

By default (`seq_mode="sequential"`) the design region is decoded a few positions
at a time, most confident first, each commitment written back into `restype` and
re-encoded so the positions still to be called can see it — LLaDA's
low-confidence ordering, which that paper measured as clearly better than
committing at random. `commit_strategy` chooses how many go per round.

`seq_mode="complete_unmask"` is the previous default: every position is
re-predicted from scratch every step, nothing is ever committed early, and the
answer is a single argmax at the last step. It is the right mode for an AA head
trained with `aa_mask_mode="all"`, which has never seen a partially decided
sequence.

This is a MINIMAL, correctness-first sampler (deterministic EDM Euler step, no
predictor-corrector / churn). Quality tuning is out of scope here.
"""
from typing import Any, Optional

import logging

import torch

from pxdesign_train.sampler import build_aa20_to_restype36, _unmask_counts


# 20-AA index -> 3-letter (matches PRO_STD order used elsewhere).
_AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


COMMIT_STRATEGIES = ("topk", "threshold", "random")


def _select_commits(
    *,
    strategy: str,
    conf: torch.Tensor,          # [N_token] confidence of the current prediction
    masked_idx: torch.Tensor,    # [M] still-masked design token indices
    k: int,                      # positions this step's schedule wants committed
    threshold: float,
    max_frac: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Which still-masked positions to freeze this step.

    Every strategy commits from `masked_idx` only, and returns token indices.

    "topk"       the schedule decides HOW MANY (k, an even split of the design
                 region over the sampling steps), confidence decides WHICH. This
                 is LLaDA's low-confidence remasking, which that paper measured
                 as clearly better than committing at random.

    "threshold"  the MODEL decides how many: everything above `threshold` goes
                 this round, whatever it can't call yet waits for the next one,
                 by which time the positions committed now are context. The
                 schedule is ignored.

    "random"     the control arm. Same count as "topk", chosen uniformly instead
                 of by confidence, so "does ranking by confidence actually buy
                 anything on proteins" is measurable here rather than cited from
                 a language-model paper.
    """
    if masked_idx.numel() == 0:
        return None

    if strategy == "topk":
        if k <= 0:
            return None
        k = min(int(k), int(masked_idx.numel()))
        return masked_idx[torch.topk(conf[masked_idx], k).indices]

    if strategy == "random":
        if k <= 0:
            return None
        k = min(int(k), int(masked_idx.numel()))
        perm = torch.randperm(masked_idx.numel(), generator=generator,
                              device=masked_idx.device)
        return masked_idx[perm[:k]]

    if strategy != "threshold":
        raise ValueError(
            f"commit_strategy must be one of {COMMIT_STRATEGIES}, got {strategy!r}"
        )

    masked_conf = conf[masked_idx]
    over = masked_conf > float(threshold)
    n_over = int(over.sum())

    # Two guards, because a bare threshold degenerates in both directions and the
    # direction depends on calibration, which is a property of the checkpoint
    # rather than of this code.
    #
    #   nothing clears the bar -> no commit -> the same state next step -> the
    #   loop runs out of steps having decided nothing. Commit the single most
    #   confident position so the trajectory always advances.
    #
    #   everything clears it -> the whole design region lands in round one,
    #   which is complete_unmask wearing a schedule: no position is ever chosen
    #   with another one's identity visible, i.e. exactly the context this mode
    #   exists to provide. Cap each round at `max_frac` of what is left.
    n_cap = max(1, int(masked_idx.numel() * float(max_frac)))
    n_take = max(1, min(n_over, n_cap))
    return masked_idx[torch.topk(masked_conf, n_take).indices]


@torch.no_grad()
def cogenerate(
    model,
    input_feature_dict: dict[str, Any],
    N_step: int = 20,
    temperature: float = 0.0,
    chunk_size: Optional[int] = None,
    sidechain_cycle: bool = False,
    sc_start_frac: float = 1.0,
    stop_on_seq_stable: bool = False,
    seq_patience: int = 3,
    # "sequential" decodes the design region a few positions at a time, most
    # confident first, writing each commitment back into restype so the
    # positions still to be called can see it. "complete_unmask" re-predicts
    # every position from scratch every step and never lets one identity inform
    # another; its answer is a single argmax at the final step.
    #
    # PREREQUISITE, and it is not optional. This mode hands the AA head a
    # falling aa_t and a partially filled restype. A head trained with
    # aa_mask_mode="all" has seen neither -- only aa_t == 1.0 and an
    # end-to-end masked design region -- so running it here is off-distribution
    # on both axes and will sample WORSE than complete_unmask. Training moved to
    # AA_MASK_MODE_TRAINING ("time_dependent") for exactly this reason; a head
    # trained after that change is the one this default assumes.
    #
    # Pass seq_mode="complete_unmask" to get the old behaviour back for a head
    # that predates it. See docs/aa_head_masking.md.
    seq_mode: str = "sequential",
    commit_strategy: str = "topk",
    commit_threshold: float = 0.9,
    commit_max_frac: float = 0.5,
    generator: Optional[torch.Generator] = None,
    refinement_steps: int = 3,
) -> dict[str, Any]:
    """Co-generate (backbone coordinates, residue sequence) from noise.

    Requires model.aa_input_source == "diffusion_internal" (needs a_token).
    `input_feature_dict` must be a real featurized input (for N_atom,
    atom_to_token_idx, design_token_mask, restype template, ...); its GT
    coordinates are NOT used — structure starts from noise.

    `seq_mode` selects how residue identities are produced along the reverse
    trajectory:
      * "complete_unmask" (DEFAULT): the design region stays FULLY MASKED as model
        input the whole trajectory. Every step re-predicts the entire design region
        from the current structure (predict-all); the final sequence is the prediction
        at the last, cleanest step. No commit, no freeze, no schedule. Matches training
        `residue_type.mask_mode="all"` — train == inference. The trunk is encoded once
        (restype never changes), so this path costs one trunk pass.
      * "sequential" (ABLATION, LLaDA-style): start fully masked and progressively
        commit the top-k highest-confidence positions each step (schedule via
        `_unmask_counts`), freezing them and writing their types into `restype`. To make
        this a REAL co-design loop (not open-loop), the trunk is RE-ENCODED each step so
        committed residues actually condition the next backbone step — otherwise restype
        never re-enters the diffusion module. Costs one trunk pass per step. To be a
        valid comparison this ablation needs a checkpoint trained with
        `mask_mode="time_dependent"` (an all-masked model never saw partial sequences).

    `refinement_steps` is the number of post-diffusion S -> B_refine updates.
    It is used only when `sidechain_cycle` and co-evolution are enabled.

    Returns {coordinate, sequence (aa20 per design token, -1 elsewhere),
             trajectory}.
    """
    from protenix.model.protenix import update_input_feature_dict

    assert model.aa_input_source == "diffusion_internal", (
        "cogenerate needs input_source='diffusion_internal' (a_token)."
    )
    if seq_mode not in ("complete_unmask", "sequential"):
        raise ValueError(
            f"seq_mode must be 'complete_unmask' or 'sequential', got {seq_mode!r}"
        )
    if refinement_steps < 0:
        raise ValueError(f"refinement_steps must be >= 0, got {refinement_steps}")
    if commit_strategy not in COMMIT_STRATEGIES:
        raise ValueError(
            f"commit_strategy must be one of {COMMIT_STRATEGIES}, "
            f"got {commit_strategy!r}"
        )
    if not 0.0 < commit_max_frac <= 1.0:
        raise ValueError(f"commit_max_frac must be in (0, 1], got {commit_max_frac}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if commit_strategy != "topk" and seq_mode != "sequential":
        raise ValueError(
            f"commit_strategy={commit_strategy!r} only applies to "
            'seq_mode="sequential"; complete_unmask commits nothing until the '
            "final step"
        )
    if sc_start_frac != 1.0:
        raise ValueError(
            "sc_start_frac no longer applies to the two-phase sampler; side-chain "
            "co-evolution starts after B_0 is generated. Leave it at 1.0 and use "
            "refinement_steps to control the refinement rollout."
        )
    model.eval()

    feat = dict(input_feature_dict)
    feat = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feat)
    feat = update_input_feature_dict(feat)
    N_atom = feat["atom_to_token_idx"].shape[-1]

    dtm = feat["design_token_mask"].bool()
    while dtm.dim() > 1:
        dtm = dtm.squeeze(0)
    positions = dtm.nonzero(as_tuple=False).squeeze(-1)
    N_token = dtm.shape[-1]

    aa20_to_36, xpb = build_aa20_to_restype36()

    restype = feat["restype"].clone()
    while restype.dim() > 2:
        restype = restype.squeeze(0)
    n_ch = restype.shape[-1]

    def one_hot(ch):
        v = torch.zeros(n_ch, device=restype.device, dtype=restype.dtype)
        v[ch] = 1.0
        return v

    for i in positions.tolist():  # design region starts fully masked (xpb)
        restype[i] = one_hot(xpb)
    feat["restype"] = restype.unsqueeze(0) if input_feature_dict["restype"].dim() == 3 else restype

    s_inputs, s_trunk, z_trunk = model.get_condition_embedding(feat, chunk_size=chunk_size)
    device = s_inputs.device
    dtype = s_inputs.dtype
    aa20_to_36 = aa20_to_36.to(device)

    sampled = torch.full((N_token,), -1, dtype=torch.long, device=device)
    # sequential ablation only: `still` marks positions not yet committed (frozen);
    # `counts` is the per-step reveal schedule. Unused in complete_unmask.
    still = torch.zeros(N_token, dtype=torch.bool, device=device)
    still[positions] = True

    noise_schedule = model.inference_noise_scheduler(
        N_step=N_step, device=device, dtype=dtype
    )
    x = noise_schedule[0] * torch.randn(1, N_atom, 3, device=device, dtype=dtype)
    counts = _unmask_counts(int(positions.numel()), N_step)

    trajectory = []
    final_aa_probs = None

    # --- inference-side cycle setup (Overleaf iterative co-evolution) ---
    sc_enabled = (sidechain_cycle and getattr(model, "enable_sidechain", False)
                  and getattr(model, "enable_coevolution", False))
    bb_atom_idx = {}
    # 4-wide (N, CA, C, O) atom indices — the atom rows the ATOM-level (q_direct)
    # feedback writes q'_bb back into. Only available from the featurizer's by-name
    # resolution; the positional fallback below cannot identify O, so q_direct simply
    # stays off there rather than guessing.
    bb_atom_idx4 = {}
    if sc_enabled:
        # The mu_ideal provider is process-global mutable state. Re-register the model's
        # own choice (sc_template_provider, set from sidechain.template_provider at
        # construction) so the sampler cannot silently run a DIFFERENT template
        # construction than the one the checkpoint was trained under — which is exactly
        # the failure test_train_inference_parity exists to prevent.
        _prov = getattr(model, "sc_template_provider", None)
        if _prov is not None:
            from pxdesign_train.sidechain import templates as _tpl

            _tpl.set_provider_by_name(_prov)
        a2t = feat["atom_to_token_idx"]
        while a2t.dim() > 1:
            a2t = a2t.squeeze(0)
        # Prefer the featurizer's BY-NAME resolution (sc_bb_atom_idx: [N_token, 4] =
        # (N, CA, C, O), -1 where absent) — the same tensor training uses. Falling back to
        # the positional aot[:3] assumes every token's atoms start (N, CA, C), which the
        # featurizer explicitly does NOT assume; two sources of truth here silently yield a
        # wrong frame on any token whose atom order differs.
        _bbidx = feat.get("sc_bb_atom_idx")
        if _bbidx is not None:
            while _bbidx.dim() > 2:
                _bbidx = _bbidx.squeeze(0)
            # Frame atoms only: the O column is allowed to be -1 on a token whose
            # N/CA/C are all present, so validity must be tested over 0:3 alone.
            for tok in positions.tolist():
                quad = _bbidx[tok].to(torch.long)      # (N, CA, C, O), -1 = absent
                tri = quad[:3]
                if int(tri.min()) >= 0:
                    bb_atom_idx[tok] = tri
                    bb_atom_idx4[tok] = quad
        else:
            logging.getLogger(__name__).warning(
                "cogenerate: sc_bb_atom_idx absent; falling back to positional N/CA/C "
                "(aot[:3]). This assumes backbone-first atom order — featurize with "
                "compute_sidechain=True to get the by-name resolution training uses."
            )
            for tok in positions.tolist():
                aot = (a2t == tok).nonzero(as_tuple=False).squeeze(-1)
                if aot.numel() >= 3:
                    bb_atom_idx[tok] = aot[:3]
    h_res_prime_inject = None  # persistent side-chain-aware h_res across steps
    a_sc_inject = None         # persistent per-token side-chain summary (a_direct)
    # persistent ATOM-level side-chain features for the 4 backbone atoms (q_direct):
    # q_sc_inject [Nc, 4, c_atom] paired row-for-row with q_idx_inject [Nc, 4].
    q_sc_inject = None
    q_idx_inject = None
    # M3: keep the latest decoded side-chain global coords per committed token so
    # the final result carries a full-atom (backbone + S_phi side-chain) output.
    sidechain_out: dict[int, dict[str, Any]] = {}
    _prev_seq: tuple = ()          # sequence-stabilization tracking (paper termination)
    _seq_stable = 0

    diffusion_transitions = list(zip(noise_schedule[:-1], noise_schedule[1:]))
    # B_refine uses the same explicit conditioning value as training B_post. It is
    # a refinement-mode label AND the EDM residual scale; no noise of this
    # magnitude is added back to B_k. In particular, never use
    # `noise_schedule[-2]` merely because B_0 came from the sampler's final step.
    # For the 200-step sigma_data=16 schedule that value is ~0.007689, giving
    #
    #   D(x; sigma) = 0.99999977*x + 0.007689*F,
    #
    # which makes ordinary refinement updates nearly identity and lies in an
    # extreme tail of the training noise distribution. The fixed value 2.0 gives
    # c_skip=0.9846 and c_out=1.985. c_out is not a hard displacement cap because
    # F is unbounded, but it is the practical update scale. Keeping this constant
    # in train and inference also prevents sigma from pretending to be the physical
    # noise level of an already-denoised B_k.
    refine_sigma = noise_schedule.new_tensor(
        float(getattr(model, "sc_refinement_sigma", 2.0))
    )
    n_refine = int(refinement_steps) if sc_enabled else 0
    iterations = [
        ("diffusion", i, sig_t, sig_next)
        for i, (sig_t, sig_next) in enumerate(diffusion_transitions)
    ] + [
        ("refinement", i, refine_sigma, refine_sigma)
        for i in range(n_refine)
    ]
    counts = counts + [0] * max(0, len(iterations) - len(counts))

    for step, (phase, phase_step, sig_t, sig_next) in enumerate(iterations):
        is_refinement = phase == "refinement"
        feat["restype"] = restype.unsqueeze(0) if input_feature_dict["restype"].dim() == 3 else restype
        sigma = sig_t.reshape(1).to(dtype)

        # P3 fix (sequential ablation only): committed residues were written into
        # `restype` by prior steps, but `restype` is consumed ONLY by the trunk
        # (InputFeatureEmbedder), which we encoded once before the loop. Re-encode it
        # here so those commitments actually condition this backbone step — without
        # this the progressive commit is open-loop (restype never reaches the diffusion
        # module) and the ablation is meaningless. complete_unmask keeps the single
        # outer encode (restype never changes → nothing to re-encode).
        if seq_mode == "sequential" and step > 0:
            s_inputs, s_trunk, z_trunk = model.get_condition_embedding(
                feat, chunk_size=chunk_size
            )

        # The diffusion phase generates B_0 without side-chain feedback. Every
        # subsequent call is B_refine(B_k, feedback_k), matching the one-cycle
        # training forward. The latest B_k is passed directly as x_noisy below.
        s_trunk_step = s_trunk
        if (
            is_refinement
            and
            sc_enabled
            and getattr(model, "sc_hres_inject", True)
            and h_res_prime_inject is not None
        ):
            s_trunk_step = s_trunk + model.hres_injector(h_res_prime_inject).to(s_trunk.dtype)
        if is_refinement:
            pass_embedding = getattr(model, "refinement_pass_embedding", None)
            if pass_embedding is not None:
                s_trunk_step = s_trunk_step + pass_embedding.to(s_trunk.dtype)

        # Arm the direct a-level injection for THIS backbone call only. a_sc_inject is the
        # side-chain summary from the PREVIOUS step, mirroring how h_res_prime_inject
        # is persisted. The finally-clause
        # disarms even on exception, so step 0 can never inherit a live flag.
        # q_direct's call-key registry exists ONLY to make the forward and the
        # activation-checkpoint RECOMPUTE agree during training. Inference runs under
        # no_grad -> there is no recompute, so the registry is dead weight; left
        # uncleared it strong-references every step's q_skip for the whole sampling run.
        model._q_inject_calls = {}
        model._a_sc_cache = a_sc_inject if is_refinement else None
        # One flag arms BOTH a-level injection points: `sc_a_direct` (hook on
        # layernorm_a, after the global attention) and `sc_a_direct_pre` (pre-hook
        # on the DiffusionTransformer, before it). Only the hooks the model
        # actually registered can fire, so arming both is safe and keeps the two
        # arms symmetric at sampling.
        model._a_direct_active = bool(
            (
                getattr(model, "sc_a_direct", False)
                or getattr(model, "sc_a_direct_pre", False)
            )
            and is_refinement
            and a_sc_inject is not None
        )
        # Same for the ATOM-level channel (sidechain.q_direct): the decoder pre-hook
        # rewrites the 4 backbone atom rows of q_skip with q'_bb, using the q_sc_bb
        # S_phi produced at the PREVIOUS step ("only available after the first-round").
        # Without this the trained QAtomFusion would be dead weight at sampling — the
        # exact train/inference mismatch a_direct was just fixed for.
        model._q_sc_cache = q_sc_inject if is_refinement else None
        model._q_bb_idx_cache = q_idx_inject if is_refinement else None
        model._q_direct_active = bool(
            is_refinement
            and getattr(model, "sc_q_direct", False)
            and q_sc_inject is not None
        )
        try:
            x_denoised = model.diffusion_module(
                x_noisy=x, t_hat_noise_level=sigma, input_feature_dict=feat,
                s_inputs=s_inputs, s_trunk=s_trunk_step, z_trunk=z_trunk,
                pair_z=None, p_lm=None, c_l=None,
            )
        finally:
            model._a_direct_active = False
            model._q_direct_active = False
        if is_refinement:
            # The co-evolution state is the latest predicted backbone itself.
            # No Euler transition and no fresh noise exist inside this loop.
            x = x_denoised
        else:
            # Ordinary EDM generation of the initial backbone B_0.
            d = (x - x_denoised) / sig_t
            x = x + (sig_next - sig_t) * d
            if phase_step == len(diffusion_transitions) - 1 and sc_enabled:
                # Start refinement from the denoiser's final clean prediction,
                # even for custom schedules whose terminal sigma is non-zero.
                x = x_denoised

        # Structure-aware sequence step from the captured a_token.
        a = model._a_token_cache
        if a is None:
            continue
        a_red = model._reduce_a_token(a, sigma).to(dtype)  # [.., N_token, c_token]
        # aa_t = the mask fraction the AA head is conditioned on. complete_unmask is
        # always fully masked (1.0); sequential shrinks it as positions get committed.
        if seq_mode == "sequential":
            aa_t = float(still[positions].float().mean()) if positions.numel() else 0.0
        else:
            aa_t = 1.0
        logits = model.design_residue_type_head(
            a_red, aa_t=torch.tensor(aa_t, device=device)
        ).float()
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # [N_token, 20]
        probs = torch.softmax(logits, dim=-1)
        final_aa_probs = probs
        if temperature and temperature > 0:
            # Sample the identity from a tempered distribution, but score
            # confidence with the UNTEMPERED probability. Temperature is a
            # diversity knob for *which* residue gets chosen; confidence has to
            # keep meaning "how sure is the model", because the commit order
            # ranks on it. Reading confidence off the tempered distribution
            # would make a high temperature look like high certainty.
            #
            # ProteinMPNN-class methods run this low -- 0.1, sometimes 1e-4 --
            # and get diversity from sampling many sequences and scoring them,
            # not from loosening any single one.
            tempered = torch.softmax(logits / float(temperature), dim=-1)
            pred = torch.multinomial(tempered, 1, generator=generator).squeeze(-1)
            conf = probs.gather(-1, pred[..., None]).squeeze(-1)
        else:
            conf, pred = probs.max(dim=-1)

        if seq_mode == "sequential":
            # LLaDA-style progressive commit: choose some still-masked positions,
            # FREEZE them, and write their types into restype (re-encoded into the
            # trunk at the top of the next step — see P3 fix). `commit_strategy`
            # selects how many and which; see `_select_commits`.
            masked_idx = still.nonzero(as_tuple=False).squeeze(-1)
            reveal = _select_commits(
                strategy=commit_strategy,
                conf=conf,
                masked_idx=masked_idx,
                k=counts[step] if step < len(counts) else 0,
                threshold=commit_threshold,
                max_frac=commit_max_frac,
                generator=generator,
            )
            if reveal is not None and reveal.numel() > 0:
                sampled[reveal] = pred[reveal]
                still[reveal] = False
                for j in reveal.tolist():
                    restype[j] = one_hot(int(aa20_to_36[sampled[j]]))
        else:
            # complete_unmask (default): sequence stays fully masked as input; keep the
            # running full argmax so the side-chain cycle can instantiate atoms from the
            # current best guess. restype is NOT written back (design region stays masked
            # all trajectory); the final sequence is this prediction at the last step.
            sampled[positions] = pred[positions]

        # Seed S_0 from the final diffusion prediction B_0, then recompute S_k
        # after every refinement result B_k. Earlier noisy diffusion states never
        # enter the co-evolution loop. `sc_start_frac` is retained for API
        # compatibility but no longer gates this explicit post-diffusion phase.
        last_diffusion = (
            not is_refinement
            and phase_step == len(diffusion_transitions) - 1
        )
        run_sidechain = sc_enabled and (last_diffusion or is_refinement)
        committed = []
        if run_sidechain:
            # Tokens whose current predicted type feeds the side-chain cycle this step.
            # sequential: only frozen (committed) tokens qualify. complete_unmask:
            # nothing is frozen, so every design token with a frame + current argmax
            # qualifies (its type is this step's running prediction in `sampled`).
            if seq_mode == "sequential":
                committed = [int(j) for j in positions.tolist()
                             if (not bool(still[j])) and (j in bb_atom_idx) and int(sampled[j]) >= 0]
            else:
                committed = [int(j) for j in positions.tolist()
                             if (j in bb_atom_idx) and int(sampled[j]) >= 0]
            if committed:
                from pxdesign_train.sidechain.frames import build_frame, to_global, to_local
                from pxdesign_train.sidechain.instantiate import (
                    sidechain_atom_name_ids, sidechain_atoms, sidechain_mask,
                )
                from pxdesign_train.sidechain import init as sc_init
                from pxdesign_train.sidechain.coevolution import pool_side_chain_atoms

                # a_hat: the current full-sequence prediction. This is the SINGLE type
                # source for this block: it produces the atom set (ids / m) and the
                # ideal template below, exactly as the training path derives both from
                # one type. `sampled` is in the 20-class AA index space == _AA3 ==
                # instantiate.STD_AA_3 order, which template_init_local expects.
                a_hat = sampled[torch.as_tensor(committed, device=device)].long()  # [Nc]
                restypes3 = [_AA3[int(i)] for i in a_hat.tolist()]
                ids = sidechain_atom_name_ids(restypes3).to(device)
                m = sidechain_mask(restypes3).to(device)
                # F_hat MUST come from x_denoised (x_hat_0), NOT from the noisy sample x.
                # Training builds it with frames_from_backbone_index(out["x_denoised"], ...).
                # At the first side-chain step sigma is still ~10^3 A, so a frame built from
                # x would be an essentially random rotation with |t| ~ 10^3 A — and since the
                # frame-aware head routes ALL of S_phi's output through it, that is not a
                # rounding error, it is garbage propagated into h_res' and every later step.
                xc = x_denoised.squeeze(0)
                Ns = torch.stack([xc[bb_atom_idx[j][0]] for j in committed]).float()
                CAs = torch.stack([xc[bb_atom_idx[j][1]] for j in committed]).float()
                Cs = torch.stack([xc[bb_atom_idx[j][2]] for j in committed]).float()
                R, t = build_frame(Ns, CAs, Cs)  # F_hat from x_hat_0, as in training
                # Overleaf par.221, inference half: "side-chain atoms are initialized
                # from residue-specific ideal templates around the predicted backbone
                # frames", with the residue type = a_hat. Must mirror model.py's
                # training block exactly (same switches, same frame), otherwise S_phi
                # is sampled off the input distribution it was trained on.
                if getattr(model, "sc_template_init", False):
                    # 0714 appendix Step 2: the rotamer is conditioned on the predicted
                    # backbone's (phi, psi). These are chain-wide quantities — phi_i needs
                    # residue i-1's C — so they are computed over ALL tokens from x_denoised
                    # and only then restricted to the committed ones. Training does exactly
                    # the same thing (model._template_phi_psi); if these two drift apart,
                    # S_phi is sampled off the input distribution it was trained on.
                    phi_c = psi_c = None
                    _ri = feat.get("residue_index")
                    _ai = feat.get("asym_id")
                    if _bbidx is not None and _ri is not None and _ai is not None:
                        from pxdesign_train.sidechain.frames import backbone_phi_psi

                        _phi, _psi = backbone_phi_psi(
                            xc.detach().cpu().float(),
                            _bbidx.detach().cpu().long(),
                            _ri.detach().cpu().reshape(-1),
                            _ai.detach().cpu().reshape(-1),
                        )
                        sel = torch.as_tensor(committed)
                        phi_c, psi_c = _phi[sel], _psi[sel]
                    noisy_local = sc_init.template_init_local(
                        a_hat.cpu(), m.cpu(),
                        sigma_T=(
                            0.0 if getattr(model, "sc_edm", False)
                            else getattr(model, "sc_init_sigma_T", sc_init.DEFAULT_SIGMA_T)
                        ),
                        phi=phi_c, psi=psi_c,
                    )
                else:
                    noisy_local = sc_init.gaussian_init_local(
                        m.cpu(), sigma=model.sc_init_sigma
                    )
                noisy_local = noisy_local.to(device).to(dtype)
                a_full = a_red.squeeze(0) if a_red.dim() == 3 else a_red   # [N_token, c]
                h_c = a_full[committed]
                l_c = logits[committed]
                if not getattr(model, "sc_type_logits_input", True):
                    # Mirror sidechain.type_logits_input: training zeroes the type
                    # logits so w_aa sees a uniform distribution. Sampling a model
                    # trained that way with the real logits would feed the channel an
                    # input it never saw -- the exact class of mismatch this sampler
                    # has shipped three times already.
                    l_c = torch.zeros_like(l_c)
                # Sigma-embedding = this step's real noise level (EDM c_noise),
                # matching per-sigma training — not a constant.
                # Honour sidechain.per_sigma: training feeds S_phi a CONSTANT t=1 whenever
                # per_sigma is off (the Stage II-A warmup config), not the sigma embedding.
                if getattr(model, "sc_per_sigma", True):
                    sc_t = (0.25 * sig_t.reshape(1).clamp_min(1e-4).log()).to(device)
                else:
                    sc_t = torch.ones(1, device=device)
                # SINGLE FRAME CONTRACT (see SideChainModule.forward): S_phi is
                # handed global coordinates, always. The rotamer template is
                # generated residue-locally, so it is mapped out here exactly once,
                # the same way the training path does it. Any recentring for the
                # per-atom embedding happens inside the module.
                noisy_in = to_global(
                    noisy_local[None].float(), R[None].float(), t[None].float()
                ).to(dtype)
                # Frame-aware head (sidechain.frame_aware_head): hand S_phi the same
                # rigid frame training gives it, so it regresses local offsets and the
                # known transform maps them to global. Output space stays global.
                _fa = getattr(model, "sc_frame_aware_head", False)
                # ATOM-level channel (sidechain.q_direct): hand S_phi the residue's 4
                # backbone atoms (N, CA, C, O) in its own LOCAL frame — the SAME 14-slot
                # axis training builds, from the SAME source (the predicted backbone
                # x_denoised, gathered by name at sc_bb_atom_idx, mapped through F_hat).
                # S_phi returns their post-attention features (q_sc_bb), which the next
                # backbone step fuses into the Backbone Module's own 4 atom rows.
                sc_kwargs = {}
                q_idx_c = None
                # Gate on bb_context, NOT on q_direct. The 14-slot axis is its OWN switch:
                # the `bbctx` control arm trains with bb_context=True and q_direct=False, so
                # gating here on q_direct would run S_phi 10-slot at sampling after training
                # it 14-slot — sampling a model in a mode it was never trained in, and
                # corrupting the very control (q - bbctx) that isolates the atom channel.
                # (q_direct implies bb_context, so this still covers the q arms.)
                if getattr(model, "sc_bb_context", False) and all(
                    j in bb_atom_idx4 for j in committed
                ):
                    q_idx_c = torch.stack([bb_atom_idx4[j] for j in committed]).to(device)
                    v4 = (q_idx_c >= 0)                                   # [Nc, 4]
                    bb4 = xc[q_idx_c.clamp_min(0)].float()                # [Nc, 4, 3]
                    bb4 = bb4 * v4[..., None].to(bb4.dtype)
                    # Same frame as the ten side-chain slots they are concatenated
                    # to — `w_xyz` is a single Linear over that axis.
                    sc_kwargs = {"bb_coords": bb4[None].to(dtype)}
                    # B->S gather (sidechain.q_bs): bb_q from THIS ROUND's cached
                    # encoder q_skip (model._q_skip_cache, populated by the
                    # `_q_skip_encoder_hook` during the `model.diffusion_module(...)`
                    # call above), at the SAME (N, CA, C, O) atom indices as bb_local.
                    # Mirrors the training gather in model.py (the sc_q_bs block,
                    # ~line 1332): gather rows by q_idx_c.clamp_min(0), zero out
                    # where q_idx_c < 0. Read-only — no scatter-back (that is
                    # q_direct's job, via the decoder pre-hook on the NEXT step's
                    # refinement pass).
                    if getattr(model, "sc_q_bs", False) and model._q_skip_cache is not None:
                        q_skip_c = model._q_skip_cache
                        n_atom_q, c_q_ = q_skip_c.shape[-2], q_skip_c.shape[-1]
                        q_flat = q_skip_c.reshape(-1, n_atom_q, c_q_).to(device)
                        if q_flat.shape[0] == 1 and int(q_idx_c.max()) < n_atom_q:
                            bb_q = q_flat[0][q_idx_c.clamp_min(0)].float()  # [Nc, 4, c_q]
                            bb_q = bb_q * v4[..., None].to(bb_q.dtype)
                            sc_kwargs["bb_q"] = bb_q[None].to(dtype)
                        elif not getattr(model, "_warned_q_bs_shape", False):
                            logging.getLogger(__name__).warning(
                                "cogenerate: sidechain.q_bs=True but q_skip_cache "
                                "batch %d != 1 (or atom index out of range) — "
                                "skipping the bb_q gather for this step.",
                                q_flat.shape[0],
                            )
                            model._warned_q_bs_shape = True
                # CONTEXT (sidechain.context_aware): training lets S_phi's cross-residue
                # attention key on the receptor / motif / ligand tokens. Sampling MUST do
                # the same or the trained module runs blind to the thing it packs against.
                # Here S_phi's token axis is only the COMMITTED residues, so we append the
                # context tokens as KEY-ONLY rows: no side-chain slots (mask all False, so
                # they decode nothing), h_res as their feature, their real CA as position.
                # Everything below slices back to the first Nc rows.
                h_e, l_e, ids_e, m_e = h_c[None], l_c[None], ids[None], m[None]
                noisy_e, ca_e, ctx_e = noisy_in, t[None].float(), None
                R_e, t_e = R, t
                Nc = h_c.shape[0]
                _center = feat.get("sc_token_center_idx")
                if getattr(model, "sc_context_aware", False) and _center is not None and _bbidx is not None:
                    _c = _center.to(device).long().reshape(-1)                 # [N_token]
                    _isb = (_bbidx.to(device).long()[..., :3] >= 0).all(dim=-1)  # [N_token]
                    _cx = torch.nonzero((_c >= 0) & ~_isb, as_tuple=True)[0]     # context tokens
                    if _cx.numel() > 0:
                        Nx = int(_cx.numel())
                        ctx_ca = xc[_c[_cx]].float()                            # [Nx, 3]
                        h_e = torch.cat([h_c, a_full[_cx].to(h_c.dtype)], 0)[None]
                        l_e = torch.cat([l_c, logits[_cx].to(l_c.dtype)], 0)[None]
                        ids_e = torch.cat([ids, ids.new_zeros(Nx, ids.shape[-1])], 0)[None]
                        m_e = torch.cat([m, m.new_zeros(Nx, m.shape[-1])], 0)[None]
                        noisy_e = torch.cat(
                            [noisy_in[0], noisy_in.new_zeros(Nx, noisy_in.shape[-2], 3)], 0
                        )[None]
                        ca_e = torch.cat([t.float(), ctx_ca], 0)[None]
                        ctx_e = torch.cat(
                            [torch.zeros(Nc, dtype=torch.bool, device=device),
                             torch.ones(Nx, dtype=torch.bool, device=device)], 0
                        )[None]
                        # Frames for the padded rows are never used (we slice to :Nc), but
                        # they must be finite: identity rotation at the context CA.
                        R_e = torch.cat([R.float(), torch.eye(3, device=device).expand(Nx, 3, 3)], 0)
                        t_e = torch.cat([t.float(), ctx_ca], 0)
                        if "bb_coords" in sc_kwargs:
                            _bl = sc_kwargs["bb_coords"][0]
                            sc_kwargs["bb_coords"] = torch.cat(
                                [_bl, _bl.new_zeros(Nx, _bl.shape[-2], 3)], 0
                            )[None]
                            # Without res_mask the padded rows would get 4 VALID backbone
                            # slots, making them look like real side-chain residues.
                            sc_kwargs["res_mask"] = ctx_e.logical_not()
                        if "bb_q" in sc_kwargs:
                            # bb_q is consumed over the FULL L axis (module.py's
                            # `u[:, :, :n_bb, :]`), so it needs the same context-row
                            # padding as bb_local above, or the two tensors' L axes
                            # would disagree.
                            _bq = sc_kwargs["bb_q"][0]
                            sc_kwargs["bb_q"] = torch.cat(
                                [_bq, _bq.new_zeros(Nx, _bq.shape[-2], _bq.shape[-1])], 0
                            )[None]
                if getattr(model, "sc_edm", False):
                    # A2: a reverse loop that CARRIES its estimate, instead of the
                    # single re-initialised decode this sampler does per backbone
                    # step. Same number of S_phi calls buys N steps of refinement
                    # rather than one, because each step starts from the last.
                    from pxdesign_train.sidechain.edm import sidechain_reverse_loop

                    sch = model.sc_noise_sampler.schedule(
                        int(getattr(model, "sc_edm_infer_steps", 8)),
                        device=device, dtype=torch.float32,
                    )
                    x_init = (
                        noisy_e.float()
                        + sch[0] * torch.randn_like(noisy_e.float())
                    ).to(noisy_e.dtype)
                    y0_global, aux = sidechain_reverse_loop(
                        model.sc_edm_denoiser, x_init, sch, ca_e,
                        h_e, l_e, ids_e, m_e,
                        frame_R=(R_e[None].float() if _fa else None),
                        frame_t=(t_e[None].float() if _fa else None),
                        ctx_mask=ctx_e,
                        atom_mask=m_e,
                        return_aux=True,
                        **sc_kwargs,
                    )
                    sc_out = (y0_global, *aux)
                else:
                    sc_out = model.sidechain_module(
                        h_e, l_e, ids_e, m_e, noisy_e,
                        sc_t, ca_coords=ca_e,
                        frame_R=(R_e[None].float() if _fa else None),
                        frame_t=(t_e[None].float() if _fa else None),
                        ctx_mask=ctx_e,
                        **sc_kwargs,
                    )
                bb_feats = None
                if len(sc_out) == 3:
                    y0_global, atom_feats, bb_feats = sc_out
                else:
                    y0_global, atom_feats = sc_out
                # Drop the key-only context rows: only the committed residues decode.
                y0_global = y0_global[:, :Nc]
                atom_feats = atom_feats[:, :Nc]
                if bb_feats is not None:
                    bb_feats = bb_feats[:, :Nc]
                y0_global = y0_global.float()[0]  # [Nc, A, 3]
                for ci, j in enumerate(committed):
                    names = sidechain_atoms(restypes3[ci])
                    k = len(names)
                    sidechain_out[int(j)] = {
                        "restype3": restypes3[ci],
                        "atom_names": names,
                        "coords": y0_global[ci, :k].detach().cpu(),  # [k, 3] global
                    }
                h_prime = model.sidechain_feedback(atom_feats, m[None], h_c[None]).squeeze(0)
                full = a_full.clone()
                full[committed] = h_prime.to(full.dtype)
                h_res_prime_inject = full.unsqueeze(0) if s_trunk.dim() == 3 else full
                # DIRECT a-level feedback (sidechain.a_direct): cache the SAME per-token
                # side-chain summary training caches, so the next backbone step consumes the
                # fused token a'_bb = a_bb + MLP(concat(a_bb, W a_sc)). Without this the
                # trained ATokenFusion is silently dropped at sampling and every one of its
                # parameters is dead weight — a hard train/inference mismatch.
                if getattr(model, "sc_a_direct", False) or getattr(
                    model, "sc_a_direct_pre", False
                ):
                    a_sc_c = pool_side_chain_atoms(atom_feats, m[None]).squeeze(0)  # [Nc, c_atom]
                    a_sc_full = a_sc_c.new_zeros((a_full.shape[0], a_sc_c.shape[-1]))
                    a_sc_full[committed] = a_sc_c
                    a_sc_inject = a_sc_full
                # DIRECT q-level (atom) feedback: cache q_sc_bb + the atom indices it
                # belongs to. The rows are a SUBSET of tokens (the committed residues) —
                # the hook writes back by ATOM INDEX, so it needs no full-token axis.
                if bb_feats is not None and q_idx_c is not None:
                    q_sc_inject = bb_feats[0]                             # [Nc, 4, c_atom]
                    q_idx_inject = q_idx_c                                # [Nc, 4]

        trajectory.append({
            "step": step,
            "phase": phase,
            "phase_step": phase_step,
            "sigma": float(sig_t),
            "mask_frac": (float(still[positions].float().mean()) if seq_mode == "sequential"
                          else 1.0) if positions.numel() else 0.0,
            "mean_conf": float(conf[positions].mean()) if positions.numel() else 0.0,
            "sc_committed": len(committed) if run_sidechain else 0,
        })

        # Paper: terminate the iterative refinement when the predicted sequence
        # stabilizes for `seq_patience` steps. Off by default -> keep the full fixed
        # EDM schedule.
        if stop_on_seq_stable and (is_refinement or not sc_enabled):
            cur = tuple(sampled[positions].cpu().tolist())
            if step > 0 and cur == _prev_seq:
                _seq_stable += 1
                if _seq_stable >= seq_patience:
                    trajectory[-1]["early_stop"] = True
                    break
            else:
                _seq_stable = 0
            _prev_seq = cur

    # M3: full-atom assembly — backbone coords from diffusion + S_phi side-chain
    # global coords per committed design residue (empty dict if the cycle was off
    # or nothing committed). Each entry: {restype3, atom_names, coords[k,3]}.
    return {
        "coordinate": x.squeeze(0),
        "sequence": sampled,
        # Final-step probabilities are exposed for leakage-free inference
        # evaluation. They come from the same generated-backbone state as the
        # returned sequence; no label/GT coordinates enter cogenerate().
        "aa_probs": final_aa_probs,
        "trajectory": trajectory,
        "sidechain": sidechain_out,
        "has_full_atom_sidechain": bool(sidechain_out),
    }
