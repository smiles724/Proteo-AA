"""The one-pass backbone -> side-chain training graph, and its detach policy.

The whole experiment is a question about gradients: does a side-chain objective,
evaluated on a frozen FaMPNN reading a *predicted* backbone, improve that
backbone? So the graph has to be live in exactly the places the question is
about, and provably dead in the control.

    B_t = B* + sigma_B eps                     noisy native backbone
    B_hat = D_theta(B_t, sigma_B)              one backbone denoising call
    X37   = densify(B_hat)                     gradient-preserving
    h_V   = encode(X37, S, side chains hidden) frozen FaMPNN, live inputs
    q_hat = scn_denoiser(q_t, S, h_V)          frozen FaMPNN, live inputs
    C_hat = T(B_hat) q_hat                     predicted frames, predicted local

Three paths reach the backbone parameters, and they are not the same path:

    L_BB    -> B_hat
    L_local -> q_hat -> h_V -> X37 -> B_hat
    L_place -> q_hat -> h_V -> X37 -> B_hat   (encoder-mediated)
            -> T(B_hat)                        (direct, through the frames)

``detach_backbone=True`` is the negative control, and it detaches at *both*
entrances -- the encoder's input and the frame construction. Detaching only
``h_V`` leaves the frame path live, which would look like a control and train
the backbone anyway.

Nothing here freezes FaMPNN: that is the caller's ``requires_grad_(False)``.
Wrapping these calls in ``no_grad`` instead would also cut the input
derivatives, which are the entire mechanism.
"""

from dataclasses import dataclass

import torch

from pxf import atom37, bridge
from pxf.joint import data as joint_data


@dataclass
class JointForward:
    """Everything one training step produced, before any loss is taken."""

    sigma_b: torch.Tensor  # [1] the backbone noise level
    bb_noisy: torch.Tensor  # [N_atom, 3] what the backbone module saw
    bb_pred: torch.Tensor  # [N_atom, 3] B_hat on the flat axis
    bb_target: torch.Tensor  # [N_atom, 3] native, same axis
    bb_mask: torch.Tensor  # [N_atom] supervised backbone atoms
    coords37: torch.Tensor  # [1, L, 37, 3] densified prediction
    encoder_mask: torch.Tensor  # [1, L, 37] what the encoder was shown
    features: dict  # FaMPNN's encoder output
    prediction: object  # SidechainTrainingPrediction
    placed: torch.Tensor  # [(m b), L, 33, 3] C_hat in the shared frame
    placed_frames_exist: torch.Tensor = None  # [(m b), L] predicted frames
    detached_backbone: bool = False

    @property
    def multiplier(self):
        return int(self.prediction.multiplier)


def densify_prediction(bb_pred, topology, *, num_tokens=None):
    """Flat predicted atoms -> ``[1, L, 37, 3]`` plus the slots that were filled.

    ``bridge.atoms_to_atom37`` scatters rather than indexes, so autograd reaches
    ``bb_pred``; that is what makes the encoder a path to the backbone at all.
    """
    num_tokens = int(num_tokens or topology.num_tokens)
    dense, mask, _dropped = bridge.atoms_to_atom37(
        bb_pred, topology.atom_names, topology.atom_to_token_idx, num_tokens
    )
    if dense.dim() == 3:
        dense = dense.unsqueeze(0)
    mask = mask.reshape(-1, num_tokens, atom37.NUM_ATOM37).float()
    if mask.shape[0] != dense.shape[0]:
        mask = mask.expand(dense.shape[0], -1, -1)
    return dense, mask


def place_sidechains(local, bb_pred37, atom_mask_scn, *, backbone_mask=None):
    """Local-frame side chains onto predicted backbone frames.

    Uses FaMPNN's own ``transform_sidechain_frame``, in its own convention and
    in the direction it defines, rather than a second rotation convention that
    would have to be proved equivalent first.
    """
    from fampnn.data.data import transform_sidechain_frame

    from fampnn.data import residue_constants as rc

    x_bb = bb_pred37[..., rc.bb_idxs, :]
    if backbone_mask is None:
        backbone_mask = torch.ones(
            *x_bb.shape[:-1], device=x_bb.device, dtype=x_bb.dtype
        )
    placed, frames_exist = transform_sidechain_frame(
        local, x_bb, atom_mask_scn, backbone_mask, to_local=False
    )
    return placed, frames_exist


def joint_forward(
    driver,
    conditioning,
    fampnn,
    batch,
    *,
    sigma_b,
    backbone_noise,
    sidechain_noise=None,
    multiplier=None,
    self_cond_p=None,
    generator=None,
    detach_backbone=False,
):
    """One backbone denoising call, one encoding, and ``m`` side-chain clones.

    ``batch`` is a :class:`pxf.joint.data.JointRefinementBatch`. ``sigma_b`` and
    ``backbone_noise`` come from the run's named streams, so two arms can be
    handed the identical noisy backbone.
    """
    from pxf.couple import fampnn_iface as iface
    from pxf.train import step as train_step

    native = batch.native_batch
    sigma = torch.as_tensor(sigma_b, device=batch.device, dtype=torch.float32).reshape(-1)

    target = batch.backbone_target
    noise = backbone_noise.to(device=target.device, dtype=target.dtype)
    if noise.shape != target.shape:
        raise ValueError(
            f"backbone noise shaped {tuple(noise.shape)} for a target shaped "
            f"{tuple(target.shape)}"
        )
    # Only supervised atoms are perturbed; an unresolved slot has no coordinate
    # to add noise to and would otherwise enter the module as pure noise.
    bb_noisy = target + noise * sigma.to(target.dtype) * batch.backbone_mask_column

    bb_pred = driver.denoise_direct(conditioning, bb_noisy.unsqueeze(0), sigma)
    bb_pred = bb_pred.reshape(target.shape)

    # The control detaches at BOTH entrances. Detaching only the encoder's input
    # would leave the frame path live and the "blocked" arm would still train.
    bb_for_sidechains = bb_pred.detach() if detach_backbone else bb_pred

    coords37, supplied = densify_prediction(bb_for_sidechains, batch.topology)
    encoder_mask = joint_data.encoder_availability(
        native["aatype"], native["seq_mask"], supplied
    )
    _logits, _h_V, features = iface.encode(
        fampnn,
        coords37,
        native["aatype"],
        seq_mask=native["seq_mask"],
        residue_index=native["residue_index"],
        chain_index=native["chain_index"],
        atom_availability=encoder_mask,
    )

    prediction = train_step.sidechain_training_pass(
        fampnn,
        native,
        features,
        multiplier=multiplier,
        self_cond_p=self_cond_p,
        generator=generator,
        scn_mlm_mask=None,
        noise=sidechain_noise,
    )

    # Place every clone's prediction on the predicted frames. The frames are
    # cloned, not recomputed: one backbone, m side-chain draws.
    frames37 = prediction.clone(coords37)
    physical = prediction.clone(batch.physical_mask)
    # Frame validity follows the atoms the backbone module actually produced,
    # not an assumption that every residue has one.
    from fampnn.data import residue_constants as rc

    frame_atoms = prediction.clone(supplied[..., rc.bb_idxs])
    placed, placed_frames = place_sidechains(
        prediction.q_pred, frames37, physical, backbone_mask=frame_atoms
    )

    return JointForward(
        sigma_b=sigma,
        bb_noisy=bb_noisy,
        bb_pred=bb_pred,
        bb_target=target,
        bb_mask=batch.backbone_mask,
        coords37=coords37,
        encoder_mask=encoder_mask,
        features=features,
        prediction=prediction,
        placed=placed,
        placed_frames_exist=placed_frames,
        detached_backbone=bool(detach_backbone),
    )
