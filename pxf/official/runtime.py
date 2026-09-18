"""Drive the official PXDesign runtime, with a recordable sampler.

Why this exists rather than the monomer driver in ``pxf.backbone.driver``: the
driver runs PXDesign's model against the Protenix this repo vendors
(``c3bfc36``), which is a different API from the ``v0.5.0+pxd`` (``d18aa1da``)
that PXDesign's own installer names. Under that pairing the generated backbone
interpenetrated its target. The official runtime, unmodified, does not --
see ``docs/target_conditioning_audit.md``.

Three deliberate departures from the local path, each with a reason:

* **No coordinate overwrites.** PXDesign conditions on the target through
  ``conditional_templ`` -- pairwise distances binned over 2..22 A embedded into
  ``z`` -- plus ``restype`` (32+4) and ``hotspot``. A distogram is invariant to
  rotation and translation, so the target's *pose* is chosen by the model, not
  given to it: measured, the emitted target matches its input to 0.107 A
  superposed while sitting 26 A away raw. Forcing the native frame back on it
  puts the binder and target in different frames. ``fixed_target`` is therefore
  never passed here.

* **eta 2.5.** The official CLI's constant, not Protenix's generic 1.5.

* **No ``prepare_features``.** That step precomputes ``d_lm``, ``v_lm`` and
  ``pad_info`` for ``c3bfc36``'s ``AtomAttentionEncoder``. The official encoder
  takes ``(input_feature_dict, r_l, s, z, ...)`` and derives them internally;
  those keys do not exist in ``v0.5.0+pxd``.

The conditioning is computed by the official ``get_condition_embedding`` and
the schedule by the official ``inference_noise_scheduler``, so the only thing
this module substitutes is the *loop*, which is a transcription pinned by test.
"""

from contextlib import nullcontext

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build_runner(yaml_path, out_dir, *, load_checkpoint_dir, n_step=200,
                 n_sample=1, use_msa=False, dtype="bf16",
                 eta_type="const", eta_min=2.5, eta_max=2.5, extra_argv=()):
    """Construct the official ``InferenceRunner``, exactly as ``main()`` does.

    The preamble (input conversion, cache download, bioassembly dicts) is
    upstream's; reproducing it by hand would be one more place for the local
    path to drift from the official one.
    """
    import json
    import os

    from protenix.config import save_config

    from pxdesign.runner.inference import InferenceRunner
    from pxdesign.utils.infer import (
        convert_to_bioassembly_dict,
        download_inference_cache,
        get_configs,
    )
    from pxdesign.utils.inputs import process_input_file

    argv = [
        "--input_json_path", str(yaml_path),
        "--dump_dir", str(out_dir),
        "--load_checkpoint_dir", str(load_checkpoint_dir),
        "--use_msa", "true" if use_msa else "false",
        "--dtype", dtype,
        "--sample_diffusion.N_step", str(n_step),
        "--sample_diffusion.N_sample", str(n_sample),
        # The CLI's defaults, which `get_configs` alone does not apply. Calling
        # get_configs directly falls back to configs_base's
        # `eta_schedule = {type: piecewise_65, min: 1.0, max: 2.5}`, while the
        # shipped `pxdesign infer` -- the runs that produced plausible
        # complexes -- uses a constant 2.5. Leaving this implicit made the
        # transcription check compare a constant-eta replay against a
        # piecewise-eta upstream and diverge by 49 A.
        "--eta_type", str(eta_type),
        "--eta_min", str(eta_min),
        "--eta_max", str(eta_max),
        *map(str, extra_argv),
    ]
    configs = get_configs(argv)
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = process_input_file(
        configs.input_json_path, out_dir=configs.dump_dir
    )
    download_inference_cache(configs)
    save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
    with open(configs.input_json_path, "r") as handle:
        orig_inputs = json.load(handle)
    for entry in orig_inputs:
        convert_to_bioassembly_dict(entry, configs.dump_dir)
    configs.input_json_path = os.path.join(configs.dump_dir, "input_tasks.json")
    with open(configs.input_json_path, "w") as handle:
        json.dump(orig_inputs, handle, indent=4)

    return InferenceRunner(configs), configs


def first_batch(runner):
    """The first usable sample from the official dataloader.

    Returns ``(data, atom_array)``. Errors are surfaced rather than skipped:
    upstream's loop logs and continues, which is right for a production sweep
    and wrong for a harness that must know exactly which target it measured.
    """
    for batch in runner.design_test_dl:
        data, atom_array, message = batch[0]
        if message:
            raise RuntimeError(f"featurization failed: {message}")
        return data, atom_array
    raise RuntimeError("dataloader produced no batches")


class OfficialDenoiser:
    """The official conditioning plus a ``denoise(x_noisy, sigma, feedback)``.

    ``get_condition_embedding`` and the key deletions are upstream's
    ``_design_inference_loop``, kept in that order because the deletions free
    the template tensors the conditioning has already consumed.
    """

    def __init__(self, runner, data, *, chunk_size=None):
        from protenix.utils.torch_utils import to_device

        self.runner = runner
        self.model = runner.model
        self.configs = runner.configs
        self.device = runner.device
        self.dtype = DTYPES[self.configs.dtype]
        self.chunk_size = chunk_size

        data = to_device(data, self.device)
        self.features = data["input_feature_dict"]
        self.atom_array_data = data

        with torch.no_grad(), self._amp():
            s_inputs, s, z = self.model.get_condition_embedding(
                input_feature_dict=self.features, chunk_size=chunk_size
            )
        for key in [
            k for k in self.features
            if "template_" in k
            or k in ("msa", "has_deletion", "deletion_value", "profile",
                     "deletion_mean", "token_bonds")
        ]:
            del self.features[key]
        # The conditioning is computed under autocast, exactly as upstream's
        # `predict` does. Sampling is not: `skip_amp.sample_diffusion` is true,
        # so `autocasting_disable_decorator` disables autocast for the whole
        # sampler *and* casts every top-level float tensor argument to fp32.
        # `input_feature_dict` is a dict, not a tensor, so it is passed through
        # uncast -- replicated here rather than tidied, because a denoiser run
        # in bf16 against an fp32 reference would differ for a reason that has
        # nothing to do with feedback.
        self.s_inputs_raw = s_inputs
        self.s_inputs = s_inputs.to(torch.float32)
        self.s_trunk = s.to(torch.float32)
        self.z_trunk = z.to(torch.float32)
        self.calls = 0
        self.injections = 0

    def _amp(self):
        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def _no_amp(self):
        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    @property
    def n_atom(self):
        return int(self.features["atom_to_token_idx"].shape[-1])

    def schedule(self, n_step=None):
        # dtype is the *pre-cast* conditioning dtype, as upstream: the schedule
        # is built at s_inputs.dtype (bf16 under the default config) and only
        # then cast to fp32 by the decorator. Building it directly in fp32
        # would give slightly different noise levels.
        n_step = int(n_step or self.configs.sample_diffusion["N_step"])
        return self.model.inference_noise_scheduler(
            N_step=n_step,
            device=self.s_inputs_raw.device,
            dtype=self.s_inputs_raw.dtype,
        )

    def denoise(self, x_noisy, sigma, *, feedback=None, tap=None):
        """One denoiser evaluation, optionally with a residual injected.

        ``feedback`` is handed to the tap rather than added to coordinates: the
        residual belongs in the token representation the adapter was trained
        against, not in ``x``.
        """
        t_hat = sigma.reshape(-1)[:1].to(torch.float32)
        if tap is not None:
            tap.feedback = feedback
        with torch.no_grad(), self._no_amp():
            out = self.model.diffusion_module(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=self.features,
                s_inputs=self.s_inputs,
                s_trunk=self.s_trunk,
                z_trunk=self.z_trunk,
                chunk_size=self.chunk_size,
                inplace_safe=False,
            )
        if tap is not None:
            tap.feedback = None
        self.calls += 1
        if feedback is not None:
            self.injections += 1
        return out.to(torch.float32)

    def official_sample(self, n_sample=1, n_step=None):
        """Upstream's own ``sample_diffusion``, for the uninterrupted arm."""
        return self.model.sample_diffusion(
            denoise_net=self.model.diffusion_module,
            input_feature_dict=self.features,
            s_inputs=self.s_inputs,
            s_trunk=self.s_trunk,
            z_trunk=self.z_trunk,
            N_sample=n_sample,
            noise_schedule=self.schedule(n_step),
            inplace_safe=False,
        )
