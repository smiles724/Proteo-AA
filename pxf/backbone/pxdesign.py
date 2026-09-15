"""Full official PXDesign as the backbone module.

Nothing about the generator is reimplemented. PXDesign's own
:class:`~pxdesign.runner.inference.InferenceRunner` builds the official
``ProtenixDesign`` network, restores the published ``pxdesign_v0.1.0``
checkpoint, and featurizes inputs through PXDesign's own dataset -- exactly as
``pxdesign inference`` would. The only difference is the tail: instead of
dumping CIF files, generated backbones are captured as atom37 tensors and handed
to the side-chain module in memory.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import torch

from pxf import atom37, bridge, provenance


@dataclass
class BackboneBatch:
    """One PXDesign input target and the ``N_sample`` backbones generated for it."""
    sample_name: str
    coords_af2: torch.Tensor      # [N_sample, L, 37, 3], shared AF2 atom37 order
    atom_mask_af2: torch.Tensor   # [N_sample, L, 37]
    design_mask: torch.Tensor     # [L] bool; True where PXDesign designed the token
    native_sequence: str          # one-letter, "X" where no native identity exists
    sequence_known: torch.Tensor  # [L] bool; False at design tokens
    residue_index: torch.Tensor   # [L]
    chain_index: torch.Tensor     # [L]
    atom_array: Any = None        # biotite AtomArray, aligned with PXDesign's atom axis
    atom_to_token_idx: Optional[torch.Tensor] = None
    dropped_atoms: list = field(default_factory=list)

    @property
    def num_samples(self):
        return int(self.coords_af2.shape[0])

    @property
    def length(self):
        return int(self.coords_af2.shape[1])


class PXDesignBackbone:
    """Official PXDesign generator, returning atom37 tensors.

    ``checkpoint_dir`` must contain ``<model_name>.pt``. ``input_json`` is a
    PXDesign inference input, i.e. the same file ``pxdesign inference`` takes.
    """

    def __init__(self, *, input_json, checkpoint_dir, dump_dir,
                 model_name=provenance.PXDESIGN_MODEL_NAME, n_sample=8, n_step=200,
                 use_msa=False, dtype="bf16", load_strict=True, extra_args=(),
                 download_cache=True, strict_sources=True):
        self.sources = provenance.runtime_sources(strict=strict_sources,
                                                  components=("pxdesign", "protenix"))
        self.model_name = model_name
        self.dump_dir = Path(dump_dir).resolve()
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(checkpoint_dir).resolve() / f"{model_name}.pt"
        if not checkpoint.exists():
            raise ValueError(
                f"PXDesign checkpoint not found: {checkpoint}. Point "
                f"--pxdesign-checkpoint-dir at a directory holding {model_name}.pt "
                "(PXDesign downloads it on first run, or reuse an existing copy).")
        self.checkpoint = checkpoint

        argv = ["--input_json_path", str(Path(input_json).resolve()),
                "--load_checkpoint_dir", str(checkpoint.parent),
                "--dump_dir", str(self.dump_dir),
                "--model_name", model_name,
                "--use_msa", str(bool(use_msa)),
                "--dtype", str(dtype),
                "--load_strict", str(bool(load_strict)),
                "--sample_diffusion.N_sample", str(int(n_sample)),
                "--sample_diffusion.N_step", str(int(n_step)),
                *[str(a) for a in extra_args]]

        from pxdesign.utils.infer import (convert_to_bioassembly_dict,
                                          download_inference_cache, get_configs)
        from pxdesign.utils.inputs import process_input_file
        configs = get_configs(argv)
        configs.input_json_path = process_input_file(configs.input_json_path,
                                                     out_dir=str(self.dump_dir))
        if download_cache:
            download_inference_cache(configs)
        with open(configs.input_json_path) as stream:
            tasks = json.load(stream)
        for task in tasks:
            convert_to_bioassembly_dict(task, str(self.dump_dir))
        resolved = self.dump_dir / "input_tasks.json"
        with resolved.open("w") as stream:
            json.dump(tasks, stream, indent=4)
        configs.input_json_path = str(resolved)

        from pxdesign.runner.inference import InferenceRunner
        self.runner = InferenceRunner(configs)
        self.configs = configs
        self.identity = dict(
            backend="pxdesign", model_name=model_name,
            weights=provenance.weight_record(checkpoint),
            n_sample=int(n_sample), n_step=int(n_step), dtype=str(dtype),
            use_msa=bool(use_msa), upstream=self.sources,
            atom_mapping=atom37.mapping_record())

    @property
    def device(self):
        return self.runner.device

    @property
    def model(self):
        """The official ``ProtenixDesign`` instance."""
        return self.runner.model

    def __len__(self):
        return len(self.runner.dataset)

    def generate(self, *, seed=None, deterministic=False):
        """Yield a :class:`BackboneBatch` per input target."""
        from protenix.utils.seed import seed_everything
        if seed is not None:
            seed_everything(seed=int(seed), deterministic=deterministic)
        for batch in self.runner.design_test_dl:
            data, atom_array, error = batch[0]
            if error:
                raise RuntimeError(f"PXDesign input error: {error}")
            prediction = self.runner.predict(data)
            yield self._to_atom37(data, atom_array, prediction)

    def _to_atom37(self, data, atom_array, prediction):
        feat = data["input_feature_dict"]
        atom_to_token = feat["atom_to_token_idx"].reshape(-1).long().cpu()
        num_tokens = int(feat["token_index"].reshape(-1).shape[0])
        coordinate = prediction["coordinate"]
        if coordinate.dim() < 3:
            raise ValueError(f"Unexpected PXDesign coordinate shape {tuple(coordinate.shape)}")
        coords = coordinate.reshape(-1, coordinate.shape[-2], 3).detach().float().cpu()
        if coords.shape[-2] != atom_to_token.numel():
            raise ValueError(
                f"PXDesign returned {coords.shape[-2]} atoms but atom_to_token_idx has "
                f"{atom_to_token.numel()}; the atom axes must align")
        if len(atom_array) != atom_to_token.numel():
            raise ValueError(f"atom_array has {len(atom_array)} atoms, feature dict has "
                             f"{atom_to_token.numel()}")

        coords37, mask37, dropped = bridge.atoms_to_atom37(
            coords, list(atom_array.atom_name), atom_to_token, num_tokens)
        design_mask = bridge.design_mask_from_res_names(
            list(atom_array.res_name), atom_to_token, num_tokens)
        sequence, known = bridge.native_sequence(
            list(atom_array.res_name), atom_to_token, num_tokens)
        return BackboneBatch(
            sample_name=str(data.get("sample_name", "pxdesign")),
            coords_af2=coords37, atom_mask_af2=mask37, design_mask=design_mask,
            native_sequence=sequence, sequence_known=known,
            residue_index=self._token_tensor(feat, atom_array, atom_to_token, num_tokens, "res_id"),
            chain_index=self._token_tensor(feat, atom_array, atom_to_token, num_tokens, "chain_id"),
            atom_array=atom_array, atom_to_token_idx=atom_to_token, dropped_atoms=dropped)

    @staticmethod
    def _token_tensor(feat, atom_array, atom_to_token, num_tokens, annotation):
        """Per-token integer annotation, preferring the feature dict when present."""
        if annotation == "res_id" and "residue_index" in feat:
            return feat["residue_index"].reshape(-1)[:num_tokens].long().cpu()
        if annotation == "chain_id" and "asym_id" in feat:
            return feat["asym_id"].reshape(-1)[:num_tokens].long().cpu()
        values = bridge.token_reduce(list(getattr(atom_array, annotation)),
                                     atom_to_token, num_tokens, how="first")
        if annotation == "chain_id":
            order = {name: i for i, name in enumerate(dict.fromkeys(map(str, values)))}
            return torch.tensor([order[str(v)] for v in values], dtype=torch.long)
        return torch.tensor([int(v) for v in values], dtype=torch.long)
