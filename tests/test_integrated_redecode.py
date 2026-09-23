"""Exercise the real integrated entry point with small deterministic donors.

Real mapping, shared-prelogit hooks, event construction, solver, and final pack
wiring run here. These are contract tests, not evidence about trained donors.
"""
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pxf import atom37
from pxf.bench.integrated import run_integrated, select_event
from pxf.bench.integrated_redecode import REDECODE_SEED_OFFSET
from pxf.couple.integrated_event import prepare_event
from pxf.couple.pxdesign_iface import BackboneTap, ConditioningFeedback
from pxf.couple.replay import RngStream, run_trajectory


class Conditioning(nn.Module):
    def forward(self):
        return torch.zeros(1, 3, 4), torch.zeros(3, 3, 2)


class Denoiser:
    device = torch.device('cpu')
    n_atom = 12

    def __init__(self):
        diffusion = nn.Module()
        diffusion.layernorm_a = nn.Identity()
        diffusion.atom_attention_decoder = nn.Identity()
        diffusion.diffusion_conditioning = Conditioning()
        self.model = NS(diffusion_module=diffusion)
        self.calls = 0
        self.inputs = []

    def schedule(self, n_step):
        return torch.tensor([2., 1., .5, .1])

    def denoise(self, x, sigma, *, tap, feedback=None):
        self.calls += 1
        self.inputs.append((x.clone(), sigma.clone(), feedback))
        tap.feedback = feedback
        try:
            s, _ = self.model.diffusion_module.diffusion_conditioning()
            self.model.diffusion_module.layernorm_a(s + 0.1)
            # Feedback changes geometry and a_token through the actual tap.
            local = torch.tensor([[-1., 0., 0.], [0., 0., 0.],
                                  [0., 1., 0.], [0., 2., 0.]])
            xyz = local.repeat(3, 1).reshape(1, 3, 4, 3)
            xyz[:, :, :, 2] += torch.tensor([0., 10., 20.])[None, :, None]
            xyz[:, :, 2, 1] += s[:, :, 0]
            return xyz.reshape_as(x) + x * 0.001
        finally:
            tap.feedback = None


class SeqModule(nn.Module):
    no_aatype_pred = False

    def __init__(self):
        super().__init__()
        self.W_out = nn.Linear(4, 21)
        with torch.no_grad():
            self.W_out.weight.zero_()
            self.W_out.bias.fill_(-100.)
            self.W_out.bias[0] = .5
            self.W_out.bias[1] = 0.
            self.W_out.weight[1, 0] = 1.

    def forward(self, coords, aatype, *args):
        h = torch.zeros(*aatype.shape, 4)
        h[..., 0] = (coords[..., 2, 1] - coords[..., 1, 1]) - 1.
        return self.W_out(h), {'h_V': h}


class Adapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def delta_h(self, a_token, sigma):
        self.inputs.append((a_token.clone(), float(sigma)))
        return a_token * .25


class Designer:
    def __init__(self, extra_draws=0, fail_second=False):
        self.model = NS(denoiser=NS(seq_design_module=SeqModule()))
        self.design_inputs, self.pack_inputs = [], []
        self.extra_draws = extra_draws
        self.fail_second = fail_second

    def design(self, **kw):
        self.design_inputs.append({k: v.clone() if torch.is_tensor(v) else v
                                   for k, v in kw.items()})
        if self.fail_second and len(self.design_inputs) == 2:
            raise RuntimeError('decoder failed')
        # Mimic a donor that reseeds and consumes global randomness.
        torch.manual_seed(kw['seed'])
        np.random.seed(kw['seed'])
        torch.rand(17 + self.extra_draws)
        np.random.rand(13 + self.extra_draws)
        logits, _ = self.model.denoiser.seq_design_module(
            kw['coords_af2'], kw['aatype'])
        aa = torch.where(kw['fixed_sequence_mask'].bool(), kw['aatype'],
                         logits.argmax(-1))
        coords = kw['coords_af2'].clone()
        coords[..., 3, :] = coords[..., 1, :] + .5
        mask = torch.ones_like(kw['atom_mask'])
        return {'designs': [NS(aatype=aa[0], coords_af2=coords[0],
            atom_mask_af2=mask[0], sequence=atom37.sequence_from_aatype(aa[0]),
            psce=torch.zeros(3, 33))]}

    def __call__(self, **kw):
        self.pack_inputs.append(kw)
        self.model.denoiser.seq_design_module(kw['coords_af2'], kw['aatype'])
        return dict(coords_af2=kw['coords_af2'], aatype=kw['aatype'],
                    atom_mask_af2=kw['atom_mask'], psce=torch.ones(1, 3, 33))


def structure():
    return NS(num_tokens=3, design_mask=torch.tensor([1., 0., 1.]),
              topology=NS(atom_to_token_idx=torch.arange(3).repeat_interleave(4),
                          res_names=['xpb'] * 4 + ['ALA'] * 4 + ['xpb'] * 4,
                          atom_names=['N', 'CA', 'C', 'O'] * 3,
                          residue_index=torch.tensor([0, 0, 1]),
                          chain_index=torch.tensor([1, 0, 1])))


class Feedback:
    def __init__(self):
        self.calls = 0

    def __call__(self, packed, sigma):
        self.calls += 1
        assert packed.h_base is not None  # bb_only control remains executable
        return ConditioningFeedback(delta_single=torch.ones(1, 3, 4)), {}


def run(policy, feedback=True, extra_draws=0, adapters=True):
    denoiser, designer = Denoiser(), Designer(extra_draws)
    adapter = Adapters() if adapters else None
    conditioner = Feedback() if feedback else None
    sample = run_integrated(denoiser=denoiser, designer=designer,
        structure=structure(), adapters=adapter, conditioner=conditioner,
        event_sigma=1., n_step=3, seed=41, sequence_policy=policy)
    return sample, denoiser, designer, adapter, conditioner


@pytest.mark.parametrize('feedback', [False, True])
@pytest.mark.parametrize('policy', ['event_fixed', 'post_feedback_redesign'])
def test_actual_entrypoint_decode_counts_masks_and_final_sequence(policy, feedback):
    sample, denoiser, designer, adapter, conditioner = run(policy, feedback)
    expected = 1 + int(policy == 'post_feedback_redesign')
    assert len(designer.design_inputs) == expected
    assert len(designer.pack_inputs) == 1
    assert denoiser.calls == 4  # 3 solver calls + one provisional, no extra PX
    assert sample.diagnostics['conditioning_injections'] == int(feedback)
    assert sample.diagnostics['sequence_decode_passes'] == expected
    if conditioner:
        assert conditioner.calls == 1
    for inputs in designer.design_inputs:
        assert inputs['aatype'].tolist() == [[20, 0, 20]]
        assert inputs['fixed_sequence_mask'].tolist() == [[0., 1., 0.]]
        assert not inputs['atom_mask'][0, [0, 2]][:, list(atom37.SIDECHAIN_SLOTS)].any()
    assert torch.equal(designer.pack_inputs[0]['aatype'], sample.aatype)
    assert sample.aatype[0, 1] == 0  # fixed target
    if expected == 2:
        assert designer.design_inputs[1]['seed'] == 41 + REDECODE_SEED_OFFSET
        assert adapter.inputs[1][1] == adapter.inputs[0][1] == 1.
        if feedback:
            assert not torch.equal(adapter.inputs[0][0], adapter.inputs[1][0])
            assert sample.binder_sequence == 'RR'
            assert sample.diagnostics['event_sequence'] == 'AA'
        else:
            assert sample.binder_sequence == 'AA'
    else:
        assert sample.binder_sequence == 'AA'
    assert not designer.model.denoiser.seq_design_module._forward_hooks
    assert not denoiser.model.diffusion_module.layernorm_a._forward_hooks


def test_extra_decode_cannot_change_backbone_rng_or_add_feedback():
    fixed = run('event_fixed')[0]
    redesigned = run('post_feedback_redesign')[0]
    extra = run('post_feedback_redesign', extra_draws=5000)[0]
    assert torch.equal(fixed.x0, redesigned.x0)
    assert torch.equal(redesigned.x0, extra.x0)
    assert torch.equal(redesigned.aatype, extra.aatype)


def test_u03_also_receives_matched_second_decode():
    sample, _, designer, _, _ = run('post_feedback_redesign', feedback=False, adapters=False)
    assert len(designer.design_inputs) == 2
    assert sample.diagnostics['redecode_hook_calls'] == 0
    assert sample.diagnostics['conditioning_injections'] == 0


def test_second_decode_failure_removes_all_hooks():
    denoiser, designer = Denoiser(), Designer(fail_second=True)
    with pytest.raises(RuntimeError, match='decoder failed'):
        run_integrated(denoiser=denoiser, designer=designer, structure=structure(),
            adapters=Adapters(), conditioner=Feedback(), event_sigma=1., n_step=3,
            sequence_policy='post_feedback_redesign')
    assert not designer.model.denoiser.seq_design_module._forward_hooks
    diffusion = denoiser.model.diffusion_module
    assert not diffusion.layernorm_a._forward_hooks
    assert not diffusion.diffusion_conditioning._forward_hooks


def test_matrix_resumes_and_keeps_first_event_shared(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/run_integrated_binder_matrix.py'
    spec = importlib.util.spec_from_file_location('matrix_redecode_test', path)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    denoiser, designer, adapter = Denoiser(), Designer(), Adapters()
    schedule = denoiser.schedule(3)
    choice = select_event(schedule, 1.)
    with BackboneTap(denoiser.model.diffusion_module) as tap:
        _, records, _ = run_trajectory(
            denoise=lambda x, s, feedback=None: denoiser.denoise(x, s, tap=tap, feedback=feedback),
            schedule=schedule, n_atom=12, device='cpu', stream=RngStream('integrated', 41),
            record_steps={choice.step})
    packed_products = []
    def finalise(**kw):
        packed_products.append(kw['products'])
        return tmp_path / 'fake.pdb', kw['products'].coords_af2
    monkeypatch.setattr(matrix, '_finalise', finalise)
    (tmp_path / 'diagnostics').mkdir()
    args = NS(sequence_policy='post_feedback_redesign', context='complex_sc', step_scale_eta=2.5)
    kwargs = dict(feedback_path=None, conditioner_arm=None, adapters=adapter,
        recorded=records[0], denoiser=denoiser, structure=structure(), designer=designer,
        schedule=schedule, choices=[choice], prefix_id='test', name='test', length=2,
        gen_seed=41, bs_seed=0, args=args, out=tmp_path,
        api=dict(BackboneTap=BackboneTap, RngStream=RngStream,
                 run_trajectory=run_trajectory, prepare_event=prepare_event))
    first = matrix._one_arm(arm_label='J03', shared=None, **kwargs)
    shared = first['_shared']
    second = matrix._one_arm(arm_label='paired_control', shared=shared, **kwargs)
    assert second['_shared'] is shared
    assert all(p is not shared for p in packed_products)
    assert len(designer.design_inputs) == 3  # shared first + two separate second decodes
    assert torch.equal(packed_products[0].aatype, packed_products[1].aatype)
    assert first['redecode_seed'] == second['redecode_seed']
    assert set(first) - {'_shared'} <= set(matrix.ROW_COLUMNS)
    assert len(list((tmp_path / 'diagnostics').glob('*.redecode.pt'))) == 2


def test_redecode_rejects_missing_features_and_changed_sigma(monkeypatch):
    from pxf.bench.integrated_redecode import redecode_after_feedback
    common = dict(initial=NS(sigma=1.), corrected_bb=torch.zeros(1, 12, 3),
                  sigma=torch.tensor([1.]), structure=None, designer=None,
                  adapters=None, seed=1, context='complex_sc', design_id='x', target='x')
    with pytest.raises(RuntimeError, match='corrected a_token'):
        redecode_after_feedback(corrected_a_token=None, **common)
    common['sigma'] = torch.tensor([.5])
    with pytest.raises(ValueError, match='same actual event sigma'):
        redecode_after_feedback(corrected_a_token=torch.zeros(1, 3, 4), **common)


def test_report_refuses_to_pool_old_and_redesigned_policies(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/report_integrated_binder_matrix.py'
    spec = importlib.util.spec_from_file_location('report_redecode_test', path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    csv_path = tmp_path / 'designs.csv'
    csv_path.write_text('sample_id,sequence_policy\na,event_fixed\nb,post_feedback_redesign\n')
    with pytest.raises(ValueError, match='mixed sequence policies'):
        report.load_rows(csv_path)
    csv_path.write_text('sample_id\na\n')
    assert report.load_rows(csv_path) == [{'sample_id': 'a'}]


# ===================================================================== R2
# End-to-end through the real _one_arm with several events, both policies,
# and BOTH a control and a feedback arm. R1 and R2 were integration
# failures that unit tests missed: R2 in particular killed the controls
# before any feedback arm ran, and only surfaced when a control met a
# multi-event schedule under post_feedback_redesign.

def _matrix_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_integrated_binder_matrix.py'
    spec = importlib.util.spec_from_file_location('matrix_multi_event', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _multi_event_arm(policy, *, with_feedback, tmp_path, monkeypatch, n_step=6):
    from pxf.bench.integrated import select_events
    from pxf.couple.integrated_event import (assert_target_rows_untouched,
                                             mask_feedback)

    matrix = _matrix_module()
    denoiser, designer, adapter = Denoiser(), Designer(), Adapters()
    schedule = denoiser.schedule(n_step)
    choices = select_events(schedule, [4.0, 1.0])
    assert len({c.key for c in choices}) == 2, 'fixture needs two distinct events'

    with BackboneTap(denoiser.model.diffusion_module) as tap:
        _x, records, _s = run_trajectory(
            denoise=lambda x, s, feedback=None: denoiser.denoise(
                x, s, tap=tap, feedback=feedback),
            schedule=schedule, n_atom=12, device='cpu',
            stream=RngStream('integrated', 41), record_steps={choices[0].step})

    monkeypatch.setattr(matrix, '_finalise',
                        lambda **kw: (tmp_path / 'fake.pdb', kw['products'].coords_af2))
    (tmp_path / 'diagnostics').mkdir(exist_ok=True)
    args = NS(sequence_policy=policy, context='complex_sc', step_scale_eta=2.5,
              seq_steps=2, pack_steps=1, temperature=0.1,
              allow_feedback_policy_transfer=False,
              bs_checkpoint=['0=fake.pt'], checkpoint_dir=str(tmp_path),
              fampnn_checkpoint='fake.pt', fampnn_variant='0.3')
    conditioner = Feedback()
    api = dict(BackboneTap=BackboneTap, RngStream=RngStream,
               run_trajectory=run_trajectory, prepare_event=prepare_event,
               mask_feedback=mask_feedback,
               assert_target_rows_untouched=assert_target_rows_untouched,
               conditioning_widths=lambda model: (4, 2),
               node_feature_dim=lambda model: 4,
               token_feature_dim=lambda model: 4,
               expected_policy=lambda **k: {},
               load_feedback=lambda *a, **k: (conditioner, {}, {}))
    return matrix._one_arm(
        arm_label='E1' if with_feedback else 'U03',
        feedback_path=('fake.pt' if with_feedback else None),
        conditioner_arm=('early_s_full' if with_feedback else None),
        shared=None, adapters=adapter, recorded=records[0], denoiser=denoiser,
        structure=structure(), designer=designer, schedule=schedule,
        choices=choices, prefix_id='t', name='t', length=2, gen_seed=41,
        bs_seed=0, args=args, out=tmp_path, api=api), choices, conditioner


@pytest.mark.parametrize('policy', ['event_fixed', 'post_feedback_redesign'])
def test_multi_event_control_survives_both_policies(policy, tmp_path, monkeypatch):
    # R2: the control skipped the terminal event, so the re-decode was
    # validated against a first-event reference and the sigma guard killed
    # it. This is the exact combination that failed.
    row, choices, _cond = _multi_event_arm(
        policy, with_feedback=False, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert row['event_injections'] == 0
    assert row['n_events'] == len(choices)
    assert row['output_sequence']


@pytest.mark.parametrize('policy', ['event_fixed', 'post_feedback_redesign'])
def test_multi_event_feedback_arm_injects_once_per_event(policy, tmp_path, monkeypatch):
    row, choices, cond = _multi_event_arm(
        policy, with_feedback=True, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert cond.calls == len(choices)
    assert row['event_injections'] == len(choices)
    # R6: one norm per SCHEDULED event, so position k means event k
    assert len(row['per_event_feedback_norms'].split(';')) == len(choices)
    # R6: decodes are the work actually done, not a constant
    assert row['event_decodes'] >= len(choices)


def test_multi_event_feedback_changes_the_backbone_not_just_the_counters(
        tmp_path, monkeypatch):
    # Compare real outputs: a zero-payload conditioner must leave the
    # trajectory where a no-conditioner control leaves it, and a non-zero
    # one must not.
    control, _c, _ = _multi_event_arm(
        'event_fixed', with_feedback=False, tmp_path=tmp_path, monkeypatch=monkeypatch)
    treated, _c2, _ = _multi_event_arm(
        'event_fixed', with_feedback=True, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert treated['event_injections'] > control['event_injections']
    assert float(treated['min_bb_bb_distance']) != float(control['min_bb_bb_distance'])


def test_schedule_id_reaches_the_row_and_the_sample_id(tmp_path, monkeypatch):
    # R5: without this a one-event and a four-event run collide on filename
    # and on the reporter's pairing key.
    row, choices, _ = _multi_event_arm(
        'event_fixed', with_feedback=True, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert row['schedule_id'] and row['schedule_id'] != 'ev1'
    assert row['schedule_id'] in row['sample_id']
