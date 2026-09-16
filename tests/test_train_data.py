"""Cropping, padding and structural noise, per Appendix B.1 / B.3."""

import pytest
import torch

from pxf.train import data as D


def test_contiguous_crop_is_a_single_span():
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        indices = D.contiguous_crop(500, 128, generator=generator)
        assert len(indices) == 128
        assert bool((indices.diff() == 1).all()), "crop must be one contiguous span"
        assert 0 <= int(indices[0]) and int(indices[-1]) < 500


def test_short_examples_are_not_cropped():
    assert len(D.contiguous_crop(50, 256)) == 50


def test_crop_start_covers_the_whole_feasible_range():
    generator = torch.Generator().manual_seed(1)
    starts = {int(D.contiguous_crop(20, 10, generator=generator)[0]) for _ in range(300)}
    # Appendix B.3.1: start uniform on [0, length - size] = [0, 10].
    assert starts == set(range(11))


def test_multimer_crop_respects_the_budget_and_both_chains():
    generator = torch.Generator().manual_seed(0)
    for lengths in ([100, 150], [300, 20], [10, 400], [64, 64]):
        indices = D.multimer_contiguous_crop(lengths, 128, generator=generator)
        assert len(indices) == min(128, sum(lengths))
        assert int(indices.max()) < sum(lengths)
        assert len(torch.unique(indices)) == len(indices), "no residue twice"


def test_multimer_crop_needs_exactly_two_chains():
    with pytest.raises(ValueError, match="two chains"):
        D.multimer_contiguous_crop([10, 20, 30], 16)


def test_multimer_crop_keeps_everything_when_it_fits():
    assert len(D.multimer_contiguous_crop([10, 20], 128)) == 30


def test_spatial_crop_seeds_at_the_interface():
    # Two well-separated blobs; the crop should straddle the closest pair.
    a = torch.randn(100, 3) + torch.tensor([0.0, 0, 0])
    b = torch.randn(100, 3) + torch.tensor([50.0, 0, 0])
    a[-1] = torch.tensor([24.0, 0, 0])
    b[0] = torch.tensor([26.0, 0, 0])
    coords = torch.cat([a, b])
    chains = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    indices = D.spatial_crop(coords, chains, 40)
    assert len(indices) == 40
    assert len(torch.unique(chains[indices])) == 2, "interface crop should span both chains"


def test_noise_is_independent_gaussian_of_the_requested_scale():
    generator = torch.Generator().manual_seed(0)
    x = torch.zeros(2000, 37, 3)
    noised = D.add_structural_noise(x, 0.3, generator=generator)
    assert noised.std() == pytest.approx(0.3, abs=0.01)
    assert noised.mean() == pytest.approx(0.0, abs=0.01)


def test_zero_noise_is_exactly_a_no_op():
    x = torch.randn(4, 37, 3)
    assert torch.equal(D.add_structural_noise(x, 0.0), x)
    assert torch.equal(D.add_structural_noise(x, None), x)


def test_cluster_index_samples_one_member_per_cluster(tmp_path):
    csv = tmp_path / "clusters.csv"
    csv.write_text("cluster,path\nc1,a.pdb\nc1,b.pdb\nc2,c.pdb\n")
    index = D.ClusterIndex.from_csv(csv)
    assert len(index) == 2
    picks = index.sample(torch.Generator().manual_seed(0))
    assert len(picks) == 2
    assert picks[1] == "c.pdb" and picks[0] in ("a.pdb", "b.pdb")


def test_cluster_index_rejects_an_empty_file(tmp_path):
    empty = tmp_path / "e.csv"
    empty.write_text("cluster,path\n")
    with pytest.raises(ValueError, match="No cluster,path rows"):
        D.ClusterIndex.from_csv(empty)


def test_dataset_produces_fixed_size_batches_with_padding_marked():
    from pxf.provenance import repo_root

    paths = [
        str(repo_root() / f"fampnn/data/casp14/pdbs/{n}.pdb") for n in ("T1031", "T1024")
    ]
    dataset, loader = D.build_loader(
        paths, batch_size=2, crop_size=96, noise=0.3, shuffle=False
    )
    batch = next(iter(loader))
    for key in D.BATCH_KEYS:
        assert batch[key].shape[:2] == (2, 96), key
    # T1031 is 95 residues, so exactly one padding position is marked.
    assert batch["seq_mask"].sum(-1).tolist() == [95.0, 96.0]
    assert "x_input" in batch, "noised view is exposed separately"


def test_epoch_changes_the_crop():
    from pxf.provenance import repo_root

    path = str(repo_root() / "fampnn/data/casp14/pdbs/T1024.pdb")  # 391 residues
    dataset = D.StructureCropDataset([path], crop_size=64, seed=0)
    dataset.set_epoch(0)
    first = dataset[0]["x"].clone()
    dataset.set_epoch(1)
    second = dataset[0]["x"]
    assert not torch.equal(first, second), "a new epoch should resample the crop"
