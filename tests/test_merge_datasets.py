import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import merge_datasets as md  # noqa: E402

UNIFIED = ["box", "bicycle", "stroller"]


def test_read_source_class_names(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"names": ["cardboard", "damaged"], "nc": 2}))
    assert md.read_source_class_names(p) == ["cardboard", "damaged"]


def test_read_source_class_names_handles_dict_form(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump({"names": {0: "cardboard", 1: "damaged"}, "nc": 2}))
    assert md.read_source_class_names(p) == ["cardboard", "damaged"]


def test_build_id_mapping_with_target_maps_every_class():
    mapping = md.build_id_mapping(["a", "b", "c"], {"target": "box"}, UNIFIED)
    assert mapping == {0: 0, 1: 0, 2: 0}


def test_build_id_mapping_with_class_map_drops_unlisted():
    spec = {"class_map": {"stroller": "stroller", "bicycle": "bicycle"}}
    names = ["human", "wheelchair", "suitcase", "stroller", "bicycle"]
    assert md.build_id_mapping(names, spec, UNIFIED) == {3: 2, 4: 1}


def test_build_id_mapping_rejects_unknown_unified_class():
    with pytest.raises(ValueError):
        md.build_id_mapping(["a"], {"target": "truck"}, UNIFIED)


def test_build_id_mapping_rejects_source_class_not_present():
    with pytest.raises(ValueError):
        md.build_id_mapping(["a", "b"], {"class_map": {"zzz": "box"}}, UNIFIED)


def test_remap_label_text_rewrites_class_ids():
    text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"
    assert md.remap_label_text(text, {0: 2, 1: 1}) == (
        "2 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"
    )


def test_remap_label_text_drops_unmapped_classes():
    text = "0 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.05 0.05\n"
    assert md.remap_label_text(text, {0: 1}) == "1 0.5 0.5 0.2 0.2\n"


def test_remap_label_text_skips_malformed_lines():
    text = "0 0.5 0.5\nnot a label\n1 0.2 0.2 0.1 0.1\n"
    assert md.remap_label_text(text, {0: 0, 1: 1}) == "1 0.2 0.2 0.1 0.1\n"


def test_remap_label_text_returns_empty_for_no_valid_rows():
    assert md.remap_label_text("9 0.1 0.1 0.1 0.1\n", {0: 0}) == ""


def _make_split(root: Path, stem: str, label_text: str):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (root / "labels" / f"{stem}.txt").write_text(label_text)


def test_merge_split_copies_and_prefixes(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "img1", "0 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="boxset", mapping={0: 0})
    assert stats["copied"] == 1
    assert (dst / "images" / "boxset__img1.jpg").exists()
    assert (dst / "labels" / "boxset__img1.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_merge_split_skips_image_without_label(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "images" / "lonely.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0})
    assert stats["copied"] == 0
    assert stats["skipped_no_label"] == 1


def test_merge_split_skips_when_all_boxes_dropped(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "img1", "5 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0})
    assert stats["copied"] == 0
    assert stats["skipped_empty_after_remap"] == 1


def test_write_data_yaml_has_three_classes(tmp_path):
    out = md.write_data_yaml(tmp_path, UNIFIED)
    data = yaml.safe_load(out.read_text())
    assert data["nc"] == 3
    assert data["names"] == UNIFIED
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert Path(data["path"]).is_absolute()


def test_reset_output_dir_creates_dir_with_gitkeep(tmp_path):
    out = tmp_path / "merged"
    md.reset_output_dir(out)
    assert out.is_dir()
    assert (out / ".gitkeep").exists()


def test_reset_output_dir_clears_previous_contents(tmp_path):
    out = tmp_path / "merged"
    (out / "images").mkdir(parents=True)
    (out / "images" / "stale.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    md.reset_output_dir(out)
    assert not (out / "images").exists()


def test_reset_output_dir_restores_gitkeep_after_wipe(tmp_path):
    out = tmp_path / "merged"
    (out / "images").mkdir(parents=True)
    (out / ".gitkeep").touch()
    md.reset_output_dir(out)
    assert (out / ".gitkeep").exists(), "재실행 후에도 .gitkeep이 남아야 한다"


def test_build_oversample_factors_maps_names_to_ids():
    assert md.build_oversample_factors({"bicycle": 8}, UNIFIED) == {1: 8}


def test_build_oversample_factors_handles_none_and_empty():
    assert md.build_oversample_factors(None, UNIFIED) == {}
    assert md.build_oversample_factors({}, UNIFIED) == {}


def test_build_oversample_factors_rejects_unknown_class():
    with pytest.raises(ValueError):
        md.build_oversample_factors({"truck": 2}, UNIFIED)


def test_build_oversample_factors_rejects_bad_factor():
    with pytest.raises(ValueError):
        md.build_oversample_factors({"bicycle": 0}, UNIFIED)
    with pytest.raises(ValueError):
        md.build_oversample_factors({"bicycle": "8"}, UNIFIED)


def test_oversample_factor_defaults_to_one():
    assert md.oversample_factor("0 0.5 0.5 0.2 0.2\n", {}) == 1
    assert md.oversample_factor("0 0.5 0.5 0.2 0.2\n", {1: 8}) == 1


def test_oversample_factor_returns_factor_for_present_class():
    assert md.oversample_factor("1 0.5 0.5 0.2 0.2\n", {1: 8}) == 8


def test_oversample_factor_returns_max_of_present_classes():
    text = "1 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    assert md.oversample_factor(text, {1: 8, 2: 3}) == 8


def test_merge_split_duplicates_oversampled_images(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "bike1", "0 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="b", mapping={0: 1}, factors={1: 3})
    assert stats["copied"] == 1
    assert stats["duplicated"] == 2
    assert (dst / "images" / "b__bike1.jpg").exists()
    assert (dst / "images" / "b__bike1__dup1.jpg").exists()
    assert (dst / "images" / "b__bike1__dup2.jpg").exists()
    assert (dst / "labels" / "b__bike1__dup2.txt").read_text() == "1 0.5 0.5 0.2 0.2\n"


def test_merge_split_does_not_duplicate_without_factors(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "img1", "0 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0})
    assert stats["duplicated"] == 0
    assert len(list((dst / "images").iterdir())) == 1


def test_merge_split_does_not_duplicate_unaffected_class(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_split(src, "boxonly", "0 0.5 0.5 0.2 0.2\n")
    stats = md.merge_split(src, dst, prefix="s", mapping={0: 0}, factors={1: 8})
    assert stats["duplicated"] == 0
    assert len(list((dst / "images").iterdir())) == 1


def test_main_returns_nonzero_when_nothing_merged(tmp_path, capsys, monkeypatch):
    config = tmp_path / "datasets.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "classes": UNIFIED,
                "sources": [{"name": "absent", "workspace": "w", "project": "p", "target": "box"}],
            }
        )
    )
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()
    out = tmp_path / "merged"
    monkeypatch.setattr(
        "sys.argv",
        ["merge_datasets.py", "--config", str(config), "--raw", str(empty_raw), "--out", str(out)],
    )
    assert md.main() == 1
    assert "학습 이미지가 하나도" in capsys.readouterr().out
