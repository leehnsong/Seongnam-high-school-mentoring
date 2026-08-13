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
