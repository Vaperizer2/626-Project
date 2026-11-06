"""The portable bundle's index and packaged model manifests describe the same models."""
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def test_extracted_bundle_index_has_no_missing_manifests(tmp_path):
    staged = tmp_path / "staged"
    manifests = set()
    for line in (ROOT / "apps/gpui-demo/packaging/bundle.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source, *destination = line.split()
        dest = Path(destination[0] if destination else source)
        if dest.suffix == ".json":
            out = staged / dest
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, out)
        if dest.name == "model.json":
            manifests.add(dest.relative_to("exports").as_posix())
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="bundle")
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "extracted", filter="data")
    exports = tmp_path / "extracted/bundle/exports"
    index = json.loads((exports / "models.json").read_text())
    assert set(index["manifests"]) == manifests
    heads = set()
    for path in index["manifests"]:
        manifest = json.loads((exports / path).read_text())
        heads.update(head["id"] for head in manifest["heads"])
        for head in manifest["heads"]:
            assert (exports / path).parent.joinpath(head["task"]["labels"]).is_file()
    assert set(index["default"]) <= heads
    assert "small" in heads
    assert "deployed_crop100" not in heads
