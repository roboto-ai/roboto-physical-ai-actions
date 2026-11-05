"""A tagged conversion's files are all in the dataset before its tag can fire a trigger."""
import pathlib

from roboto_to_lerobot.output_publish import parse_tags, publish_tagged_output


class FakeDataset:
    def __init__(self, tags=()):
        self.tags = list(tags)
        self.calls = []
        self.uploaded = {}

    def upload_files(self, files, file_destination_paths, print_progress):
        self.calls.append("upload")
        self.uploaded = {file_destination_paths[path]: path.read_text() for path in files}

    def remove_tags(self, tags):
        self.calls.append(("remove", list(tags)))
        self.tags = [tag for tag in self.tags if tag not in tags]

    def put_tags(self, tags):
        self.calls.append(("put", list(tags)))
        self.tags = [*self.tags, *tags]


def _conversion(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir = tmp_path / "output"
    subfolder = output_dir / "iv_abc"
    (subfolder / "combined" / "meta").mkdir(parents=True)
    (subfolder / "combined" / "meta" / "info.json").write_text("info")
    (subfolder / "manifest.json").write_text("manifest")
    (output_dir / ".metadata").mkdir()
    (output_dir / ".metadata" / "changeset.json").write_text("runtime's own")
    return output_dir, subfolder


def test_files_are_uploaded_before_the_tag_and_not_left_for_the_runtime(tmp_path):
    output_dir, subfolder = _conversion(tmp_path)
    dataset = FakeDataset()

    publish_tagged_output(dataset, output_dir, subfolder, ["lerobot_complete"])

    assert dataset.calls == ["upload", ("put", ["lerobot_complete"])]
    assert dataset.uploaded == {
        "iv_abc/combined/meta/info.json": "info",
        "iv_abc/manifest.json": "manifest",
    }
    assert not subfolder.exists()
    assert (output_dir / ".metadata" / "changeset.json").exists()


def test_a_tag_the_dataset_already_has_is_added_again_so_its_trigger_fires(tmp_path):
    output_dir, subfolder = _conversion(tmp_path)
    dataset = FakeDataset(tags=["lerobot_complete", "keep_me"])

    publish_tagged_output(dataset, output_dir, subfolder, ["lerobot_complete"])

    assert dataset.calls == ["upload", ("remove", ["lerobot_complete"]), ("put", ["lerobot_complete"])]
    assert sorted(dataset.tags) == ["keep_me", "lerobot_complete"]


def test_output_tags_parameter_ignores_blanks():
    assert parse_tags(None) == []
    assert parse_tags(" , ") == []
    assert parse_tags("lerobot_complete, demo ,") == ["lerobot_complete", "demo"]
