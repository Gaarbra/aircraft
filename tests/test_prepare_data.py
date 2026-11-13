"""Dataset preparation checks without network downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_data import prepare_split


class PreparationTests(unittest.TestCase):
    def test_empty_source_stops_without_requesting_more_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("scripts.prepare_data.get_rows", return_value={"rows": []}) as fetch:
                with self.assertRaisesRegex(RuntimeError, "Not enough suitable"):
                    prepare_split("train", 1, 1, set(), set(), data_dir=Path(folder))
                fetch.assert_called_once_with("train", 0)
            self.assertFalse((Path(folder) / "train" / "manifest.json").exists())

    def test_different_counts_do_not_overwrite_cached_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            split = Path(folder) / "train"
            split.mkdir()
            manifest = split / "manifest.json"
            original = json.dumps({"samples": []})
            manifest.write_text(original, encoding="utf-8")
            with patch("scripts.prepare_data.get_rows") as fetch:
                with self.assertRaisesRegex(ValueError, "Cached split counts differ"):
                    prepare_split("train", 1, 1, set(), set(), data_dir=Path(folder))
                fetch.assert_not_called()
            self.assertEqual(manifest.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
