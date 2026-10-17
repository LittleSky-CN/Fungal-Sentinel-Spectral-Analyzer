import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.image_processing import (
    DEFAULT_BLACK_LEVEL,
    _black_level_from_page,
    get_exposure_time,
)


class DngMetadataTests(unittest.TestCase):
    def test_reads_rational_black_level_array(self):
        page = SimpleNamespace(tags={
            50714: SimpleNamespace(dtype=5, value=(0, 100, 0, 100, 0, 100, 0, 100))
        })

        self.assertEqual(_black_level_from_page(page), 0.0)

    def test_uses_configured_black_level_when_tag_is_missing(self):
        page = SimpleNamespace(tags={})

        self.assertEqual(_black_level_from_page(page), DEFAULT_BLACK_LEVEL)

    @patch("core.image_processing.tiff.TiffFile")
    def test_reads_exposure_from_exif_ifd(self, open_tiff):
        page = SimpleNamespace(tags={
            33434: None,
            34665: SimpleNamespace(value={
                "ExposureTime": (200_000_000, 1_000_000_000)
            }),
        })
        tif = MagicMock()
        tif.__enter__.return_value.pages = [page]
        open_tiff.return_value = tif

        self.assertEqual(get_exposure_time("sample.dng"), 0.2)

    @patch("core.image_processing.tiff.TiffFile")
    def test_defaults_exposure_when_metadata_is_absent(self, open_tiff):
        page = SimpleNamespace(tags={})
        tif = MagicMock()
        tif.__enter__.return_value.pages = [page]
        open_tiff.return_value = tif

        self.assertEqual(get_exposure_time("sample.dng"), 1.0)


if __name__ == "__main__":
    unittest.main()
