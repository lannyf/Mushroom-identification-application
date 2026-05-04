"""Dataset loader for the evaluation image collection.

Scans the folder structure under ``data/raw/evaluation_images/``,
maps folder names to canonical species_ids, and enriches each sample
with metadata such as in-distribution flags and Swedish-key support.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from benchmarks.config import EVAL_IMAGES_DIR, FOLDER_TO_SPECIES_ID, SPECIES_CSV, IN_DISTRIBUTION_SPECIES


@dataclass
class BenchmarkSample:
    """A single evaluation image with all associated ground-truth metadata.

    Attributes:
        image_path: Absolute path to the image file.
        image_bytes: Raw file contents (used by runners that need a byte stream).
        folder_name: Name of the parent folder (may be a species_id or descriptive name).
        species_id: Canonical species identifier (e.g. "AM.MU").
        english_name: Human-readable English name from ``species.csv``.
        in_distribution: True if the species is among the 7 CNN training classes.
        key_xml_support: How the Swedish decision tree handles this species:
            ``"exact"``, ``"lookalike_only"``, or ``"unsupported"``.
    """

    image_path: Path
    image_bytes: bytes
    folder_name: str
    species_id: str
    english_name: str
    in_distribution: bool
    key_xml_support: str


class GroundTruthDataset:
    """Loads and filters the evaluation image set.

    Iterating over this object yields ``BenchmarkSample`` instances in
    folder order. Convenience accessors split the set into in-distribution
    and out-of-distribution subsets.
    """

    def __init__(self, images_dir: Path = EVAL_IMAGES_DIR):
        self.samples: List[BenchmarkSample] = []
        self._id_to_name: dict = {}
        self._load_species_names()
        self._scan_images(images_dir)

    def _load_species_names(self):
        """Build a species_id → english_name lookup from ``species.csv``."""
        with open(SPECIES_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._id_to_name[row["species_id"]] = row["english_name"]

    def _scan_images(self, images_dir: Path):
        """Walk the evaluation folder and construct ``BenchmarkSample`` objects."""
        # Hard-coded support level for each species in the Swedish XML key.
        key_support = {
            "AM.MU": "unsupported",
            "AM.VI": "lookalike_only",
            "BO.ED": "exact",
            "BO.BA": "exact",
            "CA.CI": "exact",
            "CR.CO": "exact",
            "HY.PS": "lookalike_only",
            "CO.CO": "exact",
            "FO.BE": "unsupported",
            "LY.PE": "exact",
            "RA.BO": "exact",
            "SP.CR": "exact",
        }
        for folder in sorted(images_dir.iterdir()):
            if not folder.is_dir():
                continue
            species_id = FOLDER_TO_SPECIES_ID.get(folder.name) or folder.name
            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append(
                    BenchmarkSample(
                        image_path=img_path,
                        image_bytes=img_path.read_bytes(),
                        folder_name=folder.name,
                        species_id=species_id,
                        english_name=self._id_to_name.get(species_id, folder.name),
                        in_distribution=(species_id in IN_DISTRIBUTION_SPECIES),
                        key_xml_support=key_support.get(species_id, "unknown"),
                    )
                )

    def __iter__(self) -> Iterator[BenchmarkSample]:
        yield from self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def in_distribution(self) -> List[BenchmarkSample]:
        """Return only the samples whose species the CNN was trained on."""
        return [s for s in self.samples if s.in_distribution]

    def out_of_distribution(self) -> List[BenchmarkSample]:
        """Return only the OOD samples (species absent from CNN training)."""
        return [s for s in self.samples if not s.in_distribution]

    def by_species(self, species_id: str) -> List[BenchmarkSample]:
        """Return all samples belonging to a given species."""
        return [s for s in self.samples if s.species_id == species_id]
