"""Central configuration for the benchmark suite.

All file paths and name-mapping tables live here so they can be
imported by any benchmark module without hard-coding paths.
"""

from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "evaluation_images"
SPECIES_CSV = PROJECT_ROOT / "data" / "raw" / "species.csv"
KEY_XML = PROJECT_ROOT / "data" / "raw" / "key.xml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks"
ORACLE_JSON = PROJECT_ROOT / "benchmarks" / "oracle_answers.json"

# -----------------------------------------------------------------------------
# Folder name → species_id mapping
# -----------------------------------------------------------------------------
# Evaluation images are organised in sub-folders. Some folders use the
# raw species_id ("AM.MU"), others use English descriptive names
# ("fomitopsis_betulina").
FOLDER_TO_SPECIES_ID: Dict[str, str] = {
    "AM.MU": "AM.MU",
    "AM.VI": "AM.VI",
    "BO.ED": "BO.ED",
    "BO.BA": "BO.BA",
    "CA.CI": "CA.CI",
    "CR.CO": "CR.CO",
    "HY.PS": "HY.PS",
    "coprinus_comatus": "CO.CO",
    "fomitopsis_betulina": "FO.BE",
    "lycoperdon_utriforme": "LY.PE",
    "ramaria_botrytis": "RA.BO",
    "sparassis_crispa": "SP.CR",
}

# -----------------------------------------------------------------------------
# CNN output name → species_id mapping
# -----------------------------------------------------------------------------
# The CNN was trained on 7 classes and emits English display names.
# This table bridges those names to the canonical species_id used
# everywhere else in the project.
CNN_NAME_TO_SPECIES_ID: Dict[str, str] = {
    "Fly Agaric": "AM.MU",
    "Chanterelle": "CA.CI",
    "False Chanterelle": "HY.PS",
    "Porcini": "BO.ED",
    "Other Boletus": "BO.BA",
    "Amanita virosa": "AM.VI",
    "Black Trumpet": "CR.CO",
}

# Reverse lookup — needed when building inputs for the FinalAggregator.
SPECIES_ID_TO_CNN_NAME: Dict[str, str] = {
    v: k for k, v in CNN_NAME_TO_SPECIES_ID.items()
}

# -----------------------------------------------------------------------------
# In-distribution species
# -----------------------------------------------------------------------------
# These 7 species are the ones the CNN was trained on. Everything else
# in the evaluation set is out-of-distribution (OOD).
IN_DISTRIBUTION_SPECIES: List[str] = [
    "AM.MU", "AM.VI", "BO.ED", "BO.BA", "CA.CI", "CR.CO", "HY.PS"
]
