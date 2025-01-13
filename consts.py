import os
from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(__file__))

TEST_PATH = PROJECT_PATH

DATA_PATH = PROJECT_PATH / 'data'
TARGETS_FA = DATA_PATH / 'targets.fa'

YEAST_DATA = PROJECT_PATH / 'yeast_data'
YEAST_FASTA_PATH = YEAST_DATA / 'GCF_000146045.2' / 'GCF_000146045.2_R64_genomic.fna'
YEAST_METADATA_PATH = YEAST_DATA / 'GCF_000146045.2' / 'genomic.gff'
