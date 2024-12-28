import codonbias
import pytest
from Bio.SeqUtils import gc_fraction
from util import nucleotide_wobble

def my_calculate_enc(seq: str):
    if len(seq) % 3 != 0:
        raise ValueError(f"sequence length {len(seq)} is not a multiple of 3")




def test_enc():
    sequence = 'ATGGGG'

    enc = codonbias.scores.EffectiveNumberOfCodons().get_score(sequence)

    print(enc)


def test_gc_fraction():
    sequence = "ATGATG"
    assert gc_fraction(sequence) == 1 / 3



def test_wobble():
    assert pytest.approx(nucleotide_wobble('A', 'A')) == 0.9999
