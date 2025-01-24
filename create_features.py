from copy import deepcopy

import RNA
import scipy
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, CodonAdaptationIndex
from BCBio import GFF
import pandas as pd
import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from concurrent.futures import ProcessPoolExecutor

from consts import TARGETS_FA, YEAST_FASTA_PATH, YEAST_METADATA_PATH
from util import calculate_enc, calculate_tai, calculate_cai

def cond_print(text, should_print=False):
    if should_print:
        print(text)


cai_calculator50 = None
cai_calculator200 = None


def init_cai_calculators(locus_to_data):
    global cai_calculator50
    global cai_calculator200

    top1000 = pd.read_csv('features/native_expression_top1000.tsv', sep='\t')

    top1000_seq = []
    for locus_tag in top1000['locus_tag']:
        # Gene is in native_expression but not in expression.csv
        if locus_tag not in locus_to_data:
            continue
        data = "".join([str(seq) for seq in locus_to_data[locus_tag]])

        top1000_seq.append(Seq(data))

    cai_calculator50 = CodonAdaptationIndex(top1000_seq[:50])
    cai_calculator200 = CodonAdaptationIndex(top1000_seq[:200])


gfp_gene = None
rfp_gene = None


def load_fp_genes():
    global gfp_gene
    global rfp_gene

    with open(TARGETS_FA, 'r') as f:
        lines = f.readlines()
        gfp_gene = lines[1][:-1].upper()  # drop \n
        rfp_gene = lines[3][:-1].upper()  # drop \n

    return gfp_gene, rfp_gene


def initialize_all(locus_to_data, genes):
    init_cai_calculators(locus_to_data)
    global gfp_gene, rfp_gene
    gfp_gene, rfp_gene = genes


def calculate_charge(seq: str) -> float:
    # 6.8 - pH charge of yeast cytosol
    analysis = ProteinAnalysis(seq)
    return analysis.charge_at_pH(6.8)


def calculate_fold_energy(seq: str):
    structure, mfe = RNA.fold(seq)
    return mfe

def calculate_fold_duplex(seq: str, fp: str):
    fc = RNA.duplexfold(seq, fp)
    return fc.energy

def calculate_cofold(seq: str, fp: str):
    co_res = RNA.cofold(seq, fp)
    return co_res[-1]


def parallel_row_calculate(key_value):
    locus_tag, coding_sequences = key_value

    coding_sequence_total = "".join([str(seq) for seq in coding_sequences])

    results = []
    results.append(locus_tag)
    results.append(calculate_enc(coding_sequence_total))

    results.append(gc_fraction(coding_sequence_total))
    results.append(calculate_tai(coding_sequence_total))

    results.append(len(coding_sequence_total))

    global cai_calculator50, cai_calculator200
    results.append(cai_calculator50.calculate(coding_sequence_total))
    results.append(cai_calculator200.calculate(coding_sequence_total))

    results.append(cai_calculator200.calculate(coding_sequence_total[:150]))
    results.append(cai_calculator200.calculate(coding_sequence_total[-150:]))

    results.append(calculate_charge(coding_sequence_total))
    results.append(calculate_charge(coding_sequence_total[:30]))

    global gfp_gene
    global rfp_gene

    results.append(scipy.stats.hmean([calculate_tai(coding_sequence_total), calculate_tai(gfp_gene)]))
    results.append(scipy.stats.hmean([calculate_tai(coding_sequence_total), calculate_tai(rfp_gene)]))

    results.append(calculate_fold_energy(coding_sequence_total[:40]))

    results.append(calculate_fold_duplex(coding_sequence_total, gfp_gene[::-1]))
    results.append(calculate_fold_duplex(coding_sequence_total, rfp_gene[::-1]))

    results.append(calculate_cofold(gfp_gene, coding_sequence_total))
    results.append(calculate_cofold(rfp_gene, coding_sequence_total))

    results.append(calculate_fold_duplex(coding_sequence_total, gfp_gene[::-1][:150]))
    results.append(calculate_fold_duplex(coding_sequence_total, rfp_gene[::-1][:150]))

    results.append(calculate_fold_duplex(coding_sequence_total[:150], gfp_gene[::-1][:150]))
    results.append(calculate_fold_duplex(coding_sequence_total[:150], rfp_gene[::-1][:150]))

    # Very slow, run only when needed
    # results.append(calculate_fold_energy(coding_sequence_total))

    return results


def get_locus_to_data_dict():
    locus_to_data = dict()

    start = time.time()
    with open(YEAST_FASTA_PATH, 'r') as fasta_handle, open(YEAST_METADATA_PATH, 'r') as gff_handle:
        for record in GFF.parse(gff_handle, base_dict=SeqIO.to_dict(SeqIO.parse(fasta_handle, "fasta"))):
            if "NC_001224.1" == record.id:  # Mitochondrial DNA, skip
                continue

            cond_print(f"> {record.id}")

            cond_print(record.features)
            for feature in record.features:
                cond_print(f"> {feature.type}")
                if feature.type == 'gene':
                    cond_print(f" > {feature}")
                    cond_print(f" > {feature.id}")
                    for sub_feature in feature.sub_features:
                        for sub_sub_feature in sub_feature.sub_features:
                            cond_print(f">> {sub_sub_feature}")
                            if sub_sub_feature.type == 'CDS':
                                cond_print(f"  CDS ID: {sub_sub_feature.id}")
                                cond_print(f"  Location: {sub_sub_feature.location}")
                                cond_print(f"  Parent Gene: {sub_sub_feature.qualifiers.get('Parent')}")
                                if len(sub_sub_feature.qualifiers['locus_tag']) != 1:
                                    raise ValueError(f"More than 1 locus tag on CDS ID: {sub_sub_feature.id}")
                                locus_tag = sub_sub_feature.qualifiers['locus_tag'][0]
                                seq = sub_sub_feature.extract(record.seq)

                                locus_to_data.setdefault(locus_tag, []).append(seq.upper())
    end = time.time()
    print(f"Took: {end - start}s")

    return locus_to_data


FEATURES = ['ENC', 'gc_content', 'tAI', 'size', 'cAI50', 'cAI200', 'cAI200_first50', 'cAI200_last50', 'charge',
            'charge10',
            'gfp_tai',
            'rfp_tai', 'fold_energy_begin',
            'fold_duplex_gfp', 'fold_duplex_rfp',
            'fold_cofold_gfp', 'fold_cofold_rfp',
            'cofold_start_gfp', 'cofold_start_rfp', # TODO: rename
            'duplex_ends_gfp', 'duplex_ends_rfp', # TODO: rename

            ]

ALL_FEATURES = set(FEATURES+ ['fold_energy'])

if __name__ == '__main__':

    locus_to_data = get_locus_to_data_dict()

    print("Starting to create features...")

    # locus_to_data = dict(list(locus_to_data.items())[:10])

    # Needed for cAI features
    init_cai_calculators(locus_to_data)
    # Needed for FP dependent features
    load_fp_genes()
    genes = (deepcopy(gfp_gene), deepcopy(rfp_gene))

    start = time.time()
    results = []
    with ProcessPoolExecutor(initializer=initialize_all, initargs=(locus_to_data, genes)) as executor:
        futures = executor.map(parallel_row_calculate, locus_to_data.items())

        # Track progress
        completed = 0
        total_tasks = len(locus_to_data)
        progress_interval = 10

        for result in futures:
            results.append(result)  # Collect result
            completed += 1
            if completed % progress_interval == 0:
                print(f"Completed {completed} of {total_tasks} tasks")
                assert len(result) == len(FEATURES) + 1, "Incorrect alignment"

    end = time.time()
    print(f"Creating all features took: {end - start}s")

    columns = ['locus_tag']
    columns.extend(FEATURES)

    features_df = pd.DataFrame(results,
                               columns=columns)


    # Caching
    cached_features = pd.read_csv('features/features_cache.csv')
    features_df['fold_energy'] = cached_features['fold_energy']


    native_expression = pd.read_csv('features/native_expression.tsv', sep='\t')
    features_df['native_expression'] = features_df['locus_tag'].map(
        native_expression.set_index('locus_tag')['abundance'])


    features_df.to_csv('features/features.csv')

    print("Not in locus")
    expression_df = pd.read_csv('data/expression.csv')
    for index, row in expression_df.iterrows():
        if row['ORF'] not in features_df['locus_tag'].values:
            print(row['ORF'])

    print("Not in ORF ")
    for index, row in features_df.iterrows():
        if row['locus_tag'] not in expression_df['ORF'].values:
            print(row['locus_tag'])
