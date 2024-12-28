import Bio.SeqUtils
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from BCBio import GFF
import pandas as pd
import time
import codonbias
from numba import jit
from concurrent.futures import ProcessPoolExecutor
from util import calculate_enc, calculate_tai

fasta_path = 'data/GCF_000146045.2/GCF_000146045.2_R64_genomic.fna'
metadata_path = 'data/GCF_000146045.2/genomic.gff'


def cond_print(text, should_print=False):
    if should_print:
        print(text)


def parallel_row_calculate(key_value):
    locus_tag, coding_sequences = key_value

    coding_sequence_total = "".join([str(seq) for seq in coding_sequences])

    results = []
    results.append(locus_tag)
    results.append(calculate_enc(coding_sequence_total))

    results.append(gc_fraction(coding_sequence_total))
    results.append(calculate_tai(coding_sequence_total))

    results.append(len(coding_sequence_total))

    return results


if __name__ == '__main__':
    locus_to_data = dict()

    start = time.time()
    with open(fasta_path, 'r') as fasta_handle, open(metadata_path, 'r') as gff_handle:
        for record in GFF.parse(gff_handle, base_dict=SeqIO.to_dict(SeqIO.parse(fasta_handle, "fasta"))):
            # if "NC_001133.9" != record.id:
            #     continue
            if "NC_001224.1" == record.id:
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
    print(f"Took: {end - start}")

    print(len(locus_to_data))

    feature_columns = ['ENC']

    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            parallel_row_calculate,
            locus_to_data.items()))

    features_df = pd.DataFrame(results, columns=['locus_tag', 'ENC', 'gc_content', 'tAI', 'size'])

    native_expression = pd.read_csv('native_expression.tsv', sep='\t')
    features_df['native_expression'] = features_df['locus_tag'].map(
        native_expression.set_index('locus_tag')['abundance'])

    features_df.to_csv('features.csv')

    print("Not in locus")
    expression_df = pd.read_csv('expression.csv')
    for index, row in expression_df.iterrows():
        if row['ORF'] not in features_df['locus_tag'].values:
            print(row['ORF'])

    print("Not in ORF ")
    for index, row in features_df.iterrows():
        if row['locus_tag'] not in expression_df['ORF'].values:
            print(row['locus_tag'])
