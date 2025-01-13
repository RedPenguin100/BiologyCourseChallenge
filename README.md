To install dependencies:

First install ViennaRNA:

ViennaRNA installation
https://www.tbi.univie.ac.at/RNA/#download

Then use `conda_environment.yml` to install the conda environment.

Installation will be like this: 

```shell
conda env create -f conda_environment.yml
```


<h1>Data sources</h1>

Native expression for yeast source
https://pax-db.org/downloads/5.0/datasets/4932/4932-WHOLE_ORGANISM-integrated.txt

Chimera Scores were achieved by running Matlab in the `ChimeraCode` folder.

We ran

```matlab
[suffixArray] = Chimera(0, 2, 0, reference_sequences, reference_sequences, 1)
[suffixArray, all_seqs, cARSscores, engineered_targets] = Chimera(0, 2, 1, reference_sequences, reference_sequences, 1, suffixArray, allSeqs)
```
Where reference_sequences are the rows from genes.txt (all yeast genes ORFs, no introns) \
We stored the results in cARSscoresCodons.csv


