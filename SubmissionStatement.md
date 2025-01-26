# Cell engineering challenge

In this project, I was assigned a task to create a model, to predict in the best way (measured by spearman correlation) the expression of both GFP and RFP proteins under fusion with a target gene from Yeast. 

The prediction file can be found under data/expression_prediction.csv

## Features + Data
I've __considered__ using many features, but ended up using only a few. 

The features I've chosen are 

### Explicitly learned in class 
* ENC - Effective Number of Codons
* tAI
* RSCU
* GC content
* Gene size (length)
* cofold 

### Some features I've added
* The second FP intensity level, where available.
* Duplex fold - target coding sequence with the reverse end(150n) of the FP gene
* Duplex fold - target coding sequence beginning(150n) with the reversed end(150n) of the FP gene
* Native expression - I've downloaded data from </br> https://pax-db.org/downloads/5.0/datasets/4932/4932-WHOLE_ORGANISM-integrated.txt
* RSCU with asymmetrically weighted distance - the idea is to get an RSCU weight for every codon, both for target and for FP gene. Then we get 61-length vectors. We subtract them, __and multiply__ with the FP gene. The idea is that we should weigh the distances of the frequent codons in FP more than the ones that are less frequently used. 


I have not used any further external data.

I've *considered* (and calculated into features.csv) many feature, including 

1. total fold of target gene
2. ChimeraARS codon-wise (using Tamir's Matlab)
3. Harmonic mean of tAI
4. Fold energy in first nucleotides
5. cAI, cAI for 200/50 most highly expressed genes
6. Electric charge

I had very poor results and decided it is best not to add these features to my model.

## The model

I did a relatively simple regression model. There are 3 nuances
1. The model is "layered" - we create 3 models
   1. The least generic, with all features
   2. A little more generic, without the second FP as a feature
   3. The most generic, without expression and FP.

This model structure allows me to get benefits from the strong correlation that RFP exhibits with GFP, and native protein abundance, without the downside of "guessing" numbers where the feature is missing.

2. The second nuance, the model has some element of polynomial regression. Both GC content, and Size, are not linear features. I chose polynomial to express the non-linearity. The fitting process was still standard, as I added a column of the squared feature to the fitting.
3. The final nuance, is that I did not use the OLS but rather a L1 regression. Idea being that we are less sensitive to outliers this way, and the data is noisy in any case.

## Ideas 

I spent a lot of time contemplating how I may introduce non-linearity in a wise way, I have not done that successfully (other than the polynomials).
My main idea was the following
1. Split the data into highly and lowly expressed genes
2. Fit them using logistic regression to decide if a gene is a highly or lowly expressed one
3. Only then, do 2 separate linear regressions on the data.

This subtle way of non-linearity improved the model by a miniscule amount (~0.01 spearman in for FPs), but the hassle and inconsistency with new features seemed to me not worth it. Note-worth, some features exhibit much stronger correlation with low/high expressed genes, which was interesting to me. Notably the duplex fold was relatively correlative (0.27) in lowly expressed RFP but not correlative in highly expressed.

## Apology
I apologize for the code, usually I'm better. I spent most of the time in the researching hat and not developing hat, I should have done both simulatenously. 




