import pandas as pd
import numpy as np
from Bio.SeqUtils import gc_fraction
from sklearn.linear_model import LinearRegression, QuantileRegressor, Lasso
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from util import calculate_enc


# targets
class ProteinMetric:
    def __init__(self, seq):
        self.enc = calculate_enc(seq)
        self.gc_content = gc_fraction(seq)


with open('targets.fa', 'r') as f:
    lines = f.readlines()
    gfp_seq = lines[1][:-1]  # drop \n
    rfp_seq = lines[3][:-1]  # drop \n
    gfp_metric = ProteinMetric(gfp_seq)
    rfp_metric = ProteinMetric(rfp_seq)
    print(f"GFP GC CONTENT: {gfp_metric.gc_content}")
    print(f"GFP ENC: {gfp_metric.enc}")

expression_df = pd.read_csv('expression.csv')
features_df = pd.read_csv('features.csv')
features_df.reset_index(drop=True)

merged_df = expression_df.merge(features_df, left_on='ORF', right_on='locus_tag', how='inner')
print(merged_df)

print(np.all(merged_df['ORF'] == merged_df['locus_tag']))
merged_df = merged_df.drop(columns=['locus_tag'])
merged_df = merged_df.drop(columns=['Unnamed: 0'])

variable_ranges = {'ENC': (20, 61), 'gc_content': (0, 1),
                   'gc_deviation': (0, 1), 'native_expression': (0, 10), 'tAI': (0, 1), 'size': (0, 10)}


def one_feature_linear_model(merged_df, feature, fp_type='gfp_intensity'):
    print(f"Fitting feature: {feature}")

    if feature not in ['ENC', 'gc_content', 'gc_deviation', 'native_expression', 'tAI', 'size']:
        raise ValueError(f"Unknown feature: {feature}")
    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    combined_df = merged_df.dropna(subset=[fp_type]).copy()
    if feature == 'native_expression':
        before = len(combined_df)
        combined_df = combined_df.dropna(subset=[feature])
        combined_df['native_expression'] = np.log(combined_df['native_expression'])

        combined_df[fp_type] = np.log(combined_df[fp_type])
        combined_df[fp_type] = combined_df[fp_type].replace(-np.inf, np.nan)
        combined_df = combined_df.dropna(subset=[fp_type])

        after = len(combined_df)
        print(f"Dropped {before - after} entries")
        model = LinearRegression()

    elif feature == 'ENC':
        # model = Lasso(alpha=0.1)
        model = LinearRegression()
    elif feature == 'gc_deviation':
        # combined_df[fp_type] = np.log(combined_df[fp_type])
        # combined_df[fp_type] = combined_df[fp_type].replace(-np.inf, np.nan)
        # combined_df = combined_df.dropna(subset=[fp_type])

        gc_content_mean = np.mean(combined_df['gc_content'])
        print("GC content mean: ", gc_content_mean)

        combined_df['gc_deviation'] = np.abs(gc_content_mean - combined_df['gc_content']) + 0.001
        # combined_df['gc_deviation'] = np.log(combined_df['gc_deviation'])

        model = LinearRegression()
    elif feature == 'tAI':
        combined_df[fp_type] = np.log(combined_df[fp_type])
        combined_df[fp_type] = combined_df[fp_type].replace(-np.inf, np.nan)
        combined_df = combined_df.dropna(subset=[fp_type])

        # combined_df['tAI'] = combined_df['tAI'] ** 2

        model = LinearRegression()
    elif feature == 'size':
        combined_df[fp_type] = np.log(combined_df[fp_type])
        combined_df[fp_type] = combined_df[fp_type].replace(-np.inf, np.nan)
        combined_df = combined_df.dropna(subset=[fp_type])

        combined_df['size'] = np.log(combined_df['size'])

        model = LinearRegression()
    else:
        model = LinearRegression()

    print("Combined DF ", combined_df[combined_df[fp_type] < -2.])

    x = combined_df[feature].to_numpy().reshape(-1, 1)
    y = combined_df[fp_type].to_numpy()

    model.fit(x, y)

    start, stop = variable_ranges[feature]
    x_line = np.linspace(start, stop, 100)
    y_line = model.coef_ * x_line + model.intercept_

    plt.plot(x_line, y_line, color='black')
    plt.scatter(x, y, alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel(fp_type)
    plt.show()

    y_predicted = model.predict(x)
    pearson_correlation, _ = pearsonr(y, y_predicted)
    spearman_correlation, _ = spearmanr(y, y_predicted)
    print("Model score: ", model.score(x, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")

    print(f"Intercept: {model.intercept_} Slope: {model.coef_}")
    print(f"Average intensity {fp_type}: ", np.mean(combined_df[fp_type]))

    # TODO: remove
    if feature == 'gc_deviation':
        print(f"Spearman correlation gc_deviation: {spearmanr(y, combined_df['gc_deviation'])[0]}")


def one_feature_polynomial_model(merged_df, feature, fp_type='gfp_intensity'):
    print(f"Fitting polynomially feature: {feature}")

    if feature not in ['ENC', 'gc_content', 'gc_deviation', 'size']:
        raise ValueError(f"Unknown feature: {feature}")
    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    combined_df = merged_df.dropna(subset=[fp_type]).copy()
    #

    if feature == 'size':
        combined_df.loc[:, fp_type] = np.log(combined_df[fp_type]).copy()
        combined_df.loc[:, fp_type] = combined_df[fp_type].replace(-np.inf, np.nan).copy()
        combined_df = combined_df.dropna(subset=[fp_type])

        combined_df['size'] = np.log(combined_df['size'])
    elif feature == 'gc_content':
        # combined_df.loc[:, fp_type] = np.log(combined_df[fp_type]).copy()
        # combined_df.loc[:, fp_type] = combined_df[fp_type].replace(-np.inf, np.nan).copy()
        # combined_df = combined_df.dropna(subset=[fp_type])

        pass

    model = LinearRegression()

    x = combined_df[feature].to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(x)

    y = combined_df[fp_type].to_numpy()
    model.fit(X_poly, y)

    start, stop = variable_ranges[feature]
    x_line = np.linspace(start, stop, 100)
    y_line = model.coef_[1] * x_line ** 2 + model.coef_[0] * x_line + model.intercept_

    plt.plot(x_line, y_line, color='black')
    plt.scatter(x, y)
    plt.xlabel(feature)
    plt.ylabel(fp_type)
    plt.show()

    y_predicted = model.predict(X_poly)
    pearson_correlation, _ = pearsonr(y, y_predicted)
    spearman_correlation, _ = spearmanr(y, y_predicted)
    print("Model score: ", model.score(X_poly, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")

    print(f"Intercept: {model.intercept_} coeffs: {model.coef_}")
    print(f"Average intensity {fp_type}: ", np.mean(combined_df[fp_type]))


def good_model(merged_df, fp_type='gfp_intensity'):
    features = ['ENC', 'gc_content', 'native_expression', 'tAI', 'size']

    print(f"Fitting polynomially features: {features}")

    for feature in features:
        if feature not in ['ENC', 'gc_content', 'gc_deviation', 'native_expression', 'tAI', 'size']:
            raise ValueError(f"Unknown feature: {feature}")

    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    combined_df = merged_df.dropna(subset=[fp_type]).copy()

    # TODO: what do we do when there are no expression levels?
    combined_df = combined_df.dropna(subset=['native_expression'])
    combined_df['native_expression'] = np.log(combined_df['native_expression'])

    # Log the fluorescent protein intensity, because expression is exponential
    # min_intensity = combined_df[combined_df[fp_type] > 0.][fp_type].min()
    # combined_df[fp_type] = np.log(combined_df[fp_type])
    # combined_df[fp_type] = combined_df[fp_type].replace(-np.inf, np.log(min_intensity))

    # combined_df = combined_df.dropna(subset=[fp_type])

    model = LinearRegression()

    gc_content_x = combined_df['gc_content'].to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    gc_content_poly = poly.fit_transform(gc_content_x)

    combined_df['log_size'] = np.log(combined_df['size'])

    size_x = combined_df['log_size'].to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    size_poly = poly.fit_transform(size_x)

    x_columns = []

    if 'ENC' in features:
        x_columns.append(combined_df['ENC'].to_numpy().reshape(-1, 1))
    if 'native_expression' in features:
        x_columns.append(combined_df['native_expression'].to_numpy().reshape(-1, 1))
    if 'tAI' in features:
        x_columns.append(combined_df['tAI'].to_numpy().reshape(-1, 1))
    if 'gc_content' in features:
        x_columns.append(gc_content_poly)
    if 'size' in features:
        x_columns.append(size_poly)

    X_final = np.column_stack(x_columns)

    y = combined_df[fp_type].to_numpy()
    model.fit(X_final, y)

    y_predicted = model.predict(X_final)

    pearson_correlation, _ = pearsonr(y, y_predicted)
    spearman_correlation, _ = spearmanr(y, y_predicted)
    print("Model score: ", model.score(X_final, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")

    print(f"Intercept: {model.intercept_} coeffs: {model.coef_}")
    print(f"Average intensity {fp_type}: ", np.mean(combined_df[fp_type]))

#
# good_model(merged_df, fp_type='gfp_intensity')
# good_model(merged_df, fp_type='rfp_intensity')


# one_feature_linear_model(merged_df, feature='size', fp_type='rfp_intensity')
# one_feature_polynomial_model(merged_df, feature='size', fp_type='rfp_intensity')
# one_feature_linear_model(merged_df, feature='ENC', fp_type='rfp_intensity')
one_feature_linear_model(merged_df, feature='tAI', fp_type='gfp_intensity')
# one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='rfp_intensity')
# one_feature_linear_model(merged_df, feature='gc_deviation', fp_type='gfp_intensity')

# one_feature_linear_model(merged_df, feature='native_expression', fp_type='gfp_intensity')
# one_feature_linear_model(merged_df, feature='native_expression', fp_type='rfp_intensity')

# one_feature_linear_model(merged_df, feature='ENC', fp_type='gfp_intensity')
# one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='gfp_intensity')
# one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='gfp_intensity')

# one_feature_linear_model(merged_df, feature='ENC', fp_type='rfp_intensity')
# one_feature_linear_model(merged_df, feature='gc_content', fp_type='rfp_intensity')
