import pandas as pd
import numpy as np
from Bio.SeqUtils import gc_fraction
from sklearn.linear_model import LinearRegression, QuantileRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

import create_features
from consts import TARGETS_FA
from create_features import get_locus_to_data_dict
from util import calculate_enc


# targets
class ProteinMetric:
    def __init__(self, seq):
        self.enc = calculate_enc(seq)
        self.gc_content = gc_fraction(seq)


VARIABLE_RANGES = {'ENC': (0, 1), 'gc_content': (0, 1),
                   'gc_deviation': (0, 1), 'native_expression': (0, 10), 'tAI': (0, 1), 'size': (0, 10),
                   'cAI50': (0., 1.), 'cAI200': (0., 1.), 'charge': (-10, 10), 'charge10': (-1, 1),
                   'mrna_estimated': (0, 10), 'cAI200_first50': (0, 1), 'cAI200_last50': (0, 1),
                   'gfp_intensity': (-3, 10), 'rfp_intensity': (-3, 10),
                   'gfp_charge': (0, 1), 'gfp_tai': (0, 1), 'gfp_cai': (0, 1),
                   'gfp_fraction': (0, 1),
                   'cARSscoresCodons': (0, 1),
                   'fold_energy': (-100, 0),
                   'fold_duplex_gfp': (-100, 0)
                   }


def log_with_zero(df: pd.DataFrame, column_name: str, pos=True, min_value=None) -> pd.DataFrame:
    # Calculate the minimum positive value divided by 2
    df = df.copy()

    if not pos:
        df[column_name] = -df[column_name]

    if min_value is None:
        min_intensity = df.loc[df[column_name] > 0, column_name].min()
    else:
        min_intensity = min_value

    # Replace zero values with min_intensity
    df.loc[df[column_name] == 0, column_name] = min_intensity

    # Apply the log transformation
    df[column_name] = np.log(df[column_name])

    return df


def one_feature_linear_model(merged_df, feature, fp_type='gfp_intensity'):
    print(f"Linearly fitting  feature: {feature}")

    if feature not in create_features.ALL_FEATURES and feature not in ['second_fp', 'native_expression',
                                                                       'cARSscoresCodons']:
        raise ValueError(f"Unknown feature: {feature}")
    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    combined_df = merged_df.copy().dropna(subset=[fp_type])
    combined_df = log_with_zero(combined_df, fp_type)

    if feature == 'native_expression':
        before = len(combined_df)
        combined_df = combined_df.dropna(subset=[feature])
        combined_df['native_expression'] = np.log(combined_df['native_expression'])
        after = len(combined_df)
        print(f"native_expression used: Dropped {before - after} entries")

        model = LinearRegression()
    elif feature == 'second_fp':
        feature = 'rfp_intensity'
        combined_df = combined_df.dropna(subset=['rfp_intensity', 'gfp_intensity'])
        combined_df = log_with_zero(combined_df, 'rfp_intensity')

        model = LinearRegression()
    elif feature == 'ENC':
        combined_df['ENC'] = (combined_df['ENC'] - 20) / 44
        combined_df['ENC'] = np.log(combined_df['ENC'])
        model = LinearRegression()
    elif feature == 'gc_deviation':
        gc_content_mean = np.mean(combined_df['gc_content'])
        print("GC content mean: ", gc_content_mean)

        combined_df['gc_deviation'] = np.abs(gc_content_mean - combined_df['gc_content']) + 0.001
        # combined_df['gc_deviation'] = np.log(combined_df['gc_deviation'])

        model = LinearRegression()
    elif feature == 'tAI':
        combined_df['tAI'] = np.log(combined_df['tAI'])
        model = LinearRegression()
    elif feature == 'gfp_tai':
        combined_df['gfp_tai'] = np.log(combined_df['gfp_tai'])
        model = LinearRegression()
    elif feature == 'size':
        combined_df['size'] = np.log(combined_df['size'])

        model = LinearRegression()
    elif feature.startswith('cAI'):
        combined_df = log_with_zero(combined_df, feature)
        model = LinearRegression()
    elif feature.startswith('charge'):
        # combined_df[feature] = -combined_df[feature]
        # combined_df = log_with_zero(combined_df, feature)
        model = LinearRegression()
    elif feature == 'fold_energy':
        df_copy = log_with_zero(combined_df, 'fold_energy', pos=False)
        combined_df['fold_energy'] = -df_copy['fold_energy']

        model = LinearRegression()
    elif feature == 'fold_duplex_gfp':
        df_copy = log_with_zero(combined_df, 'fold_duplex_gfp', pos=False)
        combined_df['fold_duplex_gfp'] = -df_copy['fold_duplex_gfp']

        model = LinearRegression()
    else:
        model = LinearRegression()

    # print("Combined DF ", combined_df[combined_df[fp_type] < -2.])

    x = combined_df[feature].to_numpy().reshape(-1, 1)
    y = combined_df[fp_type].to_numpy()

    model.fit(x, y)

    start, stop = VARIABLE_RANGES[feature]
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
    combined_df = log_with_zero(combined_df, fp_type)

    if feature == 'size':
        combined_df['size'] = np.log(combined_df['size'])
    elif feature == 'gc_content':
        combined_df['gc_content'] = np.log(combined_df['gc_content'])
    else:
        raise ValueError(f"Ignoring feature: {feature}")

    model = LinearRegression()

    x = combined_df[feature].to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(x)

    y = combined_df[fp_type].to_numpy()
    model.fit(X_poly, y)

    start, stop = VARIABLE_RANGES[feature]
    x_line = np.linspace(start, stop, 100)
    y_line = model.coef_[1] * x_line ** 2 + model.coef_[0] * x_line + model.intercept_

    plt.plot(x_line, y_line, color='black')
    plt.scatter(x, y, alpha=0.1)
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
    fp_name = 'gfp' if fp_type == 'gfp_intensity' else 'rfp'

    if fp_name == 'gfp':
        features = ['second_fp', 'ENC', 'gc_content', 'native_expression', 'tAI', 'size', 'cAI50', 'cAI200',
                    'cAI200_first50', 'cAI200_last50', fp_name + '_tai', 'fold_duplex_' + fp_name]
    else:
        features = ['second_fp', 'ENC', 'gc_content', 'native_expression', 'tAI', 'size', 'cAI50', 'cAI200',
                    'cAI200_first50', 'cAI200_last50']

    print(f"Fitting good model for: {features}")

    for feature in features:
        if feature not in create_features.ALL_FEATURES and feature not in ['second_fp', 'native_expression']:
            raise ValueError(f"Unknown feature: {feature}")

    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    print(f"Total: {merged_df[fp_type].notna().sum()} points")
    combined_df = merged_df.dropna(subset=[fp_type]).copy()

    # TODO: what do we do when there are no expression levels?
    combined_df = combined_df.dropna(subset=['native_expression'])
    combined_df['native_expression'] = np.log(combined_df['native_expression'])

    # Log the fluorescent protein intensity, because expression is exponential
    log_with_zero(combined_df, fp_type)

    # combined_df = combined_df.dropna(subset=[fp_type])

    x_columns = []

    # if 'ENC' in features:
    #     combined_df['ENC'] = (combined_df['ENC'] - 20) / 44
    #     combined_df['ENC'] = np.log(combined_df['ENC'])
    #     x_columns.append(combined_df['ENC'].to_numpy().reshape(-1, 1))
    if 'native_expression' in features:
        x_columns.append(combined_df['native_expression'].to_numpy().reshape(-1, 1))
    if 'tAI' in features:
        combined_df['tAI'] = np.log(combined_df['tAI'])
        x_columns.append(combined_df['tAI'].to_numpy().reshape(-1, 1))
    if 'gc_content' in features:
        gc_content_x = combined_df['gc_content'].to_numpy().reshape(-1, 1)
        gc_content_x = np.log(gc_content_x)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        gc_content_poly = poly.fit_transform(gc_content_x)

        x_columns.append(gc_content_poly)

    # if fp_type + 'tai' in features:
    #     x_columns.append(combined_df[fp_type + '_tai'].to_numpy().reshape(-1, 1))

    # if fp_type + '_charge' in features:
    #     x_columns.append(combined_df[fp_type + '_charge'].to_numpy().reshape(-1, 1))
    # if fp_type + '_fraction' in features:
    #     x_columns.append(combined_df[fp_type + '_fraction'].to_numpy().reshape(-1, 1))

    if 'size' in features:
        max_size = 14733
        combined_df['log_size'] = np.log(combined_df['size'] / max_size)

        size_x = combined_df['log_size'].to_numpy().reshape(-1, 1)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        size_poly = poly.fit_transform(size_x)

        x_columns.append(size_poly)
    # if 'cAI200' in features:
    #     combined_df['cAI200'] = np.log(combined_df['cAI200'])
    #     x_columns.append(combined_df['cAI200'].to_numpy().reshape(-1, 1))
    # if 'cAI200_first50' in features:
    #     combined_df['cAI200_first50'] = np.log(combined_df['cAI200_first50'])
    #     x_columns.append(combined_df['cAI200_first50'].to_numpy().reshape(-1, 1))
    # if 'cAI200_last50' in features:
    #     combined_df['cAI200_last50'] = np.log(combined_df['cAI200_last50'])
    #     x_columns.append(combined_df['cAI200_last50'].to_numpy().reshape(-1, 1))

    if 'fold_duplex_' + fp_name in features:
        df_copy = log_with_zero(combined_df, 'fold_duplex_gfp', pos=False)
        combined_df['fold_duplex_gfp'] = -df_copy['fold_duplex_gfp']
        x_columns.append(combined_df['fold_duplex_gfp'].to_numpy().reshape(-1, 1))

    X_final = np.column_stack(x_columns)

    y = combined_df[fp_type].to_numpy()

    model = LinearRegression()
    model.fit(X_final, y)
    print(f"Fit: {len(y)} points")

    y_predicted = model.predict(X_final)

    pearson_correlation, _ = pearsonr(y, y_predicted)
    spearman_correlation, _ = spearmanr(y, y_predicted)
    print("Model score: ", model.score(X_final, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")

    print(f"Intercept: {model.intercept_} coeffs: {model.coef_}")
    print(f"Average intensity {fp_type}: ", np.mean(combined_df[fp_type]))


def random_forest(df: pd.DataFrame, fp_type='gfp_intensity'):
    mirror_fp = 'rfp_intensity' if fp_type == 'gfp_intensity' else 'gfp_intensity'

    df = df.copy().dropna(subset=[fp_type])

    df.dropna()

    rf = RandomForestRegressor(n_estimators=100)
    df = df.drop(['ORF', mirror_fp], axis=1)

    x = df.drop(fp_type, axis=1)
    df = log_with_zero(df.copy(), fp_type)
    y = df[fp_type]
    rf.fit(x, y)

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': x.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(feature_importances)


if __name__ == '__main__':
    gfp_seq, rfp_seq = create_features.load_fp_genes()
    gfp_metric = ProteinMetric(gfp_seq)
    rfp_metric = ProteinMetric(rfp_seq)
    print(f"GFP GC CONTENT: {gfp_metric.gc_content}")
    print(f"RFP GC CONTENT: {rfp_metric.gc_content}")
    print(f"GFP ENC: {gfp_metric.enc}")
    print(f"RFP ENC: {rfp_metric.enc}")

    expression_df = pd.read_csv('data/expression.csv')
    features_df = pd.read_csv('features/features.csv')
    cARSscoresCodons_df = pd.read_csv('features/cARSscoresCodons.csv')
    features_df.reset_index(drop=True)

    merged_df = expression_df.merge(features_df, left_on='ORF', right_on='locus_tag', how='inner')
    merged_df['cARSscoresCodons'] = cARSscoresCodons_df['cARSscoresCodons']
    print(merged_df)

    print(np.all(merged_df['ORF'] == merged_df['locus_tag']))
    merged_df = merged_df.drop(columns=['locus_tag'])
    merged_df = merged_df.drop(columns=['Unnamed: 0'])

    print("Total gene amount: ", len(merged_df))
    print("Total GFP zero amount: ", (merged_df['gfp_intensity'] == 0).sum())
    print("Total RFP zero amount: ", (merged_df['rfp_intensity'] == 0).sum())

    joint_df = merged_df[merged_df[['gfp_intensity', 'rfp_intensity']].notna().all(axis=1)]
    gfp_not_na = merged_df[merged_df[['gfp_intensity']].notna().all(axis=1)]
    rfp_not_na = merged_df[merged_df[['rfp_intensity']].notna().all(axis=1)]
    missing = merged_df[merged_df[['rfp_intensity', 'gfp_intensity']].isna().all(axis=1)]
    #
    # merged_df = log_with_zero(merged_df, 'gfp_intensity')
    # merged_df = log_with_zero(merged_df, 'native_expression')
    # merged_df['gfp_intensity'] = np.exp(merged_df['gfp_intensity'] - 0.22 * merged_df['native_expression']  + 0.91)
    # merged_df['native_expression'] = np.exp(merged_df['native_expression'])

    print("Joint FP amount: ", len(joint_df))
    print("RFP not NA: ", len(rfp_not_na))
    print("GFP not NA: ", len(gfp_not_na))
    print("Missing amount: ", len(missing))

    locus_to_data = get_locus_to_data_dict()
    create_features.init_cai_calculators(locus_to_data)
    with open(TARGETS_FA, 'r') as f:
        lines = f.readlines()
    GFP_genome = lines[1].rstrip('\n').upper()
    RFP_genome = lines[3].rstrip('\n').upper()

    print("GFP CAI: ", create_features.cai_calculator200.calculate(GFP_genome))
    print("RFP CAI: ", create_features.cai_calculator200.calculate(RFP_genome))

    merged_df['diff'] = np.abs(log_with_zero(merged_df.copy(), 'gfp_intensity')['gfp_intensity'] -
                               log_with_zero(merged_df.copy(), 'rfp_intensity')['rfp_intensity'])
    # merged_df = merged_df.sort_values(by='diff', ascending=False).tail(100)
    # merged_df = merged_df.sort_values(by='diff', ascending=False).dropna(subset=['diff']).head(1000)
    # merged_df = merged_df[(merged_df['gfp_intensity'] > merged_df['rfp_intensity'])]
    # merged_df = merged_df[(merged_df['gfp_intensity'] > 1) & (merged_df['rfp_intensity'] > 1)]
    # merged_df = merged_df[(merged_df['gfp_intensity'] < 0.5)].copy()
    # merged_df.loc[merged_df['gfp_intensity'] < 0.5, 'gfp_size'] = 0.5
    # merged_df.loc[merged_df['gfp_intensity'] > 0.5, 'gfp_size'] = 1
    # print("Merged df: ", merged_df['diff'])
    # print("Merged df: ", merged_df['gfp_intensity'])

    plt.scatter(merged_df['gfp_intensity'], merged_df['rfp_intensity'], alpha=0.5)
    plt.xlabel('gfp_intensity')
    plt.ylabel('rfp_intensity')
    plt.plot()
    plt.show()

    plt.scatter(log_with_zero(merged_df, 'gfp_intensity')['gfp_intensity'],
                log_with_zero(merged_df, 'rfp_intensity')['rfp_intensity'], alpha=0.5)
    plt.xlabel('gfp_intensity')
    plt.ylabel('rfp_intensity')
    plt.plot()
    plt.show()

    random_forest(merged_df, 'gfp_intensity')
    random_forest(merged_df, 'rfp_intensity')

    good_model(merged_df, fp_type='gfp_intensity')
    good_model(merged_df, fp_type='rfp_intensity')

    # Standard
    fp_type = 'gfp_intensity'
    # one_feature_linear_model(merged_df, feature='ENC', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='tAI', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='cAI200', fp_type=fp_type)
    # one_feature_polynomial_model(merged_df, feature='gc_content', fp_type=fp_type)
    # one_feature_polynomial_model(merged_df, feature='size', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='charge', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='charge', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='gfp_tai', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='tAI', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='gfp_cai', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='cARSscoresCodons', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='fold_energy', fp_type=fp_type)
    one_feature_linear_model(merged_df, feature='fold_duplex_gfp', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='gc_content', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='gc_content', fp_type='rfp_intensity')

    # one_feature_linear_model(merged_df, feature='gfp_charge', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='gfp_fraction', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='gfp_cai', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='gfp_tai', fp_type='gfp_intensity')

    # one_feature_linear_model(merged_df, feature='second_fp', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='native_expression', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI200_last50', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI200_first50', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI200', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='mrna_estimated', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI50', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI200', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='cAI200', fp_type='rfp_intensity')
    # one_feature_polynomial_model(merged_df, feature='size', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='ENC', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='ENC', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='tAI', fp_type='gfp_intensity')
    # one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='gfp_intensity')
    # one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='gc_deviation', fp_type='gfp_intensity')

    # one_feature_linear_model(merged_df, feature='native_expression', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='native_expression', fp_type='rfp_intensity')

    # one_feature_linear_model(merged_df, feature='ENC', fp_type='gfp_intensity')
    # one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='gfp_intensity')
    # one_feature_polynomial_model(merged_df, feature='gc_content', fp_type='gfp_intensity')

    # one_feature_linear_model(merged_df, feature='gc_content', fp_type='rfp_intensity')
