import pandas as pd
import numpy as np
from Bio.SeqUtils import gc_fraction
from pandas import read_csv
from sklearn.linear_model import LogisticRegression, LinearRegression, QuantileRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

import create_features
from consts import TARGETS_FA
from create_features import get_locus_to_data_dict
from util import calculate_enc, determination


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
                   'fold_energy_begin': (-10, 0),
                   'fold_duplex_gfp': (-10, 0),
                   'fold_duplex_rfp': (-10, 0),
                   'fold_cofold_gfp': (-10, 0),
                   'fold_cofold_rfp': (-10, 0),
                   'cofold_start_gfp': (3.5, 4.5),
                   'cofold_start_rfp': (3.5, 4.5),
                   'duplex_ends_gfp': (3.5, 4.5),
                   'duplex_ends_rfp': (3.5, 4.5),
                   'rscu': (0, 1),
                   'rscu_distance_gfp': (0, 10),
                   'rscu_distance_rfp': (0, 10),
                   'rscu_distance_asym_gfp': (0, 10),
                   'rscu_distance_asym_rfp': (0, 10),
                   'rscu_distance_weighted_asym_gfp': (0, 10),
                   'rscu_distance_weighted_asym_rfp': (0, 10)
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
    fp_name = 'gfp' if fp_type == 'gfp_intensity' else 'rfp'

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
        combined_df['ENC'] = (combined_df['ENC'] - 20) / 41
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
        combined_df = log_with_zero(combined_df, 'fold_energy', pos=False)
        combined_df['fold_energy'] = -combined_df['fold_energy']
        print(combined_df['fold_energy'])

        model = LinearRegression()
    elif feature == 'fold_energy_begin':
        combined_df = log_with_zero(combined_df, 'fold_energy_begin', pos=False)
        combined_df['fold_energy_begin'] = -combined_df['fold_energy_begin']
        print(combined_df['fold_energy_begin'])

        model = LinearRegression()

    elif feature.startswith('fold_duplex_'):
        df_copy = log_with_zero(combined_df, feature, pos=False)
        combined_df[feature] = -df_copy[feature]

        model = LinearRegression()
    elif feature.startswith('fold_cofold_'):
        df_copy = log_with_zero(combined_df, feature, pos=False)
        combined_df[feature] = -df_copy[feature]

        model = LinearRegression()
    elif feature == 'cARSscoresCodons':
        combined_df = log_with_zero(combined_df, feature)
        model = LinearRegression()
    elif feature.startswith('cofold_start_'):
        # df_copy = log_with_zero(combined_df, feature, pos=False)
        combined_df[feature] = np.log(-combined_df[feature])
        model = LinearRegression()
    elif feature.startswith('duplex_ends_gfp_'):
        # df_copy = log_with_zero(combined_df, feature, pos=False)
        combined_df[feature] = np.log(-combined_df[feature])
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

    if feature not in ['ENC', 'gc_content', 'gc_deviation', 'size', 'cofold_start_gfp', 'cofold_start_rfp']:
        raise ValueError(f"Unknown feature: {feature}")
    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    combined_df = merged_df.dropna(subset=[fp_type]).copy()
    combined_df = log_with_zero(combined_df, fp_type)

    if feature == 'size':
        combined_df['size'] = np.log(combined_df['size'])
    elif feature == 'gc_content':
        combined_df['gc_content'] = np.log(combined_df['gc_content'])
    elif feature.startswith('cofold_start_'):
        combined_df[feature] = np.log(-combined_df[feature])
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


def get_all_features(fp_type='gfp_intensity'):
    fp_name = 'gfp' if fp_type == 'gfp_intensity' else 'rfp'

    features = ['second_fp', 'ENC',
                'gc_content', 'gc_content_poly1', 'gc_content_poly2', # TODO: fix this
                'native_expression', 'tAI',
                'size', 'size_poly1', 'size_poly2', # TODO: fix this
                'cAI50', 'cAI200',
                'cAI200_first50', 'cAI200_last50', fp_name + '_tai', 'fold_duplex_' + fp_name,
                'cofold_start_' + fp_name,
                'rscu',
                # 'rscu_distance_' + fp_name,
                # 'rscu_distance_asym_' + fp_name
                'rscu_distance_weighted_asym_' + fp_name
                ]
    return features


def validate_fp_type(fp_type):
    if fp_type not in ['gfp_intensity', 'rfp_intensity']:
        raise ValueError(f"Unknown fp_type: {fp_type}")


def get_standard_features_from_df(df: pd.DataFrame, fp_type='gfp_intensity', features=None, native_expression=False,
                                  second_fp=False, out_df=None):
    fp_name = 'gfp' if fp_type == 'gfp_intensity' else 'rfp'
    second_fp_feature_name = 'rfp_intensity' if fp_type == 'gfp_intensity' else 'gfp_intensity'

    validate_fp_type(fp_type)

    if features is None:
        features = get_all_features(fp_type=fp_type)

    # combined_df = df.dropna(subset=[fp_type]).copy()
    combined_df = df
    if second_fp:
        combined_df = combined_df[combined_df[second_fp_feature_name].notna()]

        combined_df = combined_df[combined_df[second_fp_feature_name] != 0]  # Useless as a feature
        combined_df = log_with_zero(combined_df, second_fp_feature_name)

    # Log the fluorescent protein intensity, because expression is exponential
    combined_df = log_with_zero(combined_df, fp_type)

    x_columns = []
    actually_used = []

    # Special features
    if native_expression:
        combined_df = combined_df.dropna(subset=['native_expression'])
        combined_df['native_expression'] = np.log(combined_df['native_expression'])
        x_columns.append(combined_df['native_expression'].to_numpy().reshape(-1, 1))
        actually_used.append('native_expression')

    if 'ENC' in features:
        combined_df['ENC'] = (combined_df['ENC'] - 20) / 44
        combined_df['ENC'] = np.log(combined_df['ENC'])
        x_columns.append(combined_df['ENC'].to_numpy().reshape(-1, 1))
        actually_used.append('ENC')
    if 'tAI' in features:
        combined_df['tAI'] = np.log(combined_df['tAI'])
        x_columns.append(combined_df['tAI'].to_numpy().reshape(-1, 1))
        actually_used.append('tAI')
    if 'rscu' in features:
        combined_df['rscu'] = np.log(combined_df['rscu'])
        x_columns.append(combined_df['rscu'].to_numpy().reshape(-1, 1))
        actually_used.append('rscu')
    if 'rscu_distance_' + fp_name in features:
        feature_name = 'rscu_distance_' + fp_name
        combined_df[feature_name] = np.log(combined_df[feature_name])
        x_columns.append(combined_df[feature_name].to_numpy().reshape(-1, 1))
        actually_used.append(feature_name)
    if 'rscu_distance_asym_' + fp_name in features:
        feature_name = 'rscu_distance_asym_' + fp_name
        combined_df[feature_name] = np.log(combined_df[feature_name])
        x_columns.append(combined_df[feature_name].to_numpy().reshape(-1, 1))
        actually_used.append(feature_name)
    if 'rscu_distance_weighted_asym_' + fp_name in features:
        feature_name = 'rscu_distance_weighted_asym_' + fp_name
        combined_df[feature_name] = np.log(combined_df[feature_name])
        x_columns.append(combined_df[feature_name].to_numpy().reshape(-1, 1))
        actually_used.append(feature_name)
    if 'gc_content' in features:
        gc_content_x = combined_df['gc_content'].to_numpy().reshape(-1, 1)
        gc_content_x = np.log(gc_content_x)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        gc_content_poly = poly.fit_transform(gc_content_x)

        x_columns.append(gc_content_poly)
        actually_used.append('gc_content_poly1')
        actually_used.append('gc_content_poly2')
        if not native_expression and not second_fp:
            combined_df['gc_content_poly1'] = gc_content_poly[:, 0] # TODO
            combined_df['gc_content_poly2'] = gc_content_poly[:, 1] # TODO

    if 'size' in features:
        max_size = 14733
        combined_df['log_size'] = np.log(combined_df['size'] / max_size)

        size_x = combined_df['log_size'].to_numpy().reshape(-1, 1)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        size_poly = poly.fit_transform(size_x)
        x_columns.append(size_poly)
        actually_used.append('size_poly1')
        actually_used.append('size_poly2')
        if not native_expression and not second_fp:
            combined_df['size_poly1'] = size_poly[:, 0] # TODO
            combined_df['size_poly2'] = size_poly[:, 1] # TODO


    if 'cofold_start_' + fp_name in features:
        feature_name = 'cofold_start_' + fp_name
        combined_df[feature_name] = np.log(-combined_df[feature_name])
        x_columns.append(combined_df[feature_name].to_numpy().reshape(-1, 1))
        actually_used.append(feature_name)

    # if fp_type + 'tai' in features:
    #     x_columns.append(combined_df[fp_type + '_tai'].to_numpy().reshape(-1, 1))

    # if fp_type + '_charge' in features:
    #     x_columns.append(combined_df[fp_type + '_charge'].to_numpy().reshape(-1, 1))
    # if fp_type + '_fraction' in features:
    #     x_columns.append(combined_df[fp_type + '_fraction'].to_numpy().reshape(-1, 1))



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
        feature_name = 'fold_duplex_' + fp_name
        df_copy = log_with_zero(combined_df, feature_name, pos=False)
        combined_df[feature_name] = -df_copy[feature_name]
        x_columns.append(combined_df[feature_name].to_numpy().reshape(-1, 1))
        actually_used.append(feature_name)

    print("Actually used features: ", actually_used)
    X_final = np.column_stack(x_columns)

    y = combined_df[fp_type].to_numpy()
    if not out_df is None:
        out_df.drop(out_df.index, inplace=True)  # Drop all rows in df1
        out_df.drop(out_df.columns, axis=1, inplace=True)  # Drop all columns in df1
        for col in combined_df.columns:
            out_df[col] = combined_df[col]  # Add columns and data from df2

    return pd.DataFrame(X_final, columns=actually_used), y


def model_with_features(df: pd.DataFrame, fp_type='gfp_intensity', features=None, native_expression=False,
                        second_fp=False):
    df = df.copy()
    if second_fp:
        second_fp_feature_name = 'rfp_intensity' if fp_type == 'gfp_intensity' else 'gfp_intensity'
        mask = (df[second_fp_feature_name] != 0) & df[[second_fp_feature_name]].notna().all(axis=1)
        # X_all = X_all[mask]
        # y_all = y_all[mask.to_numpy()]
        df = df[mask]
    if native_expression:
        mask = df[['native_expression']].notna().all(axis=1)
        # X_all = X_all[mask]
        # y_all = y_all[mask.to_numpy()]
        df = df[mask]
        df = df.dropna(subset=['native_expression'])

    # combined_df = df.dropna(subset=[fp_type]).copy()

    # all --> including where fp is NaN
    X_all, y_all = get_standard_features_from_df(df.copy(), fp_type, features, native_expression, second_fp)




    mask  = df.reset_index(drop=True)[fp_type].notna()

    X_final, y = X_all[mask], y_all[mask.to_numpy()]


    ###
    # zero_intensity = np.min(df[fp_type].to_numpy())
    # zero_threshold = -3.1 if fp_type == 'rfp_intensity' else -1.
    # low_threshold = (-1.5 if fp_type == 'rfp_intensity' else -1.) + 0.5
    # threshold = low_threshold
    # prob_cutoff = 0.5
    #
    # y_binarized = (y > threshold).astype(int)
    # zeros = len(y_binarized) - np.sum(y_binarized)
    # print("Logistic zero: ", zeros)
    # logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    # logistic_model.fit(X_final, y_binarized)
    # y_probs = logistic_model.predict_proba(X_final)
    #
    # y_pred_custom = (y_probs >= prob_cutoff).astype(int)[:, 1]
    #
    # # print("True predicted: ", np.sum(y_pred_custom == 0))
    # # true_predicted = np.sum(y_pred_custom == 0)
    # # tp = np.sum((y == zero_intensity) & (y_pred_custom == 0) )
    # # print("True positives: ", tp)
    # # print ("TP ratio: ", tp / true_predicted)
    # print("Zeros: ", np.sum(y_pred_custom ==0))
    # print("Ones: ", np.sum(y_pred_custom ==1))
    #
    # X_lowly = X_final[y_pred_custom == 0]
    # y_lowly = y[y_pred_custom == 0]
    # X_highly = X_final[y_pred_custom == 1]
    # y_highly = y[y_pred_custom == 1]
    # model1 = QuantileRegressor(quantile=0.5, alpha=0.001)
    # model1.fit(X_lowly, y_lowly)
    # ylow_predicted = model1.predict(X_lowly)
    # # ylow_predicted = np.ones_like(y_lowly) * zero_intensity
    # model2 = QuantileRegressor(quantile=0.5, alpha=0.001)
    # model2.fit(X_highly, y_highly)
    # yhigh_predicted = model2.predict(X_highly)
    #
    # y_combined = np.concatenate((y_lowly, y_highly))
    # y_predicted = np.concatenate((ylow_predicted, yhigh_predicted))
    #
    # pearson_correlation, _ = pearsonr(y_combined, y_predicted)
    # spearman_correlation, _ = spearmanr(y_combined, y_predicted)
    # print("Model score: ", determination(y_combined, y_predicted))
    # print(f"Pearson Correlation: {pearson_correlation}")
    # print(f"Spearman Correlation: {spearman_correlation}")
    #
    # print(f"Intercept1: {model1.intercept_} coeffs: {model1.coef_}")
    # print(f"Intercept2: {model2.intercept_} coeffs: {model2.coef_}")
    # return model1
    ###

    model = QuantileRegressor(quantile=0.5, alpha=0.)
    model.fit(X_final, y)

    print(f"Fit: {len(y)} points")

    y_predicted = model.predict(X_final)

    pearson_correlation, _ = pearsonr(y, y_predicted)
    spearman_correlation, _ = spearmanr(y, y_predicted)
    print("Model score: ", model.score(X_final, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")

    print(f"Intercept: {model.intercept_} coeffs: {model.coef_}")

    return model, X_all, y_all


def good_model(merged_df, fp_type='gfp_intensity', features=None):
    fp_name = 'gfp' if fp_type == 'gfp_intensity' else 'rfp'
    second_fp_type = 'gfp_intensity' if fp_type == 'gfp_intensity' else 'rfp_intensity'
    validate_fp_type(fp_type)
    if features is None:
        features = get_all_features(fp_type)

    print("Starting good model fit: ")

    model_expression, x_all_native, y_all_native = model_with_features(merged_df.copy(), fp_type, native_expression=True)
    model_expression_second_fp, x_all_native_fp, y_all_native_fp = model_with_features(merged_df.copy(), fp_type, native_expression=True, second_fp=True)

    model_generic, x_all_generic, y_all_generic = model_with_features(merged_df.copy(), fp_type)



    joint_df = merged_df[merged_df[['gfp_intensity', 'rfp_intensity']].notna().all(axis=1)]
    joint_df_expression = joint_df[joint_df[['native_expression']].notna().all(axis=1)]
    expression = merged_df[merged_df[['native_expression']].notna().all(axis=1)]

    X_expr_fp, y_expr_fp = get_standard_features_from_df(joint_df_expression.copy(), fp_type, features, native_expression=True,
                                                         second_fp=True)
    print("X_expr_fp_size, ", len(X_expr_fp))
    y_predict_expr_fp = model_expression_second_fp.predict(X_expr_fp)

    previous_features = []

    previous_features.append(X_expr_fp)

    X_expr, y_expr = get_standard_features_from_df(joint_df_expression.copy(), fp_type, features, native_expression=True)

    previous_features.append(X_expr)

    df_to_drop = pd.concat(previous_features).drop_duplicates(keep=False)
    mask = X_expr.index.isin(df_to_drop.index)
    X_expr = X_expr[mask]
    y_expr = y_expr[mask]

    y_predict_expr = model_expression.predict(X_expr)

    X, y_rest = get_standard_features_from_df(joint_df_expression.copy(), fp_type, features)
    previous_features.append(X)


    df_to_drop = pd.concat(previous_features).drop_duplicates(keep=False)
    mask = X.index.isin(df_to_drop.index)
    X = X[mask]
    y_rest = y_rest[mask]

    y_predict_rest = model_generic.predict(X)

    y = np.concatenate((y_expr_fp, y_expr, y_rest))
    y_predict = np.concatenate((y_predict_expr_fp, y_predict_expr, y_predict_rest))

    pearson_correlation, _ = pearsonr(y, y_predict)
    spearman_correlation, _ = spearmanr(y, y_predict)
    print("Aggregate correlation: ")
    print("Y_predict: ", y_predict)
    print("Model score: ", determination(y, y_predict))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")


    # Now the same thing for the entire sample
    full_df = pd.DataFrame()
    _= get_standard_features_from_df(merged_df, fp_type, features, out_df=full_df)

    df_expression_full = full_df[full_df[['native_expression']].notna().all(axis=1)]
    joint_df_full = df_expression_full[df_expression_full[[second_fp_type]].notna().all(axis=1)]

    native_features = X_expr.columns
    native_fp_features = X_expr_fp.columns
    all_features = X.columns

    y_predict = []
    for index, row in merged_df.iterrows():
        locus_tag = row['ORF']
        if locus_tag in joint_df_full:
            y = model_expression.predict(joint_df_full[joint_df_full['ORF'] == locus_tag][native_fp_features])[0]
        elif locus_tag in df_expression_full:
            y = model_expression_second_fp.predict(df_expression_full[df_expression_full['ORF'] == locus_tag][native_features])[0]
        else:
            y = model_generic.predict(full_df[full_df['ORF'] == locus_tag][all_features])[0]
        y_predict.append((locus_tag, y))
    return pd.DataFrame(y_predict, columns=['ORF', 'fp_intensity'])



def random_forest(df: pd.DataFrame, fp_type='gfp_intensity'):
    mirror_fp = 'rfp_intensity' if fp_type == 'gfp_intensity' else 'gfp_intensity'

    df = df.copy().dropna(subset=[fp_type])

    df.dropna()

    rf = RandomForestRegressor(n_estimators=100)
    df = df.drop(['ORF', mirror_fp], axis=1)

    # df = df[df[fp_type] < 0.1]
    df = log_with_zero(df.copy(), fp_type)
    df['log_size'] = np.log(df['size'] / np.max(df['size']))

    x = df.drop(fp_type, axis=1)
    y = df[fp_type]
    rf.fit(x, y)

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': x.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(feature_importances)


def fp_as_a_feature(merged_df: pd.DataFrame, fp_value='gfp_intensity'):
    fp_feature = 'rfp_intensity' if fp_value == 'gfp_intensity' else 'gfp_intensity'
    print(f"Using {fp_feature} as a feature and {fp_value} as a value")

    joint_df = merged_df[merged_df[['gfp_intensity', 'rfp_intensity']].notna().all(axis=1)]
    ##
    joint_df = joint_df[joint_df['gfp_intensity'] != 0.]
    joint_df = joint_df[joint_df['rfp_intensity'] != 0.]
    ##

    # gfp_not_na = merged_df[merged_df[['gfp_intensity']].notna().all(axis=1)]
    # gfp_not_na_rfp_na = merged_df[merged_df[['gfp_intensity']].notna().all(axis=1) & (merged_df[['rfp_intensity']].isna().all(axis=1))]
    # rfp_not_na_gfp_na = merged_df[merged_df[['rfp_intensity']].notna().all(axis=1) & (merged_df[['gfp_intensity']].isna().all(axis=1))]
    # rfp_not_na = merged_df[merged_df[['rfp_intensity']].notna().all(axis=1)]
    # missing = merged_df[merged_df[['rfp_intensity', 'gfp_intensity']].isna().all(axis=1)]
    #
    #
    #
    # print("Joint FP amount: ", len(joint_df))
    # print("RFP not NA:  ", len(rfp_not_na))
    # print("RFP not NA GFP NA: ", len(rfp_not_na_gfp_na))
    # print("GFP not NA RFP NA: ", len(gfp_not_na_rfp_na))
    # print("GFP not NA: ", len(gfp_not_na))
    # print("Missing amount: ", len(missing))

    joint_df = log_with_zero(joint_df, fp_feature)
    joint_df = log_with_zero(joint_df, fp_value)

    y = joint_df[fp_value]
    x = joint_df[fp_feature].to_numpy().reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)

    y_predict = model.predict(x)
    pearson_correlation, _ = pearsonr(y, y_predict)
    spearman_correlation, _ = spearmanr(y, y_predict)
    print("FP type correlation score: ")
    print("Model score: ", model.score(x, y))
    print(f"Pearson Correlation: {pearson_correlation}")
    print(f"Spearman Correlation: {spearman_correlation}")


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

    fp_as_a_feature(merged_df, 'gfp_intensity')
    fp_as_a_feature(merged_df, 'rfp_intensity')

    # merged_df = log_with_zero(merged_df, 'gfp_intensity')
    # merged_df = log_with_zero(merged_df, 'native_expression')
    # merged_df['gfp_intensity'] = np.exp(merged_df['gfp_intensity'] - 0.22 * merged_df['native_expression']  + 0.91)
    # merged_df['native_expression'] = np.exp(merged_df['native_expression'])

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
    # merged_df = merged_df.sort_values(by='diff', ascending=False).tail(500)
    # merged_df = merged_df.sort_values(by='diff', ascending=False).dropna(subset=['diff']).head(600)
    # merged_df = merged_df[merged_df['native_expression'].isna()]
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(merged_df)

    # merged_df = merged_df[(merged_df['gfp_intensity'] > merged_df['rfp_intensity'])]
    # merged_df = merged_df[(merged_df['gfp_intensity'] > 1) & (merged_df['rfp_intensity'] > 1)]
    # merged_df = merged_df[(merged_df['gfp_intensity'] < 0.5)].copy()
    # merged_df = merged_df[(merged_df['rfp_intensity'] < 0.2)].copy()
    # merged_df.loc[merged_df['gfp_intensity'] < 0.5, 'gfp_size'] = 0.5
    # merged_df.loc[merged_df['gfp_intensity'] > 0.5, 'gfp_size'] = 1
    # print("Merged df: ", merged_df['diff'])
    # print("Merged df: ", merged_df['gfp_intensity'])

    plt.scatter(merged_df['cARSscoresCodons'], merged_df['size'], alpha=0.5)
    plt.xlabel('chimera')
    plt.ylabel('size')
    plt.plot()
    plt.show()

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

    # random_forest(merged_df, 'gfp_intensity')
    # random_forest(merged_df, 'rfp_intensity')

    print("Testing model: ")
    # model_with_features(merged_df, fp_type='gfp_intensity')
    # model_with_features(merged_df, fp_type='rfp_intensity')

    gfp_prediction = good_model(merged_df, fp_type='gfp_intensity')
    rfp_prediction = good_model(merged_df, fp_type='rfp_intensity')
    gfp_prediction['fp_intensity'] = np.exp(gfp_prediction['fp_intensity'])
    rfp_prediction['fp_intensity'] = np.exp(rfp_prediction['fp_intensity'])

    expression = pd.read_csv('data/expression.csv')
    expression.set_index('ORF')
    gfp_prediction.set_index('ORF')
    expression['gfp_intensity'] = expression['gfp_intensity'].fillna(gfp_prediction['fp_intensity'])
    print(expression)
    expression['rfp_intensity'] = expression['rfp_intensity'].fillna(rfp_prediction['fp_intensity'])
    print(expression)

    expression.to_csv('data/expression_prediction.csv', index=False)

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
    one_feature_linear_model(merged_df, feature='fold_energy_begin', fp_type=fp_type)
    one_feature_linear_model(merged_df, feature='fold_energy_begin', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='cofold_start_gfp', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='cofold_start_rfp', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='duplex_ends_gfp', fp_type='gfp_intensity')
    # one_feature_linear_model(merged_df, feature='duplex_ends_rfp', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='fold_cofold_gfp', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='fold_cofold_rfp', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='fold_cofold_gfp', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='fold_cofold_rfp', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='fold_duplex_gfp', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='fold_duplex_rfp', fp_type=fp_type)
    # one_feature_linear_model(merged_df, feature='fold_duplex_gfp', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='fold_duplex_rfp', fp_type='rfp_intensity')
    # # one_feature_linear_model(merged_df, feature='gc_content', fp_type=fp_type)
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
    # one_feature_linear_model(merged_df, feature='tAI', fp_type='rfp_intensity')
    # one_feature_linear_model(merged_df, feature='tAI', fp_type='rfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_gfp', fp_type='gfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_rfp', fp_type='rfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_asym_gfp', fp_type='gfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_asym_rfp', fp_type='rfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_weighted_asym_gfp', fp_type='gfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu_distance_weighted_asym_rfp', fp_type='rfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu', fp_type='gfp_intensity')
    one_feature_linear_model(merged_df, feature='rscu', fp_type='rfp_intensity')
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
