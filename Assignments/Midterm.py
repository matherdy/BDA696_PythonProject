import random
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import seaborn
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy import stats
from sklearn import datasets
from sklearn.metrics import confusion_matrix

# from cat_correlation import *
# from load_test_data import *


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = [
                "pclass",
                "sex",
                "age",
                "sibsp",
                "embarked",
                "class",
            ]
            response = "survived"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = pd.Categorical(data_set["CHAS"])
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"
    return data_set, predictors, response


def is_continuous(data, col):
    # This Function takes in a column of a pandas data frame and returns a boolean depending on if the column variables
    # are continuous or not.

    if (
        len(data[col].unique()) <= 0.1 * len(data[col])
        or data[col].dtype == "O"
        or data[col].dtype == str
    ):
        return False
    return True


def cor_calc_cont_cont(data, pred_lst, resp):
    cor_tbl = pd.DataFrame(
        columns=[
            "Predictor 1",
            "pred1_plot",
            "Predictor 2",
            "pred2_plot",
            "Correlation Metric",
        ]
    )
    cor_matrix = np.zeros((len(pred_lst), len(pred_lst)))
    i = 0
    path = "/Users/dylanmather/Documents/Grad_School/Fall_21_temp/BDA696_PythonProject2/Assignments/"
    for x, pred1 in enumerate(pred_lst):
        for y, pred2 in enumerate(pred_lst):
            cor_metric = stats.pearsonr(data[pred1], data[pred2])

            if is_continuous(data, resp):
                file1_name = plot_contp_contr(pred1, resp, data)
                file2_name = plot_contp_contr(pred2, resp, data)
            else:
                file1_name = plot_contp_catr(pred1, resp, data)
                file2_name = plot_contp_catr(pred2, resp, data)

            cor_tbl.loc[i] = [
                pred1,
                path + file1_name,
                pred2,
                path + file2_name,
                cor_metric[0],
            ]
            cor_matrix[x, y] = cor_metric[0]
            i += 1
    sort_tbl = cor_tbl.sort_values(by="Correlation Metric", ascending=False)
    return sort_tbl, cor_matrix


def cor_calc_cont_cat(data, cat_pred_lst, cont_pred_lst, resp):
    cor_tbl = pd.DataFrame(
        columns=[
            "Categorical Predictor",
            "cat_pred_plot",
            "Continous Predictor",
            "cont_pred_plot",
            "Correlation Metric",
        ]
    )
    cor_matrix = np.zeros((len(cat_pred_lst), len(cont_pred_lst)))
    i = 0
    path = "/Users/dylanmather/Documents/Grad_School/Fall_21_temp/BDA696_PythonProject2/Assignments/"
    for x, cat_pred in enumerate(cat_pred_lst):
        for y, cont_pred in enumerate(cont_pred_lst):
            cor_metric = cat_cont_correlation_ratio(data[cat_pred], data[cont_pred])

            if is_continuous(data, resp):
                cont_pred_file_name = plot_contp_contr(cont_pred, resp, data)
                cat_pred_file_name = "i tried"  # plot_catp_contr(cat_pred,resp,data)
            else:
                cont_pred_file_name = plot_contp_catr(cont_pred, resp, data)
                cat_pred_file_name = plot_catp_catr(cat_pred, resp, data)

            cor_tbl.loc[i] = [
                cat_pred,
                path + cont_pred_file_name,
                cont_pred,
                path + cat_pred_file_name,
                cor_metric,
            ]
            cor_matrix[x, y] = cor_metric
            i += 1
    sort_tbl = cor_tbl.sort_values(by="Correlation Metric", ascending=False)
    return sort_tbl, cor_matrix


def cor_calc_cat_cat(data, cat_pred_lst, resp):
    cor_tbl = pd.DataFrame(
        columns=[
            "Predictor 1",
            "pred1_plot",
            "Predictor 2",
            "pred2_plot",
            "Correlation Metric",
        ]
    )
    cor_matrix = np.zeros((len(cat_pred_lst), len(cat_pred_lst)))
    i = 0
    path = "/Users/dylanmather/Documents/Grad_School/Fall_21_temp/BDA696_PythonProject2/Assignments/"
    for x, pred1 in enumerate(cat_pred_lst):
        for y, pred2 in enumerate(cat_pred_lst):
            cor_metric = cat_correlation(data[pred1], data[pred2])

            if is_continuous(data, resp):
                file1_name = "i tried"  # plot_catp_contr(pred1, resp, data)
                file2_name = "i tried"  # plot_catp_contr(pred2, resp, data)
            else:
                file1_name = plot_catp_catr(pred1, resp, data)
                file2_name = plot_catp_catr(pred2, resp, data)

            cor_tbl.loc[i] = [
                pred1,
                path + file1_name,
                pred2,
                path + file2_name,
                cor_metric,
            ]
            cor_matrix[x, y] = cor_metric
            i += 1
    sort_tbl = cor_tbl.sort_values(by="Correlation Metric", ascending=False)
    return sort_tbl, cor_matrix


def split_cont_cat_lst(data):
    cont_lst = []
    cat_lst = []
    for col in data.columns:
        if is_continuous(data, col):
            cont_lst.append(col)
        else:
            cat_lst.append(col)
    return cont_lst, cat_lst


def plot_catp_catr(pred, resp, data):
    # This function plots a heatmap of the two catigrocial variables
    data[pred] = data[pred].astype("string")
    conf_matrix = confusion_matrix(data[pred], data[resp])
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
    fig.update_layout(
        title=f"Categorical Predictor: {pred} by Categorical Response {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    file = f"plots/cat_{resp}_by_cat_{pred}_heatmap.html"
    # fig.show()
    fig.write_html(
        file=f"./plots/cat_{resp}_by_cat_{pred}_heatmap.html",
        include_plotlyjs="cdn",
    )
    return file


def plot_catp_contr(pred, resp, data):
    copy_data = data.copy()
    copy_data[pred] = copy_data[pred].astype(
        "string"
    )  # made this a string since its a catigorical variable
    cat_pred_list = list(copy_data[pred].unique())
    cont_resp_list = [
        copy_data[resp][copy_data[pred] == pred_name] for pred_name in cat_pred_list
    ]

    fig_1 = ff.create_distplot(cont_resp_list, cat_pred_list, bin_size=0.2)
    fig_1.update_layout(
        title=f"Continuous Response: {resp} by Categorical Predictor: {pred}",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    file = f"plots/cont_{resp}_cat_{pred}_dist_plot.html"
    # fig_1.show()
    fig_1.write_html(
        file=f"./plots/cont_{resp}_cat_{pred}_dist_plot.html",
        include_plotlyjs="cdn",
    )
    return file


def plot_contp_catr(pred, resp, data):
    copy_data = data.copy()
    copy_data[resp] = copy_data[resp].astype("string")
    cat_resp_list = list(copy_data[resp].unique())
    n = len(cat_resp_list)
    cont_pred_list = [
        copy_data[pred][copy_data[resp] == resp_name] for resp_name in cat_resp_list
    ]
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(cont_pred_list, cat_resp_list):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    file = f"plots/cat_{resp}_cont_{pred}_violin_plot.html"
    fig_2.update_layout(
        title=f"Continuous Predictor: {pred} by Categorical Response: {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    # fig_2.show()
    fig_2.write_html(
        file=f"./plots/cat_{resp}_cont_{pred}_violin_plot.html",
        include_plotlyjs="cdn",
    )
    return file


def plot_contp_contr(pred, resp, data):
    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Continuous Response: {resp} by Continuous Predictor: {pred}",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    file = f"plots/cont_{resp}_cont_{pred}_scatter_plot.html"
    # fig.show()
    fig.write_html(
        file=f"./plots/cont_{resp}_cont_{pred}_scatter_plot.html",
        include_plotlyjs="cdn",
    )
    return file


def mean_diff_tbl(pred, resp, data):
    df = pd.DataFrame(
        columns=[
            "(i)",
            "Bin_Pop",
            "Bin_Mean",
            "Pop_Mean",
            "Pop_Proportion",
            "Mean_Sq_Diff",
            "Mean_Sq_Diff_Weighted",
        ]
    )
    pop_mean = data[resp].mean()
    if not is_continuous(data, pred):
        for i, val in enumerate(data[pred].unique()):
            bin_mean = data[resp][data[pred] == val].mean()
            bin_pop = len(data[resp][data[pred] == val])
            pop_prop = bin_pop / len(data[resp])
            mean_sq_diff = (pop_mean - bin_mean) ** 2
            mean_sq_diff_w = mean_sq_diff * pop_prop
            df.loc[i] = [
                val,
                bin_pop,
                bin_mean,
                pop_mean,
                pop_prop,
                mean_sq_diff,
                mean_sq_diff_w,
            ]
    else:
        data_sorted = data.sort_values(pred)
        data_sorted["bins"] = pd.cut(data_sorted[pred], 10)
        for i, cur_bin in enumerate(data_sorted["bins"].unique()):
            bin_mean = data_sorted[resp][data_sorted["bins"] == cur_bin].mean()
            bin_pop = len(data_sorted[pred][data_sorted["bins"] == cur_bin])
            pop_prop = bin_pop / len(data_sorted[pred])
            mean_sq_diff = (pop_mean - bin_mean) ** 2
            mean_sq_diff_w = mean_sq_diff * pop_prop
            df.loc[i] = [
                i,
                bin_pop,
                bin_mean,
                pop_mean,
                pop_prop,
                mean_sq_diff,
                mean_sq_diff_w,
            ]

    return df


def brute_force(pred1, pred2, resp, data):
    # df = pd.DataFrame(
    #     columns=[
    #         "x_1",
    #         "x_2",
    #         "Bin_Pop",
    #         "Bin_Mean",
    #         "Pop_Mean",
    #         "Pop_Proportion",
    #         "Mean_Sq_Diff",
    #         "Mean_Sq_Diff_Weighted",
    #     ]
    # )
    copy_data = data.copy()
    if is_continuous(data, pred1):
        copy_data["bins1"] = pd.cut(copy_data[pred1], 10, labels=False)
        if is_continuous(data, pred2):
            copy_data["bins2"] = pd.cut(copy_data[pred2], 10, labels=False)
            calc_matrix = np.zeros((11, 11))
            for x_1, val1 in enumerate(copy_data["bins1"].unique()):
                for x_2, val2 in enumerate(copy_data["bins2"].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data["bins1"] == x_1) & (copy_data["bins2"] == x_2)
                    ].mean()
                    calc_matrix[x_1, x_2] = bin_mean
        else:
            calc_matrix = np.zeros((11, len(copy_data[pred2].unique())))
            for x_1, val1 in enumerate(copy_data["bins1"].unique()):
                for x_2, val2 in enumerate(copy_data[pred2].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data["bins1"] == x_1) & (copy_data[pred2] == val2)
                    ].mean()
                    calc_matrix[x_1, x_2] = bin_mean
    else:
        if is_continuous(data, pred2):
            calc_matrix = np.zeros((len(copy_data[pred1].unique()), 11))
            copy_data["bins2"] = pd.cut(copy_data[pred2], 10, labels=False)
            for x_1, val1 in enumerate(copy_data[pred1].unique()):
                for x_2, val2 in enumerate(copy_data["bins2"].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data[pred1] == val1) & (copy_data["bins2"] == x_2)
                    ].mean()
                    calc_matrix[x_1, x_2] = bin_mean
        else:
            calc_matrix = np.zeros(
                (len(copy_data[pred1].unique()), len(copy_data[pred2].unique()))
            )
            for x_1, val1 in enumerate(copy_data[pred1].unique()):
                for x_2, val2 in enumerate(copy_data[pred2].unique()):
                    bin_mean = copy_data[resp][
                        (copy_data[pred1] == val1) & (copy_data[pred2] == val2)
                    ].mean()
                    calc_matrix[x_1, x_2] = bin_mean
    return calc_matrix


def make_clickable(url, name):
    # This function is from stack overflow link that the professor provided in the slides
    # https://stackoverflow.com/questions/42263946/how-to-create-a-table-with-clickable-hyperlink-in-pandas-jupyter-notebook
    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(
        url, name
    )


def add_plot_links(table, data, resp):
    path = "/Users/dylanmather/Documents/Grad_School/Fall_21_temp/BDA696_PythonProject2/Assignments/"
    table["pred1_filepath"] = "placeholder" * len(table.iloc[0])
    table["pred2_filepath"] = "placeholder" * len(table.iloc[0])
    if is_continuous(data, resp):
        i = 0
        for ind in range(2):
            for pred in table.iloc[:, ind]:
                if is_continuous(data, pred):
                    file_name = plot_contp_contr(pred, resp, data)
                    table[f"pred{ind+1}_filepath"][i] = path + file_name
                    table.iloc[:, ind] = table.apply(
                        lambda x: make_clickable(
                            x[f"pred{ind+1}_filepath"], x.iloc[:, ind]
                        ),
                        axis=1,
                    )
                else:
                    file_name = plot_catp_contr(pred, resp, data)
                    table[f"pred{ind+1}_filepath"][i] = path + file_name
                    table.iloc[:, ind] = table.apply(
                        lambda x: make_clickable(
                            x[f"pred{ind+1}_filepath"], x.iloc[:, ind]
                        ),
                        axis=1,
                    )
                i += 1
    else:
        i = 0
        for ind in range(2):
            for pred in table.iloc[:, ind]:
                if is_continuous(data, pred):
                    file_name = plot_contp_catr(pred, resp, data)
                    table[f"pred{ind+1}_filepath"] = path + file_name
                    table.iloc[:, ind] = table.apply(
                        lambda x: make_clickable(
                            x[f"pred{ind+1}_filepath"], x.iloc[:, ind]
                        ),
                        axis=1,
                    )
                else:
                    file_name = plot_catp_catr(pred, resp, data)
                    table[f"pred{ind+1}_filepath"] = path + file_name
                    table.iloc[:, ind] = table.apply(
                        lambda x: make_clickable(
                            x[f"pred{ind+1}_filepath"], x.iloc[:, ind]
                        ),
                        axis=1,
                    )
                i += 1


def main():
    test_data, pred_lst, response = get_test_data_set("mpg")
    pred_data = test_data.loc[:, test_data.columns != response]
    cont_cols_lst, cat_cols_lst = split_cont_cat_lst(pred_data)

    for col1 in cat_cols_lst:
        for col2 in cont_cols_lst:
            print(
                col1,
                is_continuous(test_data, col1),
                col2,
                is_continuous(test_data, col2),
            )

            print(brute_force(col1, col2, response, test_data))

    print(pred_data[cont_cols_lst].head())
    cont_cont_pred_table, cont_cont_pred_matrix = cor_calc_cont_cont(
        test_data, cont_cols_lst, response
    )
    # add_plot_links(pred_table,test_data,response)
    cont_cont_pred_table["Predictor 1"] = cont_cont_pred_table.apply(
        lambda x: make_clickable(x["pred1_plot"], x["Predictor 1"]), axis=1
    )
    cont_cont_pred_table["Predictor 2"] = cont_cont_pred_table.apply(
        lambda x: make_clickable(x["pred2_plot"], x["Predictor 2"]), axis=1
    )

    cont_cont_pred_table.to_html(
        "html_files/cont_cont_cor.html", escape=False, render_links=True
    )
    print(cont_cont_pred_matrix)

    print(pred_data[cat_cols_lst].head())
    cont_cat_pred_table, pred_matrix2 = cor_calc_cont_cat(
        test_data, cat_cols_lst, cont_cols_lst, response
    )
    cont_cat_pred_table["Categorical Predictor"] = cont_cat_pred_table.apply(
        lambda x: make_clickable(x["cat_pred_plot"], x["Categorical Predictor"]), axis=1
    )
    cont_cat_pred_table["Continous Predictor"] = cont_cat_pred_table.apply(
        lambda x: make_clickable(x["cont_pred_plot"], x["Continous Predictor"]), axis=1
    )

    cont_cat_pred_table.to_html(
        "html_files/cat_cont_cor.html", escape=False, render_links=True
    )

    print(pred_matrix2)

    cat_cat_pred_table, pred_matrix3 = cor_calc_cat_cat(
        test_data, cat_cols_lst, response
    )
    cat_cat_pred_table["Predictor 1"] = cat_cat_pred_table.apply(
        lambda x: make_clickable(x["pred1_plot"], x["Predictor 1"]), axis=1
    )
    cat_cat_pred_table["Predictor 2"] = cat_cat_pred_table.apply(
        lambda x: make_clickable(x["pred2_plot"], x["Predictor 2"]), axis=1
    )
    cat_cat_pred_table.to_html(
        "html_files/cat_cat_cor.html", escape=False, render_links=True
    )

    print(pred_matrix3)

    fig = ff.create_annotated_heatmap(
        cont_cont_pred_matrix, cont_cols_lst, cont_cols_lst, showscale=True
    )
    fig.show()

    fig2 = ff.create_annotated_heatmap(
        pred_matrix2, cont_cols_lst, cat_cols_lst, showscale=True
    )
    fig2.show()

    fig3 = ff.create_annotated_heatmap(
        pred_matrix3, cat_cols_lst, cat_cols_lst, showscale=True
    )
    fig3.show()


if __name__ == "__main__":
    sys.exit(main())
