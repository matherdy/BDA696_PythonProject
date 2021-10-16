import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix


def is_continuous(data, col):
    # This Function takes in a column of a pandas data frame and returns a boolean depending on if the column variables
    # are continuous or not.

    if len(data[col].unique()) <= 3 or data[col].dtype == "O":
        return False
    return True


def load_data():

    while True:
        data_name = input("Enter data path: ")
        try:
            data = pd.read_csv(data_name)
            break
        except OSError:
            print("Dataset not found, try again")
    while True:
        response_name = input("Enter response column name: ")
        if response_name in data.columns:

            break
        else:
            print("Column name not found, try again")
    return data, response_name


def plot_catp_catr(pred, resp, data):
    data[pred] = data[pred].astype("string")
    conf_matrix = confusion_matrix(data[pred], data[resp])
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
    fig.update_layout(
        title=f"Categorical Predictor: {pred} by Categorical Response {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig.show()
    fig.write_html(
        file=f"./plots/cat_{resp}_by_cat_{pred}_heatmap.html",
        include_plotlyjs="cdn",
    )


def plot_catp_contr(pred, resp, data):
    data[pred] = data[pred].astype("string")
    cat_pred_list = list(data[pred].unique())
    cont_resp_list = [
        data[resp][data[pred] == pred_name] for pred_name in cat_pred_list
    ]

    fig_1 = ff.create_distplot(cont_resp_list, cat_pred_list, bin_size=0.2)
    fig_1.update_layout(
        title=f"Continuous Response: {resp} by Categorical Predictor: {pred}",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file=f"./plots/cont_{resp}_cat_{pred}_dist_plot.html",
        include_plotlyjs="cdn",
    )


def plot_contp_catr(pred, resp, data):
    data[resp] = data[resp].astype("string")
    cat_resp_list = list(data[resp].unique())
    n = len(cat_resp_list)
    cont_pred_list = [
        data[pred][data[resp] == resp_name] for resp_name in cat_resp_list
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
    fig_2.update_layout(
        title=f"Continuous Predictor: {pred} by Categorical Response: {resp}",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()
    fig_2.write_html(
        file=f"./plots/cat_{resp}_cont_{pred}_violin_plot.html",
        include_plotlyjs="cdn",
    )


def plot_contp_contr(pred, resp, data):
    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Continuous Response: {resp} by Continuous Predictor: {pred}",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()
    fig.write_html(
        file=f"./plots/cont_{resp}_cont_{pred}_scatter_plot.html",
        include_plotlyjs="cdn",
    )


def calc_reg(pred, resp, data):
    predictor = sm.add_constant(data[pred])
    model = sm.OLS(data[resp], predictor)
    model_fit = model.fit()
    p_val = "{:.6e}".format(model_fit.pvalues[1])
    t_val = round(model_fit.tvalues[1], 6)

    print(f"Predictor: {pred}")
    print(model_fit.summary())

    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred}: (t-value={t_val}) (p-value={p_val})",
        xaxis_title=f"Variable: {pred}",
        yaxis_title="y",
    )
    fig.show()

    fig.write_html(
        file=f"./plots/cont_{resp}_cont_{pred}_reg_plot.html",
        include_plotlyjs="cdn",
    )

    return p_val, t_val


def calc_log_reg(pred, resp, data):
    model_fit = sm.Logit(data[resp], data[pred]).fit()
    print(model_fit.summary())
    p_val = "{:.6e}".format(model_fit.pvalues[0])
    t_val = round(model_fit.tvalues[0], 6)
    fig = px.scatter(x=data[pred], y=data[resp], trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred}: (t-value={t_val}) (p-value={p_val})",
        xaxis_title=f"Variable: {pred}",
        yaxis_title="y",
    )
    fig.show()

    fig.write_html(
        file=f"./plots/cont_{resp}_cont_{pred}_log_reg_plot.html",
        include_plotlyjs="cdn",
    )
    return p_val, t_val


# def calc_rand_forest_importance(pred,resp,data)


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


def plot_mean_diff(df):

    fig = px.bar(x=df["(i)"], y=df["Bin_Mean"])
    fig.add_hline(y=df["Pop_Mean"][0])
    fig.show()


def main():
    variable_df = pd.DataFrame(
        columns=[
            "Predictor",
            "Response",
            "P Value",
            "T Value",
            "Regression Plot",
            "General Plot",
            "Difference of Mean Plot",
            "Mean Squared Differnece Weighted",
        ]
    )

    data_df, response = load_data()
    # col = "cyl"
    # plot_catp_contr(col, response, data_df)
    # msd_tbl = mean_diff_tbl(col, response, data_df)
    # sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
    # plot_mean_diff(msd_tbl)
    # print(col,response)
    # p_val, t_val = calc_reg(col, response, data_df)

    if is_continuous(data_df, response):
        for i, col in enumerate(
            data_df.loc[:, ~data_df.columns.isin([response, "Unnamed: 0"])].columns
        ):

            if is_continuous(data_df, col):
                plot_contp_contr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)
                p_val, t_val = calc_reg(col, response, data_df)
            else:
                plot_catp_contr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)
                p_val, t_val = calc_reg(col, response, data_df)
            variable_df.loc[i] = [
                col,
                response,
                p_val,
                t_val,
                "see output",
                "see output",
                "see output",
                sum_msdw,
            ]
    else:
        for i, col in enumerate(
            data_df.loc[:, ~data_df.columns.isin([response, "Unnamed: 0"])].columns
        ):

            if is_continuous(data_df, col):
                plot_contp_catr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)

            else:
                plot_catp_catr(col, response, data_df)
                msd_tbl = mean_diff_tbl(col, response, data_df)
                sum_msdw = sum(msd_tbl["Mean_Sq_Diff_Weighted"])
                plot_mean_diff(msd_tbl)

            variable_df.loc[i] = [
                col,
                response,
                "Catigorical Response",
                "Catigorical Response",
                "see output",
                "see output",
                "see output",
                sum_msdw,
            ]

    print(variable_df)


if __name__ == "__main__":
    sys.exit(main())
