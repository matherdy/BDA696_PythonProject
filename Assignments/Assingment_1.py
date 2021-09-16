import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():

    iris_data = pd.read_csv(
        "Assignments/iris.data",
        sep=",",
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )  # Make sure to add path for data file
    iris_data["class_id"] = 1
    iris_data["class_id"][iris_data["class"] == "Iris-versicolor"] = 2
    iris_data["class_id"][iris_data["class"] == "Iris-virginica"] = 3

    print("Some statistics of the flowers")
    print("*" * 32)
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    iris_mean = np.mean(iris_data[cols])
    print("Mean:")
    print(iris_mean, "\n")

    iris_min = np.min(iris_data[cols])
    print("Min:")
    print(iris_min, "\n")

    iris_quartile_SL = np.quantile(iris_data["sepal_length"], [0.25, 0.5, 0.75])
    iris_quartile_SW = np.quantile(iris_data["sepal_width"], [0.25, 0.5, 0.75])
    iris_quartile_PL = np.quantile(iris_data["petal_length"], [0.25, 0.5, 0.75])
    iris_quartile_PW = np.quantile(iris_data["petal_width"], [0.25, 0.5, 0.75])

    # Another way to show all the descriptive statistics is to do
    # DataFrame.describe() (iris_data.describe() in this case)

    print("Quartiles:     .25/.50/.75")
    print("Sepal Length: ", iris_quartile_SL)
    print("Sepal Width: ", iris_quartile_SW)
    print("Petal Length: ", iris_quartile_PL)
    print("Petal Witdh: ", iris_quartile_PW, "\n")

    fig_violin = px.violin(
        iris_data,
        y=["petal_length", "petal_width", "sepal_length", "sepal_width"],
        color="class",
        labels={
            "variable": "Part of Plant",
            "value": "Measurement in CM",
        },
    )
    fig_violin.show()

    # This code comes from plotly express documentation
    fig_scatter_matrix = px.scatter_matrix(
        iris_data,
        dimensions=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        color="class",
    )
    fig_scatter_matrix.show()

    # This code comes from plotly express documentation
    fig_parallel = px.parallel_coordinates(
        iris_data,
        color="class_id",
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2,
    )
    fig_parallel.show()

    # This code comes from plotly heatmap documentation
    fig_heatmap1 = px.density_heatmap(
        iris_data,
        x="petal_length",
        y="petal_width",
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Viridis",
        title="Petal Measurement Coorelations",
    )
    fig_heatmap1.show()

    fig_heatmap2 = px.density_heatmap(
        iris_data,
        x="sepal_length",
        y="sepal_width",
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Viridis",
        title="Sepal Measurement Coorelations",
    )
    fig_heatmap2.show()

    # This code is adapted from plotlys distplot documentation
    group_labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    data = [
        iris_data["sepal_length"],
        iris_data["sepal_width"],
        iris_data["petal_length"],
        iris_data["petal_width"],
    ]
    fig_distplot = ff.create_distplot(data, group_labels, bin_size=0.4)
    fig_distplot.show()

    # seperating the predictors from the target
    x = iris_data.iloc[:, [True, True, True, True, False, False]]
    y = iris_data["class"]

    pipe_forest = Pipeline(
        [("scaler", StandardScaler()), ("forest", RandomForestClassifier())]
    )
    pipe_forest.fit(x, y)
    # print(pipe_forest.predict(x))
    print("Prediction mean accuracy for the Forest Classifier")
    print(pipe_forest.score(x, y))

    pipe_svm = Pipeline([("scaler", StandardScaler()), ("svm", svm.SVC())])
    pipe_svm.fit(x, y)
    # print(pipe_svm.predict(x))
    print("Prediction mean accuracy for the Support Vector Classifier")
    print(pipe_svm.score(x, y))

    pipe = Pipeline([("scaler", StandardScaler()), ("svm", LogisticRegression())])
    pipe.fit(x, y)
    # print(pipe.predict(x))
    print("Prediction mean accuracy for the Logistic Regression Classifier")
    print(pipe.score(x, y))


if __name__ == "__main__":

    sys.exit(main())
