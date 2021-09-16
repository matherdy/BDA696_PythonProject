import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris_data = pd.read_csv(
    "Assignments/iris.data",
    sep=",",
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)  # Make sure to add path for data file
x = iris_data.iloc[:, [True, True, True, True, False]]
y = iris_data["class"]

# scaler = StandardScaler()
# print(scaler.fit(iris_data.iloc[:,[True,True,True,True,False]]))
# print(scaler.mean_)
# scaled_data = scaler.transform(iris_data.iloc[:,[True,True,True,True,False]])

# forest_classifier = RandomForestClassifier(max_depth = 2, random_state = 0)
# forest_classifier.fit(scaled_data,iris_data["class"])

# print(forest_classifier.predict(scaled_data))

# pipe_forest = Pipeline([("scaler", StandardScaler()),("forest",RandomForestClassifier())])
# pipe_forest.fit(x,y)
# print(pipe_forest.predict(x))
# print(pipe_forest.score(x,y))

# pipe_svm = Pipeline([("scaler", StandardScaler()),("svm",svm.SVC())])
# pipe_svm.fit(x,y)
# print(pipe_svm.predict(x))
# print(pipe_svm.score(x,y))

# pipe = Pipeline([("scaler", StandardScaler()),("svm",LogisticRegression())])
# pipe.fit(x,y)
# print(pipe.predict(x))
# print(pipe.score(x,y))


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
