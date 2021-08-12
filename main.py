import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectKBest, f_classif,
                                       mutual_info_classif)

plt.style.use("ggplot")


file_location = "PATH_TO_DATASET"
df_original = pd.read_csv(file_location)
df_features = df_original.drop(columns=["sample index", "class_label"])
df_class_label = df_original[["class_label"]]


X = df_features
Y = np.ravel(df_class_label)

# Pearson correlation between features, and between features and target class.
plt.figure(figsize=(12, 10))
sns.heatmap(df_original.corr().round(2), cmap="Blues", annot=True, linewidths=2)
plt.show()

# Target class distribution check.
sns.countplot(x="class_label", data=df_class_label)
plt.show()


# ANOVA-F test

selector = SelectKBest(f_classif, k=10)
selector.fit_transform(X, Y)
sensor_rank_dict_anova = dict()
for i, k in enumerate(selector.scores_):
    sensor_rank_dict_anova["sensor" + str(i)] = round(k, 3)
sensor_rank_dict_sorted_anova = dict(
    sorted(sensor_rank_dict_anova.items(), key=lambda x: x[1], reverse=True)
)
print(f"Ranked sensor (descending order): \n\n{sensor_rank_dict_sorted_anova}")


# Plotting for ANOVA F test
print("\nPlotting feature importance: \n")
plt.style.use("ggplot")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("F-score")
plt.bar(
    sensor_rank_dict_sorted_anova.keys(),
    sensor_rank_dict_sorted_anova.values(),
    color="tab:blue",
)
plt.show()


# Information Gain

information_gain = mutual_info_classif(X, Y)
sensor_rank_dict_ig = dict()
for i, k in enumerate(information_gain):
    sensor_rank_dict_ig["sensor" + str(i)] = round(k, 3)
sorted_sensor_rank_dictionary_ig = dict(
    sorted(sensor_rank_dict_ig.items(), key=lambda x: x[1], reverse=True)
)
print(f"Ranked sensor (descending order): \n\n{sorted_sensor_rank_dictionary_ig}")

print("\nPlotting feature importance: \n")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Information Gain")
plt.bar(
    sorted_sensor_rank_dictionary_ig.keys(),
    sorted_sensor_rank_dictionary_ig.values(),
    color="tab:blue",
)
plt.show()


# Recursive feature elimination using Random forest classifier as the estimator.
rfe = RFE(
    estimator=RandomForestClassifier(), n_features_to_select=1, step=1, verbose=False
)
rfe.fit(X, Y)

ranked_sensor_rfe = dict()
for i, k in enumerate(rfe.ranking_):
    ranked_sensor_rfe["sensor" + str(i)] = k
ranked_sensor_rfe_sorted = dict(
    sorted(ranked_sensor_rfe.items(), key=lambda x: x[1], reverse=False)
)

print(ranked_sensor_rfe_sorted)


model_rfc = RandomForestClassifier()
model_rfc.fit(X, Y)

model_rfc_feature_imp = model_rfc.feature_importances_

sensor_rank_rfc = dict()
for i, k in enumerate(model_rfc_feature_imp):
    sensor_rank_rfc["sensor" + str(i)] = round(k, 3)
sensor_rank_rfc_sorted = dict(
    sorted(sensor_rank_rfc.items(), key=lambda x: x[1], reverse=True)
)

print(f"Ranked sensor (descending order): \n\n{sensor_rank_rfc_sorted}")

print("\nPlotting feature importance: \n")
plt.style.use("ggplot")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Gini Importance")
plt.bar(
    sensor_rank_rfc_sorted.keys(), sensor_rank_rfc_sorted.values(), color="tab:blue"
)
plt.show()
