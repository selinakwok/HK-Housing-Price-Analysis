import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import traceback
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

# ----- Data cleaning -----
district_name = pd.read_csv("district_translate.csv").rename(columns={'district_chi': 'district'})
property_df = pd.read_csv("data_all.csv")
property_df = property_df.merge(district_name, on="district", how="left")
property_df = property_df.drop(columns=["developer", "url", "district"]).rename(columns={'district_eng': 'district'})

demography = pd.read_csv("district_census.csv")
property_df = property_df.merge(demography, on="district", how="left")
property_df = property_df.drop(columns="district").rename(columns={'weighted_population': 'd_population',
                                                                   'weighted_age': "d_age",
                                                                   'weighted_income': "d_income",
                                                                   'weighted_education': "d_education",
                                                                   'weighted_labourforce': "d_labourforce"
                                                                   })

def translate_floor(x):
    if x == '高層':
        return 3
    elif x == '中層':
        return 2
    elif x == '低層':
        return 1
    else:
        return np.NAN

def translate_direction(x):
    if x == "北":  # N
        return 1
    elif x == "東北":  # NE
        return 2
    elif x == "東":  # E
        return 3
    elif x == "東南":  # SE
        return 4
    elif x == "南":  # S
        return 5
    elif x == "西南":  # SW
        return 6
    elif x == "西":  # W
        return 7
    elif x == "西北":  # NW
        return 8
    else:
        return np.NAN

tree_df = property_df.replace(-999, np.NAN)
tree_df['floor'] = tree_df['floor'].apply(translate_floor)
tree_df['direction'] = tree_df['direction'].apply(translate_direction)
tree_df = tree_df.dropna()
# print(tree_df)
# print(tree_df.head())

plt.subplot(2, 4, 1)
plt.scatter(tree_df["age"], tree_df["price"], s=0.3)
plt.xlabel("age")
plt.subplot(2, 4, 2)
plt.scatter(tree_df["area"], tree_df["price"], s=0.3)
plt.xlabel("area")
plt.subplot(2, 4, 3)
plt.scatter(tree_df["efficiency"], tree_df["price"], s=0.3)
plt.xlabel("efficiency")
plt.subplot(2, 4, 4)
plt.scatter(tree_df["d_population"], tree_df["price"], s=0.3)
plt.xlabel("d_population")
plt.subplot(2, 4, 5)
plt.scatter(tree_df["d_age"], tree_df["price"], s=0.3)
plt.xlabel("d_age")
plt.subplot(2, 4, 6)
plt.scatter(tree_df["d_income"], tree_df["price"], s=0.3)
plt.xlabel("d_income")
plt.subplot(2, 4, 7)
plt.scatter(tree_df["d_education"], tree_df["price"], s=0.3)
plt.xlabel("d_education")
plt.subplot(2, 4, 8)
plt.scatter(tree_df["d_labourforce"], tree_df["price"], s=0.3)
plt.xlabel("d_laborforce")
plt.show()

tree_df.hist(color='black', figsize=(10, 20))
plt.show()

y = tree_df["price"]
X = tree_df[tree_df.columns[tree_df.columns != 'price']]

# ----- Data transformation -----
# log transformation: reduce skew
X_numerical = X.drop(["direction", "floor", "duplex", "sea", "balcony", "garden", "clubhouse", "pool", "mtr", "mall",
                      "park"], axis=1)
X_categorical = X.filter(["direction", "floor", "duplex", "sea", "balcony", "garden", "clubhouse", "pool", "mtr", "mall", "park"])

skewed = X_numerical.apply(lambda x: skew(x.dropna().astype(float)))
rskewed = skewed[skewed > 0.75].index
lskewed = skewed[skewed < -0.75].index  # None
X_numerical[rskewed].hist(bins=20, figsize=(15, 7), color='lightblue', xlabelsize=0, ylabelsize=0, grid=False, layout=(2, 4))
plt.show()
X_numerical[rskewed] = np.log1p(X_numerical[rskewed])
X_numerical[rskewed].hist(bins=20, figsize=(15, 7), color='lightblue', xlabelsize=0, ylabelsize=0, grid=False, layout=(2, 4))
plt.show()
X_log = pd.concat([X_numerical, X_categorical], axis=1)

# Standard scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# X_scaled.boxplot(vert=False, figsize=(8, 8))
# plt.show()

# ----- Feature selection -----
X_select = X_log.drop(["duplex"], axis=1)
X_select2 = X_log.drop(["duplex", "garden"], axis=1)

# ----- Decision Tree -----
rng = 1

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

regtree = DecisionTreeRegressor(random_state=rng)

param_grid = {
    'max_depth': list(range(8, 22, 2)),
    'min_samples_split': range(2, 42, 5),
    'min_samples_leaf': range(1, 22, 2)
}
grid_search = GridSearchCV(estimator=regtree, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1)

def decision_tree(name, x, tuned):
    print(f"> Model: {name}")
    (X_train, X_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=rng)
    if not tuned:
        model = regtree.fit(X_train, y_train)
    else:
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("Best gridsearch parameters:", grid_search.best_params_)
    # print(f"max depth: {regtree.tree_.max_depth}")
    y_pred = model.predict(X_test)
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred), '\n')
    plot_feature_importance(model.feature_importances_, X_train.columns, f'REGRESSION DECISION TREE [{name}]')

# decision_tree("Base", X, False)
decision_tree("Base", X, True)
# decision_tree("Log", X_log, False)
decision_tree("Log", X_log, True)
# decision_tree("Standard", X_scaled, False)
# decision_tree("Standard (tuned)", X_scaled, True)
decision_tree("Selection", X_select, True)
decision_tree("Selection2", X_select2, True)

"""import graphviz
dot_data = tree.export_graphviz(regtree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("tree")"""


# ----- Random Forest -----
"""
# initial hyperparameter tuning
regforest = RandomForestRegressor(min_samples_leaf=1, random_state=rng)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': [0.25, 0.5, 0.75, 1.0],
    'max_depth': [10, 12],
    'min_samples_split': [7, 12, 17],
    # 'min_samples_leaf': [1, 3, 5]
}

# results:
# (Base) Best gridsearch parameters: {'max_depth': 12, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 300}
# (Log) Best gridsearch parameters: {'max_depth': 12, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 200}
"""

regforest = RandomForestRegressor(max_depth=12, min_samples_split=7, min_samples_leaf=1, random_state=rng)
param_grid = {
    'n_estimators': [200, 300],
    'max_features': [0.25, 0.5, 0.75, 1.0],
}

f_grid_search = GridSearchCV(estimator=regforest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

def random_forest(name, x):
    print(f"> Model: {name}")
    (X_train, X_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=rng)
    f_grid_search.fit(X_train, y_train)
    model = f_grid_search.best_estimator_
    print("Best gridsearch parameters:", f_grid_search.best_params_)
    y_pred = model.predict(X_test)
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred), '\n')
    plot_feature_importance(model.feature_importances_, X_train.columns, f'REGRESSION RANDOM FOREST [{name}]')

random_forest("Base", X)
random_forest("Log", X_log)
random_forest("Selection", X_select)
random_forest("Selection2", X_select2)
