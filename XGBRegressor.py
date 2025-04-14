from skopt.space import Real, Categorical, Integer
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import visualization
from preparation import data_cleaning, remove_outliers


xgb_model = XGBRegressor(tree_method='gpu_hist',
                            predictor='gpu_predictor', 
                            gpu_id=0,
                            random_state=42)

# GRID SEARCH
# param_grid = {
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'n_estimators': [100, 300, 500, 1000],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
# }

# grid_xgb = GridSearchCV(estimator=xgb_model,
#                         param_grid=param_grid,
#                         cv=3,
#                         scoring='neg_root_mean_squared_error',
#                         verbose=1)

# BAYES SEARCH
param_grid = {
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.5),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.7, 1.0),
    'colsample_bytree': Real(0.2, 1.0),
    'num_leaves': Integer(100, 500),
    'reg_alpha': Real(0.0, 0.0001),
    'reg_lambda': Real(0.0, 0.5),
    'min_data_in_leaf': Integer(10, 100),
    'feature_fraction': Real(0.1, 1.0),
    'bagging_fraction': Real(0.1, 1.0),
    'bagging_freq': Integer(3, 10),
    'random_state': Integer(10, 100),
    'min_child_weight': Real(0.01, 0.1),
    'cat_smooth': Integer(10, 100),
}

xgb_model = BayesSearchCV(estimator=xgb_model,
                    search_spaces=param_grid,
                    n_iter=32,
                    cv=3,
                    scoring='neg_root_mean_squared_error',
                    random_state=42,
                    verbose=1)

xgb_model.fit(X_train, y_train)

print("Mejores parámetros:", xgb_model.best_params_)
print("Mejor RMSE:", -xgb_model.best_score_)


# BEST MODEL
best_model = xgb_model.best_estimator_
y_pred = best_model.predict(X_test)

df_resultados_xgb = pd.DataFrame({
    'Precio real': y_test,
    'Precio predicho': y_pred
})
print(y_test)
print(df_resultados_xgb.head(10))


# REAL PRICE VS PREDICT PRICE
plt.figure(figsize=(8, 8))
plt.scatter(df_resultados_xgb['Precio real'], df_resultados_xgb['Precio predicho'], alpha=0.6, color='royalblue')
plt.plot([df_resultados_xgb['Precio real'].min(), df_resultados_xgb['Precio real'].max()], [df_resultados_xgb['Precio real'].min(), df_resultados_xgb['Precio real'].max()], 'r--', lw=2)  # Línea ideal
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Dispersión: Real vs Predicho')
plt.grid(True)
plt.axis('equal')  # Para que la escala sea la misma en ambos ejes
plt.tight_layout()
plt.show()
