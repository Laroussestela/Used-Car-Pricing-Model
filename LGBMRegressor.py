from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import visualization
from preparation import data_cleaning, remove_outliers

lgb_params={
            'num_leaves': Integer(100, 1000),
            'max_depth': Integer(3, 20),
            'learning_rate': Real(0.01, 0.5),
            'n_estimators': Integer(100, 1000),
            'subsample': Real(0.4, 1.0),
            'colsample_bytree': Real(0.4, 1.0),
            'reg_alpha': Real(0.0, 0.0001),
            'reg_lambda': Real(0.4, 1.0),
            'min_data_in_leaf': Integer(10, 100),
            'feature_fraction': Real(0.4, 1.0),
            'bagging_fraction': Real(0.4, 1.0),
            'bagging_freq': Integer(3, 20),
            'random_state': Integer(10, 100),
            'min_child_weight': Real(0.01, 0.5),
            'cat_smooth': Integer(10, 100)
}

lgbm_model = LGBMRegressor(random_state=42)

# Realizar la búsqueda bayesiana con validación cruzada
lgbm_model = BayesSearchCV(estimator=lgbm_model,
                    search_spaces=lgb_params,
                    n_iter=32,
                    cv=3,
                    scoring='neg_root_mean_squared_error',
                    random_state=42,
                    verbose=1)


lgbm_model.fit(X_train, y_train)

print("Mejores parámetros:", lgbm_model.best_params_)
print("Mejor RMSE:", -lgbm_model.best_score_)


# BEST MODEL
best_model_lgbm = lgbm_model.best_estimator_

y_pred = best_model_lgbm.predict(X_test)

df_resultados_lgbm = pd.DataFrame({
    'Precio real': y_test,
    'Precio predicho': y_pred
})

print(y_test)
print(df_resultados_lgbm.head(10))


# REAL PRICE VS PREDICT PRICE
plt.figure(figsize=(8, 8))
plt.scatter(df_resultados_lgbm['Precio real'], df_resultados_lgbm['Precio predicho'], alpha=0.6, color='royalblue')
plt.plot([df_resultados_lgbm['Precio real'].min(), df_resultados_lgbm['Precio real'].max()], [df_resultados_lgbm['Precio real'].min(), df_resultados_lgbm['Precio real'].max()], 'r--', lw=2)  # Línea ideal
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Dispersión: Real vs Predicho')
plt.grid(True)
plt.axis('equal')  # Para que la escala sea la misma en ambos ejes
plt.tight_layout()
plt.show()
