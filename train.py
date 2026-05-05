from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from utils.pre_processing import *
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline

df = pd.read_csv("data/smoke_detection.csv", delimiter=',')
data = Dataframe(df)
print(data.df.info()) #só variáveis numéricas
data.print_missing() #nada faltando
data.drop_columns(['Unnamed: 0']) #desconsiderando indice da linha
#notação científica tava atrapalhando visualização de outliers
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print(data.df.describe().T)
#sinais de outliers nas colunas:
colunas_outliers = ['TVOC[ppb]', 'eCO2[ppm]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
data.box_plot_multi(colunas_outliers, "Distribuição em colunas com sinal de outlier")

data.apply_log(colunas_outliers) #faz sentido para cauda muito longa, muitos outliers e regressão logística como baseline
"""
como é um problema de classificação binária,incêndio ou não incêndio, 
regressão logística usada como baseline para provar se um algoritmo mais complexo vale a pena
random forest escolhido pelas variáveis em escalas heterogêneas e 
multicolinearidade presente entre os sensores, bom para mapear picos não-lineares de fumaça e 
calcula a importância das variáveis -> permite identificar sensores redundantes
"""
data.separar_base('Fire Alarm', columns=['Fire Alarm', 'UTC', 'CNT']) #removidas coluna de contagem de tempo e amostras pq modelo estava usando de gabarito
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'Fire Alarm')

log_reg = Pipeline([('smote', SMOTE(random_state=42)),
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(max_iter=1000))])

log_reg.fit(data.X_train, data.y_train)
pred_treino = log_reg.predict(data.X_train)
print(f"\nAcurácia da regressão logística nos dados de treino: {accuracy_score(data.y_train, pred_treino) * 100:.2f}%\n")
print(classification_report(data.y_train, pred_treino))

cv_log = cross_val_score(log_reg, data.X_train, data.y_train, cv=5)
print(f"Scores individuais da regressão logística: {np.round(cv_log * 100, 2)}%")
print(f"Média dos 5 folds: {cv_log.mean() * 100:.2f}%")
#modelo generaliza

previsoes = log_reg.predict(data.X_test)
print(f"\nAcurácia da regressão logistica: {accuracy_score(data.y_test, previsoes) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, previsoes)}")

random_forest = Pipeline([('smote', SMOTE(random_state=42)),
                          ('model', RandomForestClassifier(random_state=42))])

random_forest.fit(data.X_train, data.y_train)
pred_treino = random_forest.predict(data.X_train)
print(f"\nAcurácia de random forest nos dados de treino: {accuracy_score(data.y_train, pred_treino) * 100:.2f}%\n")
print(classification_report(data.y_train, pred_treino))

y_pred = random_forest.predict(data.X_test)
print(f"Acurácia de random forest: {accuracy_score(data.y_test, y_pred) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred)}")


param_grid = {'model__n_estimators': [50, 100, 200],'model__max_depth': [5,  10, 20],
              'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4],
              'model__max_features': ['sqrt', 'log2', None]}

random_forest_base = Pipeline([('smote', SMOTE(random_state=42)),
                               ('model', RandomForestClassifier(random_state=42))])
#50 combinações aleatórias, 5 folds de validação cruzada
random_search = RandomizedSearchCV(estimator=random_forest_base, param_distributions=param_grid,
                                   n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

random_search.fit(data.X_train, data.y_train)
melhor_modelo = random_search.best_estimator_
print(f"\nMelhores Hiperparâmetros: {random_search.best_params_}\n")

y_pred_rs = melhor_modelo.predict(data.X_test)
print(f"Acurácia do melhor modelo nos dados de teste: {accuracy_score(data.y_test, y_pred_rs) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred_rs)}")

data.heatmap() #7 sensores PM e NC trazem basicamente a mesma informação
print(f"Análise de Multicolinearidade\n: {data.get_vif()}\n")
colunas_top3 = data.X_train.columns[random_forest.named_steps['model'].feature_importances_.argsort()[-3:]]

print(f"Sensores mantidos no modelo reduzido: {list(colunas_top3)}\n")
X_train_reduzido = data.X_train[colunas_top3]
X_test_reduzido = data.X_test[colunas_top3]

random_forest_reduzido = Pipeline([('smote', SMOTE(random_state=0)),
                                   ('model', RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42))])
cv_rf_reduzido = cross_val_score(random_forest_reduzido, X_train_reduzido, data.y_train, cv=5)
print(f"Scores individuais por Fold: {np.round(cv_rf_reduzido * 100, 2)}%")
print(f"Média dos 5 folds: {cv_rf_reduzido.mean() * 100:.2f}%\n")

random_forest_reduzido.fit(X_train_reduzido, data.y_train)
y_pred_reduzido = random_forest_reduzido.predict(X_test_reduzido)
print(f"Acurácia do modelo reduzido: {accuracy_score(data.y_test, y_pred_reduzido) * 100:.2f}%\n")
print(f"Relatório de Classificação do modelo reduzido:\n{classification_report(data.y_test, y_pred_reduzido)}")