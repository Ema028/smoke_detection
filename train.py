from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from utils.pre_processing import *
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("data/smoke_detection.csv", delimiter=',')
data = Dataframe(df)
#restrições específicas nos nomes das features do xgboost
data.df.columns = data.df.columns.str.replace(r'[\[\]]', '', regex=True)
print(data.df.info()) #só variáveis numéricas
data.print_missing() #nada faltando
data.drop_columns(['Unnamed: 0']) #desconsiderando indice da linha
#notação científica tava atrapalhando visualização de outliers
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print(data.df.describe().T)
#sinais de outliers nas colunas:
colunas_outliers = ['TVOCppb', 'eCO2ppm', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
data.box_plot_multi(colunas_outliers, "Distribuição de Outliers")
'''
transformação logarítmica testada para preprocessamento antes de regressão logarítmica, 
mas não foi suficiente, muita assimetria e caudas muito longas nos sensores
por isso log+stdscaler foi substituido por quantiletransformer no pipeline
data.apply_log(colunas_outliers)
data.box_plot_multi(colunas_outliers, "Em Escala Logarítmica")'''

data.separar_base('Fire Alarm', columns=['Fire Alarm', 'UTC', 'CNT']) #removidas coluna de contagem de tempo e amostras pq modelo estava usando de gabarito
verificar_base(data.X_train, data.X_test, data.y_train, data.y_test, 'Fire Alarm')

data.heatmap() #7 sensores PM e NC trazem basicamente a mesma informação
print(f"Análise de Multicolinearidade\n: {data.get_vif()}\n")

"""
regressão logística como baseline para testar se a natureza do problema é linear
sofre com a alta multicolinearidade presente na base

xgboost como modelo principal para mapear picos não-lineares na detecção de fumaça,
lida nativamente com escalas heterogêneas, anula o impacto da multicolinearidade e fornece a 
importância das variáveis, usado para isolar apenas os sensores úteis na versão final
"""
colunas_normais = [col for col in data.X_train.columns if col not in colunas_outliers]

preprocessor = ColumnTransformer([('outliers', QuantileTransformer(output_distribution='normal', random_state=42), colunas_outliers),
                                  ('normais', StandardScaler(), colunas_normais)])

log_reg = Pipeline([('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42)),
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
conf_matrix(data.y_test, previsoes, ['Não Incêndio', 'Incêndio'])

#tirando colunas redundantes e testando regressão, ver impacto de reduzir multicolinearidade
colunas_redundantes = ['PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
colunas_outliers = ['TVOCppb', 'eCO2ppm']

X_train_reduzido = data.X_train.drop(columns=colunas_redundantes)
X_test_reduzido = data.X_test.drop(columns=colunas_redundantes)

preprocessor = ColumnTransformer([('outliers', QuantileTransformer(output_distribution='normal', random_state=42), colunas_outliers),
                                  ('normais', StandardScaler(), colunas_normais)])

log_reg_red = Pipeline([('preprocessor', preprocessor),
                        ('smote', SMOTE(random_state=42)),
                        ('model', LogisticRegression(max_iter=1000))])

log_reg_red.fit(X_train_reduzido, data.y_train)
prev_red = log_reg_red.predict(X_test_reduzido)
print(f"\nAcurácia da regressão logistica reduzida: {accuracy_score(data.y_test, prev_red) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, prev_red)}")

xgb = Pipeline([('smote', SMOTE(random_state=42)),
                ('model', XGBClassifier(random_state=0))])

xgb.fit(data.X_train, data.y_train)
pred_treino = xgb.predict(data.X_train)
print(f"\nAcurácia de xgboost nos dados de treino: {accuracy_score(data.y_train, pred_treino) * 100:.2f}%\n")
print(classification_report(data.y_train, pred_treino))

y_pred = xgb.predict(data.X_test)
print(f"Acurácia de xgboost: {accuracy_score(data.y_test, y_pred) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred)}")
'''
a hipótese linear atingiu um bom baseline de 90% de recall(considerando que falsos negativos são críticos em detecção de incêndio) 
após o pré-processamento otimizado, mas provou que 10% dos incêndios reais não conseguem ser explicados só com relações lineares
'''

param_grid = {'model__learning_rate': [0.05, 0.1, 0.2], 'model__max_depth': [4, 5, 6],
              'model__n_estimators': [100, 200, 300], 'model__subsample': [0.8, 0.9, 1.0],
              'model__colsample_bytree': [0.8, 0.9, 1.0]}

xgb_base = Pipeline([('smote', SMOTE(random_state=42)),
                     ('model', XGBClassifier(random_state=0))])
#50 combinações aleatórias, 5 folds de validação cruzada
random_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_grid,
                                   n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

random_search.fit(data.X_train, data.y_train)
melhor_modelo = random_search.best_estimator_
print(f"\nMelhores Hiperparâmetros: {random_search.best_params_}\n")

y_pred_rs = melhor_modelo.predict(data.X_test)
print(f"Acurácia do melhor modelo nos dados de teste: {accuracy_score(data.y_test, y_pred_rs) * 100:.2f}%\n")
print(f"Relatório de Classificação:\n{classification_report(data.y_test, y_pred_rs)}")
conf_matrix(data.y_test, y_pred_rs, ['Não Incêndio', 'Incêndio'])

data.feature_importance(xgb.named_steps['model'], colunas=data.X_test.columns)
#3 sensores são responsáveis por 94,85% da capacidade preditiva do modelo
#(TVOCppb-35.20%, PressurehPa-33.72% e PM1.0-25.93%) (com TemperatureC-2.36% e Raw Ethanol-2.06% chega a 99,27%)

colunas_top3 = ['TVOCppb', 'PressurehPa', 'PM1.0']
X_train_reduzido = data.X_train[colunas_top3]
X_test_reduzido = data.X_test[colunas_top3]

xgb_reduzido = Pipeline([('smote', SMOTE(random_state=0)),
                         ('model', XGBClassifier(random_state=0))])
cv_rf_reduzido = cross_val_score(xgb_reduzido, X_train_reduzido, data.y_train, cv=5)
print(f"Scores individuais por Fold: {np.round(cv_rf_reduzido * 100, 2)}%")
print(f"Média dos 5 folds: {cv_rf_reduzido.mean() * 100:.2f}%\n")

xgb_reduzido.fit(X_train_reduzido, data.y_train)
y_pred_reduzido = xgb_reduzido.predict(X_test_reduzido)
prob = xgb_reduzido.predict_proba(X_test_reduzido)
auc_roc(data.y_test, prob)
df_resultados = pd.DataFrame({'Previsão': previsoes,
                              'Probabilidade Incêndio': np.round(prob[:, 1] * 100, decimals= 2)})
print(df_resultados.head())

plt.figure(figsize=(12, 8))
sns.histplot(df_resultados['Probabilidade Incêndio'], bins=30)
plt.title('Distribuição de Probabilidades de Incêndio')
plt.show()
conf_matrix(data.y_test, y_pred_reduzido, ['Não Incêndio', 'Incêndio'])
print(f"Acurácia do modelo reduzido: {accuracy_score(data.y_test, y_pred_reduzido) * 100:.2f}%\n")
print(f"Relatório de Classificação do modelo reduzido:\n{classification_report(data.y_test, y_pred_reduzido)}")
'''
com só 3 sensores o desempenho se manteve, cravou 100% de Recall para incêndios e 99.74% de acurácia global,
viabilizando a produção e eliminando o custo de hardware excedente
'''