import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix,  roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import KNNImputer, SimpleImputer
from statsmodels.tools.tools import add_constant
from imblearn.over_sampling import SMOTE

class Dataframe:
	def __init__(self, df):
		self.df = df
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.X_train_scalled = None
		self.X_test_scalled = None

	def print_missing(self):
		missing = self.df.isnull().sum()
		percent = (missing / len(self.df)) * 100

		missing_table = pd.DataFrame({
			'Missing': missing,
			'Percent (%)': percent
		}).sort_values(by='Missing', ascending=False)
		print(missing_table[missing_table['Missing'] > 0])

	def drop_missing(self):
		self.df = self.df.dropna()

	def drop_columns(self, columns):
		self.df.drop(columns = columns, inplace=True)
		return self.df

	def to_number(self, colunas):
		for col in colunas:
			self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

	def imputar_simples(self, colunas, estrategia='median'):
		"""opções de estratégia: median, mean, most_frequent, constant"""
		imputer = SimpleImputer(strategy=estrategia)
		self.df[colunas] = imputer.fit_transform(self.df[colunas])

	def imputar_knn(self, colunas, n_vizinhos=5):
		#encher valores nulos com base nos parecidos
		imputer = KNNImputer(n_neighbors=n_vizinhos)
		self.df[colunas] = pd.DataFrame(imputer.fit_transform(self.df[colunas]), columns=colunas)

	def print_unique_values(self):
		for column in self.df.select_dtypes(include='str').columns:
			print("\n\n")
			print(self.df[column].value_counts())

	def histogram(self, column, title, x = 12, y = 8, bins = 30):
		plt.figure(figsize=(x, y))
		sns.histplot(self.df[column], bins=bins)
		plt.title(title)
		plt.show()

	def scattergram(self, axis_x, axis_y, title, x = 12, y = 8, alpha=1.0):
		plt.figure(figsize=(x, y))
		sns.scatterplot(x=axis_x, y=axis_y, data=self.df, alpha=alpha)
		plt.title(title)
		plt.xlabel(axis_x)
		plt.ylabel(axis_y)
		plt.show()

	def box_plot(self, axis_x, axis_y, title, x = 12, y = 8):
		plt.figure(figsize=(x, y))
		sns.boxplot(x=axis_x, y=axis_y, data=self.df)
		plt.title(title)
		plt.xlabel(axis_x)
		plt.ylabel(axis_y)
		plt.show()

	def box_plot_multi(self, columns, title, x = 12, y = 8):
		plt.figure(figsize=(x, y))
		sns.boxplot(data=self.df[columns])
		plt.title(title)
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.show()

	def pair_plot(self, hue = None):
		sns.pairplot(self.df, hue = hue)
		plt.show()

	def bar_plot(self, axis_x, axis_y, title, x=12, y=8, estimator= np.mean):
		plt.figure(figsize=(x, y))
		#np.mean for mean, np.sum for sum
		sns.barplot(x=axis_x, y=axis_y, data=self.df,
					estimator=estimator, errorbar=None)
		plt.title(title)
		plt.xlabel(axis_x)
		plt.ylabel(axis_y)
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.show()

	def reg_plot(self, axis_x, axis_y, title, x=12, y=8, scatter_alpha=0.5, line_color='red', log = True):
		plt.figure(figsize=(x, y))
		sns.regplot(x=axis_x, y=axis_y, data=self.df,
					scatter_kws={'alpha': scatter_alpha}, line_kws={'color': line_color}, logistic=log)

		plt.title(title)
		plt.xlabel(axis_x)
		plt.ylabel(axis_y)
		plt.show()

	def heatmap(self, x = 12, y = 8):
		plt.figure(figsize=(x, y))
		sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
		plt.title("Matriz de correlação para variáveis numéricas")
		plt.show()

	def one_hot(self):
		cat_cols = self.df.select_dtypes(include=['object', 'str']).columns
		self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
		return self.df

	def label_encoding(self):
		cat_cols = self.df.select_dtypes(include=['object', 'str']).columns
		dicionario_encoders = {}

		for col in cat_cols:
			le = LabelEncoder()
			self.df[col] = le.fit_transform(self.df[col].astype(str))
			dicionario_encoders[col] = le

		return dicionario_encoders

	def reverter_label_encoding(self, dicionario_encoders, columns):
		for col in columns:
			self.df[col] = dicionario_encoders[col].inverse_transform(self.df[col])

	def one_hot_heatmap(self):
		#one-hot encoding: cria colunas binárias para cada categoria, evita falsa ordem hierárquica como em label encoding
		cat_cols = self.df.select_dtypes(include=['object', 'str']).columns
		df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
		df_encoded.head()

		sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0)
		plt.title("Matriz de correlação com variáveis categóricas")
		plt.show()

	def capping_outliers(self, columns):
		for coluna in columns:
			q1 = self.df[coluna].quantile(0.25)
			q3 = self.df[coluna].quantile(0.75)
			iqr = q3 - q1

			limite_inferior = q1 - 1.5 * iqr
			limite_superior = q3 + 1.5 * iqr
			self.df[coluna] = self.df[coluna].clip(limite_inferior, limite_superior)

	def apply_log(self, columns):
		for col in columns:
			self.df[col] = np.log1p(self.df[col]) #log +1 para caso de 0

	#dividir base em conjuntos de treino (por default 80%) e teste (por default 20%)
	def separar_base(self, target_column, test_size=0.2, random_state=42, columns = None):
		if columns:
			x = self.df.drop(columns=columns) #x é todas as variáveis menos target e variáveis q poderiam causar data leakage
		else:
			x = self.df.drop(columns=[target_column])
		y = self.df[target_column] #y é target

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			x, y, test_size=test_size, random_state=random_state
		)

	def smote(self):
		smote = SMOTE(random_state=42)
		self.y_train = self.y_train.astype(int)
		self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

		print("\nDepois do SMOTE:\n")
		print(self.y_train.value_counts())

	def std_scaler(self):
		scaler = StandardScaler()

		self.X_train_scalled = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
		self.X_test_scalled = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

	def robust_scaler(self):
		scaler = RobustScaler()

		self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
		self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

	def get_vif(self):
		"""
        fator de inflação de variância para identificar multicolinearidade,
        VIF > 10 indica alta redundância
        """
		X = self.X_train.copy() #não modificar dados
		X_const = add_constant(X) #simula intercepto
		vif_df = pd.DataFrame()
		vif_df["Variável"] = X_const.columns
		vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
		return vif_df[vif_df["Variável"] != 'const'].sort_values(by="VIF", ascending=False).reset_index(drop=True)

	def feature_importance(self, modelo, colunas=None):
		if colunas is None:
			colunas = self.X_train.columns
		importancias = modelo.feature_importances_
		df_importancia = pd.DataFrame({'Feature': colunas, 'Importancia': importancias})
		df_importancia = df_importancia.sort_values(by='Importancia', ascending=False).reset_index(drop=True)
		df_importancia['Importancia (%)'] = (
					df_importancia['Importancia'] / df_importancia['Importancia'].sum() * 100).round(2)
		print(df_importancia)
		return df_importancia

def conf_matrix(y_test, previsoes, class_names):
	plt.figure(figsize=(8, 6))
	sns.heatmap(confusion_matrix(y_test, previsoes), annot=True, fmt='d', cmap='Blues', cbar=False,
				xticklabels=class_names, yticklabels=class_names)
	plt.title('Matriz de Confusão')
	plt.ylabel('Valor Real')
	plt.xlabel('Previsão')
	plt.show()

def auc_roc(y_test, previsoes_proba, x=8, y=6):
	plt.figure(figsize=(x, y))
	if len(previsoes_proba.shape) > 1:
		previsoes_proba = previsoes_proba[:, 1]
	# fpr(taxa falsos positivos) tpr(taxa de verdadeiros positivos) threshold(cutoff de qtns % de certeza eu tenho)
	fpr, tpr, thresholds = roc_curve(y_test, previsoes_proba)
	auc_score = roc_auc_score(y_test, previsoes_proba)

	plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (área = {auc_score:.2f})')
	# linha de referência do chute aleatório
	plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Taxa de Falsos Positivos')
	plt.ylabel('Taxa de Verdadeiros Positivos')
	plt.title('Curva ROC')
	plt.legend(loc="lower right")
	plt.grid(True)
	plt.show()

def verificar_base(X_treino, X_teste, y_treino, y_teste, target_column):
	#se y for series do pandas, tem 1 coluna
	y_treino_cols = 1 if len(y_treino.shape) == 1 else y_treino.shape[1]
	y_teste_cols = 1 if len(y_teste.shape) == 1 else y_teste.shape[1]

	print(f"X_treino: {X_treino.shape[0]} linhas | {X_treino.shape[1]} colunas")
	print(f"y_treino: {y_treino.shape[0]} linhas | {y_treino_cols} colunas")
	print(f"X_teste:  {X_teste.shape[0]} linhas | {X_teste.shape[1]} colunas")
	print(f"y_teste:  {y_teste.shape[0]} linhas | {y_teste_cols} colunas\n")

	if X_treino.shape[0] != y_treino.shape[0]:
		print("Número de linhas diferente entre X_treino e y_treino!")
	if X_teste.shape[0] != y_teste.shape[0]:
		print("Número de linhas diferente entre X_teste e y_teste!\n")

	if list(X_treino.columns) != list(X_teste.columns):
		print("As colunas de X_treino e X_teste são diferentes ou estão em ordem errada!")
	else:
		print("X_treino e X_teste possuem as mesmas colunas.")

	# verifica se o y tem apenas a coluna alvo
	nome_y = y_treino.name if isinstance(y_treino, pd.Series) else y_treino.columns.tolist()
	print(f"Colunas em y_treino: {nome_y}")
	if target_column not in X_treino.columns:
		print("Sucesso: A base X não tem a coluna alvo.\n")
	print(f"Colunas em X_treino: {X_treino.columns.tolist()}")
	print(f"Colunas em X_teste: {X_teste.columns.tolist()}\n")

	# porcentagem de cada classe na base de teste e treino, deve estar balanceado em treino e desbalanceado em teste
	proporcao_teste = y_teste.value_counts(normalize=True) * 100
	proporcao_treino = y_treino.value_counts(normalize=True) * 100
	print(f"proporção base de teste: {proporcao_teste.apply(lambda x: f"{x:.2f}%")}")
	print(f"proporção base de treino: {proporcao_treino.apply(lambda x: f"{x:.2f}%\n")}")

	nulos_treino = X_treino.isnull().sum().sum()
	nulos_teste = X_teste.isnull().sum().sum()
	if nulos_treino > 0 or nulos_teste > 0:
		print(f"Ainda há dados nulos nas bases tratadas! (Treino: {nulos_treino}, Teste: {nulos_teste})")
	else:
		print("Nenhum valor nulo encontrado nas bases.")