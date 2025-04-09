#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import imblearn
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import LocalOutlierFactor

num_intento = "22"

"""
StandardScaler
Model: Random Forest
OverSampling: NO
SimpleImputer
Score:

Using //SequentialFeatureSelector
Added visualization Outliers

Outliers
"""

def GraficoComprobarVar(data_y, path="./visualizaciones/"):
  plt.figure(figsize=(10,8))
  ax = sns.countplot(x='Category', data=data_y)
  for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.2, i.get_height()+3, \
            str(round((i.get_height()), 2)), fontsize=14, color='dimgrey')
  plt.savefig(path+'Categorías-'+num_intento+".png")
  plt.clf()

def MatrizCorrelacion(data_tra, path="./visualizaciones/"):
  correlations = data_tra.corr(numeric_only=False)
  fig, ax = plt.subplots(figsize=(18,18))
  sns.heatmap(correlations, linewidths=0.125, ax=ax)
  plt.savefig(path+"matriz-correlacion-"+num_intento+".png")
  plt.clf()

def ComprobarValPer(data_tra, nombre, path="./visualizaciones/"):
  plt.subplots(figsize=(17,17))
  data_tra.isnull().sum().plot.bar()
  plt.savefig(path+"valores-perdidos-"+num_intento+nombre+".png")
  plt.clf()

def dummies(data):
  X = pd.get_dummies(data[['clave']])
  return X

def norm_one_to_one(df):
    """Normalizamos las columnas de un dataframe al rango [0, 1]"""

    df_norm = df.copy()
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())

    return df_norm

# Dibujar BoxPlot
def BoxPlot(X, k, usadas):
  print("\nGenerando boxplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(16, 16))
  fig.subplots_adjust(wspace=0.4, hspace=0.4)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)
  rango = []

  for i in range(n_var):
    rango.append([X[usadas[i]].min(), X[usadas[i]].max()])

  for i in range(k):
    dat_filt = X.loc[X['Category']==i]
    for j in range(n_var):
      ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i, j])
      ax.set_xlim(rango[j][0], rango[j][1])

  plt.savefig("./visualizaciones/Boxplot-"+num_intento)
  plt.clf()

# ## Leyendo los datos
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_target = train_df.Category
train_df.drop(['Artista', 'Titulo'], axis=1, inplace=True)
train_df.drop(['instrumental'], axis=1, inplace=True) 
test_df.drop(['instrumental'], axis=1, inplace=True)
train_df.drop(['Category'], axis=1, inplace=True)
#MatrizCorrelacion(data_tra=train_df)
#print(train_df.columns)

#X = dummies(train_df)
#print(X.columns)

# ## Viendo nulos y procesándolos
#print(f'Conjunto entrenamiento: {train_df.shape}')
#print(f'El conjunto de entrenamiento contiene: {train_df.isna().sum()}')
#print(f'El conjunto de test contiene: {test_df.isna().sum()}')

## #OUTLIERS  #######################################
# ratio = 1.25
# Q1 = train_df.quantile(0.25)
# Q3 = train_df.quantile(0.75)
# IQR = Q3 - Q1
# outliers = ((train_df < (Q1 - ratio * IQR)) | (train_df > (Q3 + ratio * IQR))).any(axis=1)
# train_df = train_df[~((train_df < (Q1 - 1.5 * IQR)) |(train_df > (Q3 + 4 * IQR))).any(axis=1)]



atribs = [col for col in train_df.columns if col not in ['id']]
train_df[atribs]
print(atribs)
all_df = pd.concat([train_df[atribs], test_df[atribs]], axis=0)
#print(all_df.shape)

# Valores perdidos
nulos_cat = ['clave']
nulos_number = ['popularidad']#, 'instrumental']

Knn_imp = KNNImputer(n_neighbors=4).fit(all_df[nulos_number])
all_df[nulos_number] = pd.DataFrame(Knn_imp.transform(all_df[nulos_number]), columns=all_df[nulos_number].columns)
#print(imputed_X.iloc[:10,:])
#Knn_imp = KNNImputer(n_neighbors=4).fit(train_df[nulos_number]) #No se vuelve a entrenar!!
train_df[nulos_number] = pd.DataFrame(Knn_imp.transform(train_df[nulos_number]), columns=train_df[nulos_number].columns)

# label_encoder_cat = LabelEncoder()
# train_df[nulos_cat] = train_df[nulos_cat].apply(lambda series: pd.Series(label_encoder_cat.fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
# Knn_imp_cat = KNNImputer(n_neighbors=1, weights='distance').fit(train_df[nulos_cat])
# train_df[nulos_cat] = pd.DataFrame(Knn_imp_cat.transform(train_df[nulos_cat]), columns=train_df[nulos_cat].columns)
# print(train_df.head(10))
# label_encoder_cat.inverse_transform(train_df[nulos_cat])
# print(train_df.head(10))

imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_cat.fit(all_df[nulos_cat])
#imputer_number = SimpleImputer(strategy='mean')
#imputer_number.fit(all_df[nulos_number])

# Nulos en ambos conjuntos:
train_df[nulos_cat] = imputer_cat.transform(train_df[nulos_cat])
#train_df[nulos_number] = imputer_number.transform(train_df[nulos_number])
all_df[nulos_cat] = imputer_cat.transform(all_df[nulos_cat])
#all_df[nulos_number] = imputer_number.transform(all_df[nulos_number])

# Comprobamos que no tengo más nulos:
#print("Suma de nulos en train_df tras usar imputer")
#print(f'Conjunto de entrenamiento tras imputar valores nulos: {train_df.shape}')
#print(train_df.isna().sum())
#ComprobarValPer(data_tra=train_df, nombre="ValPer-OneHotEncoder")

# ## Etiquetando los datos categóricos y numéricos
# Etiquetas de datos
col_num = train_df.select_dtypes(include=np.number).columns
col_cat = train_df.select_dtypes(include=object).columns

# ## Visualización OUTLIERS con boxplot
# n_var = len(col_num)
# for i in range(n_var):
#     plt.figure()
#     sns.catplot(train_df, y=col_num[i], kind="box")
#     plt.savefig("./visualizaciones/Boxplot-"+num_intento+"TRAIN_DF-atrib"+str(i))
#     plt.clf()

#ordinalencoder = OrdinalEncoder()
#ordinalencoder.fit(all_df[col_cat])
#train_df[col_cat] = ordinalencoder.transform(train_df[col_cat])
#all_df[col_cat] = ordinalencoder.transform(all_df[col_cat])
targetencoder = LabelEncoder()
train_target = targetencoder.fit_transform(train_target)

# ##ONEHOT ENCODER ######################################################
encoder = OneHotEncoder(sparse=False, dtype=np.int32, drop='if_binary', min_frequency=.05)
all_hot = encoder.fit_transform(all_df[col_cat])
new_columns_all = encoder.get_feature_names_out()
all_hot = pd.DataFrame(all_hot, columns=new_columns_all)
#print(all_hot.columns)
#print(all_df.columns)

all_df.reset_index(inplace=True, drop=True)
all_df = pd.concat([all_hot, all_df], axis=1)
all_df.drop(col_cat, axis=1, inplace=True)
#all_df.drop(['clave_nan'], axis=1, inplace=True)
#print(all_df.columns)
#print(f'All_df columns tras cocatenar con all hot{all_df.columns}')

# Por defecto, matriz sparse
#encoder = OneHotEncoder(sparse=False, dtype=np.int32, drop='if_binary', min_frequency=.05)
train_hot = encoder.transform(train_df[col_cat])
#print("OneHotEncoder Train:")
#print(train_hot.head(5))

#Conversión a DataFrame
new_columns = encoder.get_feature_names_out()
#print(new_columns)
train_hot = pd.DataFrame(train_hot, columns=new_columns)
# Copiar el resto de atributos
#print(train_df.columns)
print(f'Tamaño de train_hot {train_hot.shape} y de train_df {train_df.shape}')
print(f'Tamaño train_target TRAIN {train_df.shape}, target {train_target.shape}')
train_df = pd.concat([train_hot, train_df], axis=1) #Le pego las columnas en binario de cada valor de clave, modo_menor
#print("Tras unir traindf tras el hot:")
#print(train_df.columns)
train_df.drop(['clave', 'modo'], axis=1, inplace=True)
#train_df.drop(['clave_nan'], axis=1, inplace=True)
print(train_df.columns)



# clf = LocalOutlierFactor(n_neighbors=5)
# train_df_out = train_df.copy()
# train_df_out["outlier"] = np.abs(clf.fit_predict(train_df) - -1) <= 1e-3
# train_df = train_df_out[~train_df_out["outlier"]]
# print(train_df_out.head(3))
# train_df.drop(["outlier"], axis=1, inplace=True)
#
# ## Cambiando escala los datos numéricos
# #scaler = MinMaxScaler().fit(all_df)
scaler = StandardScaler().fit(all_df)
train_df[train_df.columns] = scaler.transform(train_df)


#print(f'Tamaño train_target {train_df.shape}')

# ##SMOTE para desbalanceo de clases ########################################
# counter = Counter(train_target)
# print(f'Cuenta de train_target (Categorías) antes de Smote{counter}')

#oversample = SVMSMOTE(random_state=1234)
#train_df, train_target= oversample.fit_resample(train_df, train_target)
#print(f'Shape de traindfd smote {train_df_smote.shape}')
#print(f'Shape de train_target {train_target_smote.shape}')

#counter = Counter(train_target_smote)
#print(f'Cuenta de train_target (Categorías) después de Smote{counter}')


# ## Aplicando el modelo
#model = DecisionTreeClassifier(random_state=1234)
#model = MLPClassifier(alpha=1, random_state=1234, max_iter=400)
#model = RandomForestClassifier(random_state=1234, n_estimators=300, n_jobs=6, bootstrap=True, max_depth=25)
#model = LogisticRegressionCV(cv = StratifiedKFold(shuffle = True, random_state = 1234), n_jobs=5, max_iter = 1000)
model = GradientBoostingClassifier(random_state=1234, max_depth=3, criterion="friedman_mse", n_estimators=100)

#model = Pipeline([('resample', oversample), ('model', model)])
scores = cross_val_score(model, train_df, train_target, cv=5, n_jobs=5)

# La puntuación final sería la media de las cinco.
accuracy = scores.mean()
print(accuracy)

# ## Prediciendo sobre el conjunto de test
model.fit(train_df, train_target)

# Preprocesar el conjunto de entrenamiento:
# Quito los nulos
test_df[nulos_cat] = imputer_cat.transform(test_df[nulos_cat])
#test_df[nulos_number] = imputer_number.transform(test_df[nulos_number])
#Knn_imp_test = KNNImputer(n_neighbors=4).fit(test_df[nulos_number])
test_df[nulos_number] = pd.DataFrame(Knn_imp.transform(test_df[nulos_number]), columns=test_df[nulos_number].columns)

# Etiqueto:
#test_df[col_cat] = ordinalencoder.transform(test_df[col_cat])
#encoder = OneHotEncoder(sparse=False, dtype=np.int32, drop='if_binary')
test_hot = encoder.transform(test_df[col_cat])
new_columns_test = encoder.get_feature_names_out()
test_hot = pd.DataFrame(test_hot, columns=new_columns_test)
test_df = pd.concat([test_hot, test_df], axis=1)
test_df.drop(col_cat, axis=1, inplace=True)
print(test_df.columns)
#test_df.drop(['clave_nan'], axis=1, inplace=True)

########################################################################
# Escalo:
test_df[train_df.columns] = scaler.transform(test_df[train_df.columns])

# Aplico el modelo entrenado (ignorando id):
predictions = model.predict(test_df.drop(['id'], axis=1))
predictions = targetencoder.inverse_transform(predictions)
print(predictions)

# Crear el fichero
submission = pd.DataFrame({'id': test_df.id, 'Expected': predictions})

# Tamaño adecuado:
print(submission.shape)

# Crear el fichero del modelo:
submission.to_csv("simple_dc"+num_intento+".csv", index=False)
