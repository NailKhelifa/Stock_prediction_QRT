import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

################################################################################################################
################################## DATA TRANSFORMATION / DATA ANLAYSIS #########################################
################################################################################################################

def transform_data(df_train_merged, df_test):
    
    # Créer une liste de toutes les colonnes de features (RET_i et VOLUME_i)
    ret_columns = [f'RET_{i}' for i in range(1, 21)]
    volume_columns = [f'VOLUME_{i}' for i in range(1, 21)]

    # Grouper par 'STOCK' et appliquer la transformation pour obtenir des tableaux numpy par feature
    transformed_data_train = {}
    transformed_data_test = {}
    for column in ret_columns + volume_columns + ['RET']:
        transformed_data_train[column] = df_train_merged.groupby('STOCK')[column].apply(lambda x: np.array(x))

    # pareil pour test
    for column in ret_columns + volume_columns:
        transformed_data_test[column] = df_test.groupby('STOCK')[column].apply(lambda x: np.array(x))

    # Convertir en DataFrame pour obtenir le format souhaité
    transformed_x_train = pd.DataFrame(transformed_data_train)
    transformed_x_test = pd.DataFrame(transformed_data_test)

    return transformed_x_train, transformed_x_test


def one_stock_returns(ID, transformed_train, transformed_test): 

    """
    ID: integer corresponding to the ID of the stock we want to retrieve in the original dataset.
    transformed_train, transform_test: dataframe as output by transform_data
    """

    X_stock_ID_train = pd.DataFrame(transformed_train.iloc[ID]).transpose()
    X_stock_ID_test = pd.DataFrame(transformed_test.iloc[ID]).transpose()

    # Initialiser un dataframe pour stocker les nouvelles lignes
    transformed_ID_train = pd.DataFrame(columns=X_stock_ID_train.columns, index= np.arange(len(X_stock_ID_train.iloc[0][ID])))
    transformed_ID_test = pd.DataFrame(columns=X_stock_ID_test.columns, index= np.arange(len(X_stock_ID_test.iloc[0][ID])))

    for col in X_stock_ID_train.columns:
        transformed_ID_train[col] = X_stock_ID_train[col][ID]

    for col in X_stock_ID_test.columns:
        transformed_ID_test[col] = X_stock_ID_test[col][ID]

    return transformed_ID_train, transformed_ID_test

def add_features(transformed_df, split=True):

    ret_columns = [f'RET_{i}' for i in range(1, 21)]
    volume_columns = [f'VOLUME_{i}' for i in range(1, 21)]
    
    # Various statistics
    transformed_df['VOL_RET_WEIGHTED'] = transformed_df.apply(lambda row: np.sum(row[f'RET_{i}'] * row[f'VOLUME_{i}'] for i in range(1, 21)) / np.sum(row[f'VOLUME_{i}'] for i in range(1, 21)) if np.sum(row[f'VOLUME_{i}'] for i in range(1, 21)) != 0 else 0, axis=1)
    transformed_df['MEAN_RET_20'] = sum(transformed_df[col] for col in ret_columns) / 20
    transformed_df['MEAN_RET_5'] = sum(transformed_df[col] for col in ret_columns[15:]) / 5
    transformed_df['MEAN_VOL_20'] = sum(transformed_df[col] for col in volume_columns) / 20
    transformed_df['STD_VOL_20'] = np.sqrt(sum((transformed_df[col] -  transformed_df['MEAN_VOL_20'])**2 for col in ret_columns) / 20)
    transformed_df['STD_RET_20'] = np.sqrt(sum((transformed_df[col] -  transformed_df['MEAN_RET_20'])**2 for col in ret_columns) / 20)
    transformed_df['STD_RET_5'] = np.sqrt(sum((transformed_df[col] -  transformed_df['MEAN_RET_5'])**2 for col in ret_columns[15:]) / 5)

    # Cumulative return
    transformed_df['cum_RET_5'] = np.prod([1 + transformed_df[col] for col in ret_columns[:5]], axis=0) - 1
    transformed_df['cum_RET_10'] = np.prod([1 + transformed_df[col] for col in ret_columns[:10]], axis=0) - 1
    transformed_df['cum_RET_20'] = np.prod([1 + transformed_df[col] for col in ret_columns], axis=0) - 1
    # Trends
    transformed_df['RET_Trend_20'] = sum([(transformed_df[ret_columns[i+1]] - transformed_df[ret_columns[i]]) for i in range(len(ret_columns) - 1)])
    transformed_df['RET_slope_5'] = transformed_df['RET_20'] - transformed_df['RET_15']

    if split: 
        class_0_stock_1 = transformed_df[transformed_df['RET'] == 0]
        class_1_stock_1 = transformed_df[transformed_df['RET'] == 1]

    return transformed_df, class_0_stock_1, class_1_stock_1

def PCA_two_classes(features_list, class_0_df, class_1_df):
       # Normalisation des données
       scaler = StandardScaler()
       class_1_scaled = scaler.fit_transform(class_1_df)
       class_0_scaled = scaler.fit_transform(class_0_df)

       pca_class_0 = PCA(n_components=len(features_list))
       pca_class_1 = PCA(n_components=len(features_list))

       _ = pca_class_0.fit_transform(class_1_scaled)
       _ = pca_class_1.fit_transform(class_0_scaled)

       explained_variance_0 = pca_class_0.explained_variance_ratio_
       explained_variance_1 = pca_class_1.explained_variance_ratio_

       _, ax = plt.subplots(figsize=(15, 5))

       # Position des barres pour chaque feature
       x = np.arange(len(features_list))
       width = 0.35  # Largeur des barres pour les mettre côte à côte

       # Barres pour la classe 0
       ax.bar(x - width/2, explained_variance_0, 
              width, color='orange', alpha=0.7, label='Class 0')

       # Barres pour la classe 1
       ax.bar(x + width/2, explained_variance_1, 
              width, color='purple', alpha=0.7, label='Class 1')

       # Titre et labels du graphique
       ax.set_title('Variance Expliquée par Caractéristique Selon la Classe', fontsize=16)
       ax.set_xlabel('Caractéristiques', fontsize=14)
       ax.set_ylabel('Taux de Variance Expliquée', fontsize=14)
       ax.set_xticks(x)
       ax.set_xticklabels(features_list, rotation=45)  
       ax.grid(axis='y')  
       ax.legend()  

       plt.show()

def boxplots(features_list, transformed_df):
    # Tracer les boxplots pour chaque variable en fonction de la menace
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(features_list):
        plt.subplot(4, 4, i+1)  # Deux lignes, deux colonnes
        sns.boxplot(x='RET', y=var, data=transformed_df, palette=['skyblue', 'salmon'])
        plt.title(f'Boxplot de {var} selon la classe')
        plt.ylabel(var)
        plt.xlabel('Return Sign (1 = positive / 0 = negative)')

    plt.tight_layout()
    plt.show()

def histogrames(features_list, transformed_df):
    plt.figure(figsize=(15, 10))
    for i, var in enumerate(features_list):
        plt.subplot(4, 4, i + 1)  # Deux lignes, trois colonnes
        sns.histplot(data=transformed_df, x=var, hue='RET', element='step', stat='density', common_norm=False, 
                    palette=['skyblue', 'salmon'], bins=30, alpha=0.5)
        plt.title(f'Histogramme de {var}')
        plt.ylabel('Densité')
        plt.xlabel(var)

    plt.tight_layout()
    plt.show()