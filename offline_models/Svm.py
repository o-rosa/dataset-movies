import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

class Svm:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
        """
        Inicializa a classe Bayes com os dados de treinamento e teste.
        Defina aqui novos valores para os hiperparametros.
        Parâmetros:
        - X_train (pd.DataFrame): Dados de entrada para treinamento.
        - X_test (pd.DataFrame): Dados de entrada para teste.
        - y_train (pd.DataFrame): Rótulos de saída para treinamento.
        - y_test (pd.DataFrame): Rótulos de saída para teste.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Hyperparameters
        self.k = 5
        self.skf = StratifiedKFold(n_splits=self.k)
        self.param_grid = {
            'C': [10, 25, 50],  
            'gamma': [1, 0.5, 0.1],  
            'kernel': ['poly', 'rbf'] 
        }
        self.scorer = make_scorer(precision_score, average='weighted', zero_division=0)
        self.model = SVC(class_weight='balanced')
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.skf, scoring=self.scorer, n_jobs=-1)
    
    def metricas(self, precisions, recalls, f1_scores, accuracies, k, grid_search, v):
        """
        Calcula e exibe as métricas de desempenho do modelo.
        Parâmetros:
        - precisions (list): Lista de precisões.
        - recalls (list): Lista de revocações.
        - f1_scores (list): Lista de F1-scores.
        - accuracies (list): Lista de acurácias.
        - k (int): Número de folds.
        - grid_search (GridSearchCV): Objeto GridSearchCV treinado.
        - v (str): Nome da variação do modelo.
        Retorna:
        - dict: Dicionário contendo as métricas médias e seus desvios padrão.
        """
        mean_precision = sum(precisions) / k
        mean_recall = sum(recalls) / k
        mean_f1 = sum(f1_scores) / k
        mean_accuracy = sum(accuracies) / k
        # Desvio padrão das métricas através dos folds
        std_precision = np.std(precisions)
        std_recall = np.std(recalls)
        std_f1 = np.std(f1_scores)
        std_accuracy = np.std(accuracies)
        print(f'============= {v} ==============')
        print(f'Média de Precisão: {mean_precision:.4f} ± {std_precision:.4f}')
        print(f'Média de Revocação: {mean_recall:.4f} ± {std_recall:.4f}')
        print(f'Média de F1-Score: {mean_f1:.4f} ± {std_f1:.4f}')
        print(f'Média de Acurácia: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
        print(f'Melhor C encontrado: {grid_search.best_params_["C"]}')
        print(f'Melhor gamma encontrado: {grid_search.best_params_["gamma"]}')
        print(f'Melhor kernel encontrado: {grid_search.best_params_["kernel"]}')
        temp_dict = {
            "mean_f1": round(float(mean_f1), 2),
            "std_f1": round(float(std_f1), 4),
            "mean_accuracy": round(float(mean_accuracy), 2),
            "std_accuracy": round(float(std_accuracy), 4),
            "mean_recall": round(float(mean_recall), 2),
            "std_recall": round(float(std_recall), 4),
            "mean_precision": round(float(mean_precision), 2),
            "std_precision": round(float(std_precision), 4)
        }
        return temp_dict
    
    def features(self, v, only_features=[]):
        """
        Treina e avalia o modelo usando diferentes conjuntos de características.
        Parâmetros:
        - v (str): Nome da variação do modelo.
        - only_features (list): Lista de características a serem usadas. Se vazia, usa todas as características.
        Retorna:
        - tuple: Métricas médias e métricas de teste.
        """
        X_train_select = self.X_train.copy()
        if only_features != []: X_train_select = X_train_select[only_features]

        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        for train_index, val_index in self.skf.split(X_train_select, self.y_train):
            # Separação entre treinamento e validação
            X_train_fold, X_val_fold = X_train_select.iloc[train_index], X_train_select.iloc[val_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
            # Balanceamento com Random Under Sampler (RUS)
            rus = RandomUnderSampler(random_state=42)
            X_train_fold, y_train_fold = rus.fit_resample(X_train_fold, y_train_fold)
            # Melhor modelo da validação e predição
            self.grid_search.fit(X_train_fold, y_train_fold)
            best_model = self.grid_search.best_estimator_
            y_pred = best_model.predict(X_val_fold)
            # Salvando metricas de validação
            precision = precision_score(y_val_fold, y_pred, average='weighted')
            recall = recall_score(y_val_fold, y_pred, average='weighted')
            f1 = f1_score(y_val_fold, y_pred, average='weighted')
            accuracy = accuracy_score(y_val_fold, y_pred)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
        # Treinamento final com todos os dados de treinamento e usando os dados de teste
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(self.X_train, self.y_train)
        self.grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = self.grid_search.best_estimator_
        y_test_pred = best_model.predict(self.X_test)
        # Salvando metricas de teste
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        metrics_media = self.metricas(precisions, recalls, f1_scores, accuracies, self.k, self.grid_search, f"  {v} - Media  das validações ")
        metrics_test = self.metricas([test_precision], [test_recall], [test_f1], [test_accuracy], 1, self.grid_search, f"  {v} - Teste ")
        return metrics_media, metrics_test

    def features_pca(self):
        """
        Treina e avalia o modelo usando PCA para redução de dimensionalidade.
        Retorna:
        - tuple: Métricas médias e métricas de teste.
        """
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        for train_index, val_index in self.skf.split(self.X_train, self.y_train):
            # Separação entre treinamento e validação
            X_train_fold, X_val_fold = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
            y_train_fold, y_val_fold = self.y_train.iloc[train_index], self.y_train.iloc[val_index]
            # Balanceamento com Random Under Sampler (RUS)
            rus = RandomUnderSampler(random_state=42)
            X_train_fold, y_train_fold = rus.fit_resample(X_train_fold, y_train_fold)
            # PCA
            pca = PCA(n_components=10)
            X_train_fold = pca.fit_transform(X_train_fold)
            X_val_fold = pca.transform(X_val_fold)
            scaler_normalize = MinMaxScaler()
            X_train_fold = scaler_normalize.fit_transform(X_train_fold)
            X_val_fold = scaler_normalize.transform(X_val_fold)
            # Melhor modelo da validação e predição
            self.grid_search.fit(X_train_fold, y_train_fold)
            best_model = self.grid_search.best_estimator_
            y_pred = best_model.predict(X_val_fold)
            # Salvando metricas de validação
            precision = precision_score(y_val_fold, y_pred, average='weighted')
            recall = recall_score(y_val_fold, y_pred, average='weighted')
            f1 = f1_score(y_val_fold, y_pred, average='weighted')
            accuracy = accuracy_score(y_val_fold, y_pred)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
        # Treinamento final com todos os dados de treinamento e usando os dados de teste
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(self.X_train, self.y_train)
        pca = PCA(n_components=10)
        X_train_resampled_pca = pca.fit_transform(X_train_resampled)
        X_test_pca = pca.transform(self.X_test)
        scaler_normalize = MinMaxScaler()
        X_train_resampled_pca = scaler_normalize.fit_transform(X_train_resampled_pca)
        X_test_pca = scaler_normalize.transform(X_test_pca)
        self.grid_search.fit(X_train_resampled_pca, y_train_resampled)
        best_model = self.grid_search.best_estimator_
        y_test_pred = best_model.predict(X_test_pca)
        # Salvando metricas de teste
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        metrics_media = self.metricas(precisions, recalls, f1_scores, accuracies, self.k, self.grid_search, " PCA - Media das validações ")
        metrics_test = self.metricas([test_precision], [test_recall], [test_f1], [test_accuracy], 1, self.grid_search, "  PCA - Teste ")
        return metrics_media, metrics_test
  