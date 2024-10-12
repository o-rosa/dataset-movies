import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

class FeatureSelection:
    def __init__(self, X_train, y_train) -> None:
        """
        Inicializa a classe Selecao com os dados de treinamento.

        Parâmetros:
        - X_train (pd.DataFrame): Dados de entrada para treinamento.
        - y_train (pd.Series): Rótulos de saída para treinamento.
        """
        self.X_train = X_train
        self.y_train = y_train

    def func_SelectKBest(self):
        """
        Seleciona as 5 melhores características usando o método SelectKBest com o teste qui-quadrado (chi2).

        Retorna:
        - list: Lista com os nomes das 5 melhores características selecionadas.
        """
        selector = SelectKBest(chi2, k=5)
        selector.fit(self.X_train, self.y_train)
        return list(selector.get_feature_names_out())

    def func_RandomForest(self):
        """
        Seleciona as características importantes usando um modelo RandomForestClassifier.

        Retorna:
        - list: Lista com os nomes das características importantes selecionadas.
        """
        model = RandomForestClassifier(criterion='entropy', random_state=42)
        model.fit(self.X_train, self.y_train)

        # Obter a importância das features
        importances = model.feature_importances_

        threshold = 0.03

        # Selecionar as features importantes
        important_indices = np.where(importances > threshold)[0]
        important_features = self.X_train.columns[important_indices]

        return list(important_features)
