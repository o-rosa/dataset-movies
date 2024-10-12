import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Graphics:
    def __init__(self) -> None:
        pass
    def modelo_unico(self, json, metrica, interval=[0.6, 0.9]):
        """
        Gera um gráfico de barras para uma única métrica de diferentes métodos de redução de dimensionalidade.
        Parâmetros:
        - json (dict): Dicionário contendo os métodos e suas respectivas métricas.
        - metrica (str): Nome da métrica a ser plotada.
        - interval (list): Intervalo do eixo y para o gráfico.
        Retorna:
        - None: Exibe o gráfico gerado.
        """
        methods = list(json.keys())
        mean_f1 = [json[method][f'mean_{metrica}'] for method in methods]
        std_f1 = [json[method][f'std_{metrica}'] for method in methods]
        # Configuração do gráfico
        x_pos = np.arange(len(methods))
        fig, ax = plt.subplots()
        # Criando o gráfico de barras
        ax.bar(x_pos, mean_f1, yerr=std_f1, align='center', alpha=0.7, ecolor='black', capsize=10)
        ax.set_ylabel(metrica)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_title(f'{metrica} para Cada Redução de Dimensionalidade')
        ax.yaxis.grid(True)
        ax.set_ylim(interval)
        # Exibindo o gráfico
        plt.tight_layout()
        plt.show()
    
    def n_modelos(self, json, metrica, interval=[0.6, 0.9], labels=['all_features', 'pca', 'SelectKBest', 'RandomForest'], intra_group_spacing=0.02, inter_group_spacing=0.2):
        """
        Gera um gráfico de barras comparando várias métricas de diferentes modelos e métodos de redução de dimensionalidade.
        Parâmetros:
        - json (dict): Dicionário contendo os modelos, métodos e suas respectivas métricas.
        - metrica (str): Nome da métrica a ser plotada.
        - interval (list): Intervalo do eixo y para o gráfico.
        - labels (list): Lista de rótulos para os métodos de redução de dimensionalidade.
        - intra_group_spacing (float): Espaçamento entre as barras dentro de um grupo.
        - inter_group_spacing (float): Espaçamento entre os grupos de barras.
        Retorna:
        - None: Exibe o gráfico gerado.
        """
        n_models = len(json)
        x = np.arange(len(labels))  # localização dos rótulos
        total_width = 1 - inter_group_spacing  # largura total disponível para as barras
        width = (total_width - intra_group_spacing * (n_models - 1)) / n_models  # largura das barras
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (model_name, model_data) in enumerate(json.items()):
            means = [model_data[feature][f'mean_{metrica}'] for feature in model_data]
            stds = [model_data[feature][f'std_{metrica}'] for feature in model_data]
            rects = ax.bar(x - total_width / 2 + i * (width + intra_group_spacing), means, width, yerr=stds,
                           label=model_name.replace('_', ' ').title(), capsize=5, ecolor='gray')
            # Adicionando os valores acima das barras
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height*100:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        # Adicionando os rótulos, título e customizando o gráfico
        ax.set_ylabel('%')
        ax.set_title(f'{metrica} Média')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(interval)
        # Centralizando os rótulos dos grupos
        group_centers = x - total_width / 2 + (n_models - 1) * (width + intra_group_spacing) / 2
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels)
        fig.tight_layout()
        # Salvando o gráfico
        plt.savefig(f'./plots/{metrica}_comparison.png')
        plt.show()

    def correlation_matrix(self, df):
        correlation_matrix = df.corr()
        # Criando o heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, fmt='.2f', annot_kws={"size": 6})
        plt.title('Heatmap de Correlações')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.show()