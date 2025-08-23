## Classificadores 

Este repositorio apresenta os resultados da pesquisa de um método de classificação usando Vision Transformers como extratores de características em conjunto com os classificadores classicos como SVM, KNN, MLP e RandomForest.

Os Vision Transformers analisados foram:
- facebook/dino-vitb8.
- google/vit-base-patch16-224.
- google/vit-large-patch16-224,

Como referência, foram usadas as imagens e os resultados de [kianian](https://github.com/iman2693/CTCB).

Os scripts gerados são do tipo Jupyter Notebook e foram executados no Google Colab.

- classificadores_metricas_ponderadas.ipynb apresenta o metodo de classificação usando o conjunto de imagens originais.
- classificadores_metricas_ponderadas_preprocessado.ipynb apresenta o metodo de classificação usando o conjunto de imagens preprocessadas, tendo suas dimensões rearranjadas a fim de deixar altura e largura proporcionais gerando uma imagem quadarada.

### Análise dos Resultados de Avaliação do Modelo

A seguir, apresento as tabelas com os resultados de performance dos modelos de classificação de imagens, baseadas nos arquivos `evaluation_results.txt` e `evaluation_results_preprocessed.txt`. As tabelas sumarizam a performance de diferentes combinações de extratores de características, classificadores e seus respectivos hiperparâmetros.

As métricas de avaliação utilizadas foram:

*   **Acurácia Média (Ponderada pela Amostra):** Representa a proporção de predições corretas em relação ao total de amostras.
*   **Precisão (Macro):** Média da precisão de cada classe, sem levar em conta o desbalanceamento entre elas. A precisão indica a proporção de verdadeiros positivos dentre todas as predições positivas.
*   **Recall (Macro):** Média do recall de cada classe, também sem considerar o desbalanceamento. O recall (ou revocação) mede a proporção de verdadeiros positivos que foram corretamente identificados.
*   **F1-Score (Macro):** Média harmônica entre precisão e recall, calculada para cada classe e depois feita a média. É uma métrica útil quando existe um desequilíbrio entre as classes.

---

### Resultados do Arquivo: `evaluation_results.txt`

Esta tabela resume o desempenho de cada combinação de extrator de características e classificador no conjunto de teste, utilizando os melhores hiperparâmetros encontrados durante a validação cruzada.

| Extractor | Classifier | Hyperparameters | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DINO | SVM | `{'C': 10}` | 0.8234 | 0.8217 | 0.8234 | 0.8193 |
| DINO | MLP | `{'hidden_layer_sizes': (100, 100), ...}` | 0.8097 | 0.8069 | 0.8097 | 0.8049 |
| DINO | RandomForest | `{'n_estimators': 50, 'random_state': 42}` | 0.6598 | 0.7114 | 0.6598 | 0.6797 |
| **DINO** | **KNN** | **`{'n_neighbors': 1}`** | **0.9134** | **0.9316** | **0.9134** | **0.9135** |
| ViT-Base | SVM | `{'C': 100}` | 0.7945 | 0.8883 | 0.7945 | 0.8240 |
| ViT-Base | MLP | `{'hidden_layer_sizes': (200, 200), ...}` | 0.8372 | 0.7984 | 0.8372 | 0.8064 |
| ViT-Base | RandomForest | `{'n_estimators': 200, 'random_state': 42}` | 0.5699 | 0.5980 | 0.5699 | 0.5788 |
| ViT-Base | KNN | `{'n_neighbors': 1}` | 0.7981 | 0.8229 | 0.7981 | 0.8043 |
| ViT-Large | SVM | `{'C': 100}` | 0.7540 | 0.7644 | 0.7540 | 0.7547 |
| ViT-Large | MLP | `{'hidden_layer_sizes': (200, 200), ...}` | 0.7902 | 0.8248 | 0.7902 | 0.7906 |
| ViT-Large | RandomForest | `{'n_estimators': 100, 'random_state': 42}` | 0.5611 | 0.5993 | 0.5611 | 0.5698 |
| ViT-Large | KNN | `{'n_neighbors': 1}` | 0.8041 | 0.8675 | 0.8041 | 0.8218 |

---

### Resultados do Arquivo: `evaluation_results_preprocessed.txt`

Esta tabela resume os resultados do conjunto de teste para os modelos treinados com dados pré-processados.

| Extractor | Classifier | Hyperparameters | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DINO | SVM | `{'C': 10}` | 0.8354 | 0.7999 | 0.8354 | 0.8153 |
| DINO | MLP | `{'hidden_layer_sizes': (50,), ...}` | 0.7986 | 0.7539 | 0.7986 | 0.7720 |
| DINO | RandomForest | `{'n_estimators': 100, 'random_state': 42}` | 0.6890 | 0.7366 | 0.6890 | 0.7077 |
| **DINO** | **KNN** | **`{'n_neighbors': 1}`** | **0.8962** | **0.9460** | **0.8962** | **0.9126** |
| ViT-Base | SVM | `{'C': 10}` | 0.7554 | 0.7758 | 0.7554 | 0.7612 |
| ViT-Base | MLP | `{'hidden_layer_sizes': (100, 100), ...}` | 0.8329 | 0.8991 | 0.8329 | 0.8530 |
| ViT-Base | RandomForest | `{'n_estimators': 200, 'random_state': 42}` | 0.6443 | 0.7005 | 0.6443 | 0.6655 |
| ViT-Base | KNN | `{'n_neighbors': 1}` | 0.7392 | 0.7082 | 0.7392 | 0.7168 |
| ViT-Large | SVM | `{'C': 10}` | 0.8427 | 0.9672 | 0.8427 | 0.8846 |
| ViT-Large | MLP | `{'hidden_layer_sizes': (100, 100), ...}` | 0.8502 | 0.8885 | 0.8502 | 0.8600 |
| ViT-Large | RandomForest | `{'n_estimators': 100, 'random_state': 42}` | 0.6149 | 0.7122 | 0.6149 | 0.6395 |
| ViT-Large | KNN | `{'n_neighbors': 1}` | 0.8282 | 0.9042 | 0.8282 | 0.8534 |
| ViT-Base | MLP | {'hidden_layer_sizes': (100, 100), 'max_iter': 200, 'random_state': 42} | 0.8483 | 0.8287 | 0.8483 | 0.8301 |
| ViT-Base | KNN | {'n_neighbors': 1} | 0.7901 | 0.7781 | 0.7901 | 0.7671 |
| DINO | RandomForest | {'n_estimators': 100, 'random_state': 42} | 0.7304 | 0.8576 | 0.7304 | 0.7586 |
| ViT-Base | RandomForest | {'n_estimators': 200, 'random_state': 42} | 0.5871 | 0.7060 | 0.5871 | 0.6128 |
| ViT-Large | RandomForest | {'n_estimators': 100, 'random_state': 42} | 0.5782 | 0.7179 | 0.5782 | 0.6071 |
