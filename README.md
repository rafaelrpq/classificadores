Este repositorio apresenta os resultados da pesquisa de um método de classificação usando Vision Transformers como extratores de características em conjunto com os classificadores classicos como SVM, KNN, MLP e RandomForest.

Os Vision Transformers analisados foram:
- facebook/dino-vitb8
- google/vit-base-patch16-224
- google/vit-large-patch16-224

Como referência, foram usadas as imagens e os resultados de [kianian](https://github.com/iman2693/CTCB)

Os scripts gerados são do tipo Jupyter Notebook e foram executados no Google Colab.

- classificadores_metricas_ponderadas.ipynb apresenta o metodo de classificação usando o conjunto de imagens originais.
- classificadores_metricas_ponderadas_preprocessado.ipynb apresenta o metodo de classificação usando o conjunto de imagens preprocessadas, tendo suas dimensões rearranjadas a fim de deixar altura e largura proporcionais gerando uma imagem quadarada.
