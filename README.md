- Subdividir o conjunto de treino original subconjuntos de treino e validação, usando kfold com k = 5. Desse modo, é possível obter 5 medidas de desempenho (uma por validação) para cada configuração (método alternativo) avaliada. 
Obs.: Cada método alternativo será composto de um extrator de características + um classificador + um hiperparâmetro para o classificador. 

- Como extrator de características, avaliar o CLS Token de diferentes modelos ViT disponíveis no HugginFace

- Como classificador, avaliar SVM linear, MLP, Random Forest e KNN. 

- Cada classificador deve ser avaliado com 5 hiperparâmetros distintos (por exemplo, KNN com K = 1, 3, 5, 7 e 9). 

- Selecionar a configuração ideal com base nos resultados de validação. 

- Por fim, comparar a configuração ideal (modelo proposto) com alternativas estado-da-arte (por exemplo, MobileNetV2 + FCNN) no conjunto de teste. 

Sugestão: Ler o capítulo 10 do livro "Inteligência Aritifical: uma abordagem de aprendizado de máquina"; utilizar teste de Friedman + pós-teste de Nemenyi para comparar as cinco medidas (ou seja, para comparar as diferentes configurações). Os códigos para fazer os testes de hipótese podem ser encontrados em https://edisciplinas.usp.br/pluginfile.php/4129451/mod_resource/content/1/model_selection_evaluation.pdf


*Observações*:
- Clone o repositorio *https://github.com/iman2693/CTCB* e copie a pasta dataset para a raiz deste projeto
- O arquivo `main.py` serviu como base para desenvolver o projeto. Ele executa um modelo por vez indicado em `vit_model_name = options[0]`
- Os arquivos `analisador.py` e `analisador.ipynb` tem o mesmo script. A diferença entre eles é que um foi *portado* para executar no Google Colab de forma a aproveitar melhor os recursos da plataforma a fim de acelerar a execução do código.
- O arquivo results.txt contem o histório das analises geradas pelos modelos ViT e a comparação entre o script proposto em relação ao modelo do estado-da-arte