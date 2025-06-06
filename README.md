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

graph TD
    A[Start] --> B{Setup Device};
    B --> C[Define Hyperparameters];
    C --> D[Define Image Transformations];
    D --> E[Download Dataset];
    E --> F{Check for Test Directory?};
    F -- No --> G[Split Train Directory into Train/Test];
    F -- Yes --> H[Load Train and Test Datasets];
    G --> I[Initialize DataLoaders (Full Train & Test)];
    H --> I;
    I --> J[Initialize Feature Extractors (ViT Models)];
    J --> K[Initialize Classifiers (Sklearn)];

    K --> L[Extract Features from Full Train and Test Datersets];
    L --> M{Store Extracted Features};

    M --> N[Start K-Fold Cross-Validation on Training Data];
    N --> O{For Each Extractor};
    O --> P{For Each Fold};
    P --> Q[Split Data into Train/Validation for Fold];
    Q --> R{For Each Classifier};
    R --> S{For Each Hyperparameter Combination};
    S --> T[Initialize Classifier with Params];
    T --> U[Train Classifier on Fold Train Data];
    U --> V[Predict on Fold Validation Data];
    V --> W[Evaluate Metrics];
    W --> X[Store K-Fold Results];
    X --> S;
    S --> R;
    R --> Q;
    Q --> P;
    P --> O;
    O --> Y[Analyze K-Fold Results];
    Y --> Z[Select Best Hyperparameters per Classifier/Extractor];

    Z --> AA[Start Final Evaluation on Test Set];
    AA --> BB{For Each Extractor};
    BB --> CC{For Each Classifier};
    CC --> DD[Retrieve Best Params from K-Fold];
    DD --> EE{Are Best Params Valid?};
    EE -- No --> FF[Skip Final Test for this Classifier];
    EE -- Yes --> GG[Initialize Classifier with Best Params];
    GG --> HH[Train Classifier on Full Train Data];
    HH --> II[Predict on Full Test Data];
    II --> JJ[Evaluate Test Set Metrics];
    JJ --> KK[Store Final Test Results];
    KK --> CC;
    CC --> BB;
    BB --> LL[Display All Final Test Results];
    LL --> MM[Identify and Display Overall Best Result];
    MM --> NNN[End];
