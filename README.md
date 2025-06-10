- Subdividir o conjunto de treino original subconjuntos de treino e validação, usando kfold com k = 5. Desse modo, é possível obter 5 medidas de desempenho (uma por validação) para cada configuração (método alternativo) avaliada. 
Obs.: Cada método alternativo será composto de um extrator de características + um classificador + um hiperparâmetro para o classificador. 

- Como extrator de características, avaliar o CLS Token de diferentes modelos ViT disponíveis no HugginFace

- Como classificador, avaliar SVM linear, MLP, Random Forest e KNN. 

- Cada classificador deve ser avaliado com 5 hiperparâmetros distintos (por exemplo, KNN com K = 1, 3, 5, 7 e 9). 

- Selecionar a configuração ideal com base nos resultados de validação. 

```mermaid
---
config:
  theme: default
  look: neo
  layout: dagre
---
flowchart TD
    A(("Start")) --> B["Setup Device"]
    B --> C["Define Hyperparameters"]
    C --> D["Define Image Transformations"]
    D --> E[("Download Dataset")]
    E --> F{"Check for Test Directory?"}
    F -- No --> G["Split Train Directory into Train 0.8 Test 0.2"]
    F -- Yes --> H["Load Train and Test Datasets"]
    G --> I["Initialize DataLoaders for Full Train and Test"]
    H --> I
    I --> J["Initialize Feature Extractors - ViT Models"]
    J --> K["Initialize Classifiers - Sklearn"]
    K --> L["Extract Features from Full Train and Test Datasets"]
    L --> M[("Store Extracted Features")]
    M --> N["Start K-Fold Cross-Validation on Training Data"]
    N --> O{"For Each Extractor"}
    O --> P{"For Each Fold"} & Y["Analyze K-Fold Results"]
    P --> Q["Split Data into Train/Validation for Fold"] 
    Q --> R{"For Each Classifier"} 
    R --> S{"For Each Hyperparameter Combination"} & P
    S --> T["Initialize Classifier with Params"] & R
    T --> U["Train Classifier on Fold Train Data"]
    U --> V["Predict on Fold Validation Data"]
    V --> W["Evaluate Metrics"]
    W --> X[("Store K-Fold Results")]
    X --> S
    Y --> Z["Select Best Hyperparameters per Classifier/Extractor"]
    Z --> AA["Start Final Evaluation on Test Set"]
    AA --> BB{"For Each Extractor"}
    BB --> CC{"For Each Classifier"} & LL["Display All Final Test Results"]
    CC --> DD["Retrieve Best Params from K-Fold"] & BB
    DD --> EE{"Are Best Params Valid?"}
    EE -- No --> FF["Skip Final Test for this Classifier"] 
    EE -- Yes --> GG["Initialize Classifier with Best Params"]
    GG --> HH["Train Classifier on Full Train Data"]
    HH --> II["Predict on Full Test Data"]
    II --> JJ["Evaluate Test Set Metrics"]
    JJ --> KK[("Store Final Test Results")]
    KK --> CC
    LL --> MM["Identify and Display Overall Best Result"]
    MM --> NNN(("End"))
```

```mermaid
graph TD
    A[Dataset de Imagens CTCB] --> B{Carregar e Dividir Dataset};
    B --> C[Transformações de Imagem];
    C --> D{Extratores ViT DINO, ViT-Base, ViT-Large};
    D --> E[Extrair Features X_train_full, y_train_full, X_test_full, y_test_full];

    subgraph "Otimização e Treinamento"
        E --> F{Validação Cruzada K-Fold em X_train_full};
        F -- Para cada Extrator, Classificador, Hiperparâmetro --> G[Treinar Classificador no Fold de Treino];
        G --> H[Validar no Fold de Validação];
        H --> I[Calcular Métricas Acurácia Balanceada];
        I --> J{Selecionar Melhores Hiperparâmetros por Extrator, Classificador};
    end

    J --> K[Treinar Classificador Final com Melhores Hiperparâmetros em X_train_full completo];
    K --> L[Avaliar no Conjunto de Teste X_test_full, y_test_full];
    L --> M[Gerar Métricas Finais, Matriz de Confusão, TP/TN/FP/FN];
    M --> N[Identificar Melhor Modelo Geral];
    N --> O[Salvar Relatório Detalhado];

    classDef dataset fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#9cf,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef output fill:#9f9,stroke:#333,stroke-width:2px;

    class A,B,C,E dataset;
    class D,F,G,H,K,L,M,N,O process;
    class I,J decision;
```