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
graph LR
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

    classDef dataset fill:#933,stroke:#333,stroke-width:2px;
    classDef process fill:#c3c,stroke:#333,stroke-width:2px;
    classDef decision fill:#339,stroke:#333,stroke-width:2px;
    classDef output fill:#99f,stroke:#333,stroke-width:2px;

    class A,B,C,E dataset;
    class D,F,G,H,K,L,M,N,O process;
    class I,J decision;
```

```mermaid
flowchart LR
    subgraph DS["Preparando Dataset"]
        DS0[("baixa dataset")]-->
        TT0["configura transformação"]-->DS1
        
        DS1{"possui train/test?"}-- Sim -->
        DS2[carrega train<br>carrega test]
        DS3[estratifica:<li>80% train<li>20% test]
        DS1-- Nao -->DS3
        TT1[otimiza imagens para ViT]
        DS2 & DS3 --> TT1-->
        DS4[X_train_full, y_train_full, X_test_full, y_test_full]
    end

    subgraph FE["Instanciando Extratores"]
        FE0[inicializa extratores]-->
        FE1{carrega extrator}
        FE2[extrair caracteristicas de X_train_full, y_train_full, X_test_full, y_test_full]
        FE1--> FE2 -->FE1
    end

    subgraph KF[Validação Cruzada]
        KF0[instancia classificadores]-->
        KF1{fold < 5}-->
        KF2{extrator}-->
        KF3{inicia classificador}-->
        KF4{define hiperparametro}-->
        KF5[treina no fold]
        KF6[valida no fold]-->
        KF7[calcula metricas]

        KF2-->KF1
        KF3-->KF2
        KF4-->KF3
        KF5-->KF6-->KF4
         
    end

    subgraph OR[Otimização e Resultados]
        OR0[selecionar melhores Hiperparâmetros por Extrator, Classificador]-->
        OR1[Treinar Classificador Final com Melhores Hiperparâmetros em X_train_full]-->
        OR2[Avaliar no Conjunto de Teste X_test_full, y_test_full]-->
        OR3[Gerar Métricas Finais, Matriz de Confusão, TP/TN/FP/FN]-->
        OR4[dentificar Melhor Modelo Geral]-->
        OR5[(Salvar Relatório Detalhado)]
    end

    DS==>FE==>KF==>OR
```