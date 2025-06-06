- Subdividir o conjunto de treino original subconjuntos de treino e validação, usando kfold com k = 5. Desse modo, é possível obter 5 medidas de desempenho (uma por validação) para cada configuração (método alternativo) avaliada. 
Obs.: Cada método alternativo será composto de um extrator de características + um classificador + um hiperparâmetro para o classificador. 

- Como extrator de características, avaliar o CLS Token de diferentes modelos ViT disponíveis no HugginFace

- Como classificador, avaliar SVM linear, MLP, Random Forest e KNN. 

- Cada classificador deve ser avaliado com 5 hiperparâmetros distintos (por exemplo, KNN com K = 1, 3, 5, 7 e 9). 

- Selecionar a configuração ideal com base nos resultados de validação. 

```mermaid
flowchart TD
    A(("Start")) --> B["Setup Device"]
    B --> C["Define Hyperparameters"]
    C --> D["Define Image Transformations"]
    D --> E[("Download Dataset")]
    E --> F{"Check for Test Directory?"}
    F -- No --> G>"Split Train Directory into Train 0.8 Test 0.2"]
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
    P --> Q["Split Data into Train/Validation for Fold"] & O
    Q --> R{"For Each Classifier"} & P
    R --> S{"For Each Hyperparameter Combination"} & Q
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
