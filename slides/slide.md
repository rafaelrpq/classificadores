---
marp: true
theme: default
class: invert

style : |
    html {
        font-size: 62.5%;
    }
    section {
        font-size: 1.4rem;
        background-image: linear-gradient(#334, #667);
        text-shadow: .1rem .1rem .2rem #000;
    }
    
    h1, h2, h3, h4, h5, h6 {
        background: -webkit-linear-gradient(#37d, #dda); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: bold;
        text-shadow: none;
        filter:drop-shadow(.2rem .2rem .2rem #000) 
    }

    code {
        text-shadow: none;
    }

---
<!-- **Slide 1: Título** -->

# Vision Transformers para a Classificação de Cianobactérias

---

<!-- **Slide 2: Visão Geral / Agenda** -->

## O Que Vamos Ver?

1.  **Configuração Inicial:** Bibliotecas, dispositivo, hiperparâmetros.
2.  **Dados:** Carregamento, transformações e divisão do dataset.
3.  **Extração de Características:** Usando modelos Vision Transformer (ViT).
4.  **Classificadores:** Modelos tradicionais da `sklearn`.
5.  **Métricas:** Como avaliamos o desempenho.
6.  **Fluxo de Trabalho Principal:**
    *   Extração inicial de características.
    *   Validação Cruzada K-Fold (otimização de hiperparâmetros).
    *   Seleção dos melhores modelos.
    *   Avaliação final no conjunto de teste.
7.  **Resultados:** Exibição e salvamento.

---

<!-- **Slide 3: Configuração Inicial** -->

## Preparando o Ambiente

*   **Importações Essenciais:** PyTorch, TorchVision, Scikit-learn, Transformers, etc.
*   **Dispositivo (CPU/GPU):**
    ```python
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    print (f"Usando o dispositivo: {device}")
    ```
*   **Hiperparâmetros para Classificadores:**
    ```python
    hyperparameters = {
        "SVC": [{"C": c} for c in [0.1, 1, 10, 100, 1000]],
        "MLPClassifier": [{"hidden_layer_sizes": hls, ...} for hls in [...] ],
        # ... e outros
    }
    ```
*   **Transformações de Imagem:**
    ```python
    transform = transforms.Compose ([
        transforms.Resize ((224, 224)),
        transforms.ToTensor (),
        transforms.Normalize (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    ```

---

<!-- **Slide 4: Carregamento e Preparação do Dataset** -->

## O Dataset CTCB

*   **Clonagem do Repositório:**
    ```python
    !git clone https://github.com/iman2693/CTCB.git > /dev/null 2>&1
    path = "/content/CTCB"
    train_data_dir = os.path.join (path,"dataset/Train")
    test_data_dir = os.path.join (path,"dataset/Test")
    ```
---
## O Dataset CTCB

*   **Lógica de Carregamento/Divisão:**
    *   Se `test_data_dir` não existe, divide `train_data_dir` (80/20 estratificado).
    ```python
    if not os.path.exists(test_data_dir):
        full_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
        targets = full_dataset.targets
        train_indices, test_indices = train_test_split(..., stratify=targets)
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
    else:
        train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    ```
---

## O Dataset CTCB

*   **Informações do Dataset:**
    ```python
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    ```

---

<!-- **Slide 5: Extração de Características com ViTs** -->

## Classe `FeatureExtractor` 

*   **Objetivo:** Extrair vetores de características das imagens usando modelos ViT pré-treinados.
*   **Inicialização:**
    ```python
    class FeatureExtractor :
        def __init__ (self, vit_model_name) :
            self.model = AutoModel.from_pretrained(vit_model_name).to(device)
            self.model.eval () # Modo de avaliação
    ```
*   **Extração:**
    ```python
    def extract_features (self, dataloader):
        with torch.no_grad (): # Sem cálculo de gradientes
            for inputs, targets in tqdm(dataloader, desc="Extracting Features"):
                outputs = self.model (inputs)
                # Ex: Usando o token [CLS]
                cls_tokens = outputs.last_hidden_state[:, 0, :]
                # ...
        return np.vstack (features), np.hstack (labels)
    ```
---

## Classe `FeatureExtractor` 
*   **Modelos Usados:**
    ```python
    vit_extractors = {
        "DINO": FeatureExtractor("facebook/dino-vitb8"),
        "ViT-Base": FeatureExtractor("google/vit-base-patch16-224"),
        # ... e outros
    }
    ```

---

<!-- **Slide 6: Classificadores Tradicionais** -->

## Modelos `sklearn`

*   Utilizamos classificadores clássicos para aprender sobre as características extraídas.
    ```python
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    # ... e outros

    classifiers_map = {
        'SVM': SVC,
        'MLP': MLPClassifier,
        'RandomForest': RandomForestClassifier,
        'KNN': KNeighborsClassifier,
    }
    ```
*   Os hiperparâmetros para cada um são testados via K-Fold (veremos adiante).

---

<!-- **Slide 7: Métricas de Avaliação** -->

## Medindo o Desempenho (`evaluate_metrics`)

*   **Função Chave:** Calcula diversas métricas importantes.
    ```python
    def evaluate_metrics(y_true, y_pred, class_names=None, num_classes=None):
        acc = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # ... precisão/recall/f1 ponderados ...
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        
        # ... TP, TN, FP, FN por classe ...
        return { "accuracy": ..., "precision_macro": ..., ... }
    ```
*   **Métricas Principais:** Acurácia, Acurácia Balanceada, Precisão, Recall, F1-Score (Macro e Ponderado), Matriz de Confusão, TP/TN/FP/FN por classe.

---

<!-- **Slide 8: Fluxo de Trabalho - Etapa 1** -->

## 1. Extração Inicial de Características (Global)

*   **Objetivo:** Extrair características de **TODOS** os dados de treino e teste **UMA VEZ** por extrator ViT.
*   Isso economiza processamento, pois a extração é custosa.
    ```python
    feature_data = {}
    for extractor_name, extractor in vit_extractors.items():
        # DataLoaders para datasets completos
        full_train_loader_for_extraction = DataLoader(train_dataset, ...)
        test_loader_for_extraction = DataLoader(test_dataset, ...)

        X_train_full, y_train_full = extractor.extract_features(full_train_loader_for_extraction)
        X_test_full, y_test_full = extractor.extract_features(test_loader_for_extraction)

        feature_data[extractor_name] = {
            'X_train_full': X_train_full, 'y_train_full': y_train_full,
            'X_test_full': X_test_full, 'y_test_full': y_test_full,
        }
    ```

---

<!-- **Slide 9: Fluxo de Trabalho - Etapa 2** -->

## 2. Validação Cruzada K-Fold (nos dados de TREINO)

*   **Objetivo:** Encontrar os melhores hiperparâmetros para cada combinação (Extrator ViT + Classificador).
*   Usa as características `X_train_full`, `y_train_full` extraídas anteriormente.
    ```python
    k_folds = 5
    kf = KFold (n_splits=k_folds, shuffle=True, random_state=42)
    kfold_results = []

    for extractor_name, data in feature_data.items():
        X_train_full = data['X_train_full'] # Características de treino
        # ...
        for fold_idx, (train_indices_fold, val_indices_fold) in enumerate(kf.split(X_train_full)):
            X_train_fold, y_train_fold = ...
            X_val_fold, y_val_fold = ...
            for clf_name, clf_class in classifiers_map.items():
                for params in hyperparameters.get(clf_class.__name__, []):
                    clf = clf_class(**params).fit(X_train_fold, y_train_fold)
                    y_pred_fold = clf.predict(X_val_fold)
                    metrics = evaluate_metrics(y_val_fold, y_pred_fold, ...)
                    kfold_results.append({...})
    ```

---

<!-- **Slide 10: Fluxo de Trabalho - Etapa 3** -->

## 3. Análise do K-Fold e Seleção dos Melhores Hiperparâmetros

*   **Objetivo:** Agregar resultados do K-Fold e escolher os melhores parâmetros.
*   Métrica de seleção: `balanced_accuracy` (média entre os folds).
    ```python
    aggregated_kfold_metrics = defaultdict(...) # Agrega métricas por (extrator, classificador, params)
    average_kfold_metrics = {}                # Média das métricas
    best_params_per_clf_extractor = {}        # Guarda {params, avg_score}

    selection_metric = 'balanced_accuracy'

    for (extractor, classifier, params_key), metrics_list_dict in aggregated_kfold_metrics.items():
        # Calcula médias das métricas
        # ...
        # Atualiza best_params_per_clf_extractor se a pontuação atual for melhor
    ```
*   Imprime um resumo dos melhores parâmetros encontrados para cada (Extrator, Classificador).

---

<!-- **Slide 11: Fluxo de Trabalho - Etapa 4** -->

## 4. Avaliação Final no Conjunto de Teste

*   **Objetivo:** Avaliar os modelos (com os melhores hiperparâmetros selecionados) no conjunto de teste (`X_test_full`, `y_test_full`).
*   **Importante:** O classificador é treinado novamente com os melhores parâmetros, mas desta vez usando **TODAS** as características de treino (`X_train_full`).
    ```python
    final_test_results = []
    for extractor_name, data in feature_data.items():
        X_train_full = data['X_train_full'] # Características de treino COMPLETAS
        y_train_full = data['y_train_full']
        X_test_full = data['X_test_full']   # Características de teste COMPLETAS
        y_test_full = data['y_test_full']

        for clf_name, clf_class in classifiers_map.items():
             best_info = best_params_per_clf_extractor.get((extractor_name, clf_name))
             if best_info:
                params = best_info['params']
                clf = clf_class(**params)
                clf.fit(X_train_full, y_train_full) # TREINA NO TREINO COMPLETO
                y_pred_test = clf.predict(X_test_full) # PREDIZ NO TESTE
                test_metrics = evaluate_metrics(y_test_full, y_pred_test, ...)
                final_test_results.append({...})
    ```

---

<!-- **Slide 12: Exibição e Análise dos Resultados Finais** -->

## Detalhes do Desempenho no Teste

*   Para cada combinação (Extrator + Classificador com melhores params):
    *   **Métricas Escaladas:**
        ```python
        print(f"    Accuracy: {metrics.get('accuracy'):.4f}")
        print(f"    Balanced Accuracy: {metrics.get('balanced_accuracy'):.4f}")
        # ... e outras (Precision, Recall, F1 - Macro e Ponderado)
        ```
    *   **Matriz de Confusão (Visual):**
        ```python
        cm_np = np.array(metrics.get('confusion_matrix'))
        sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {result["classifier"]} ({result["extractor"]})')
        plt.show()
        ```
    *   **Métricas Por Classe (TP, TN, FP, FN):**
        ```python
        for class_label in sorted_class_labels:
            print(f"    {class_label}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        ```

---

<!-- **Slide 13: Identificando o "Campeão"** -->

## Melhor Resultado Geral no Teste

*   Identifica a configuração (Extrator + Classificador + Parâmetros) com a **maior `balanced_accuracy` no conjunto de teste**.
    ```python
    best_result_entry = None
    highest_balanced_accuracy = -1.0
    for result in final_test_results:
        if 'error' not in result and result['metrics'] is not None:
            current_bal_acc = result['metrics'].get('balanced_accuracy')
            if current_bal_acc is not None and current_bal_acc > highest_balanced_accuracy:
                highest_balanced_accuracy = current_bal_acc
                best_result_entry = result
    # Imprime detalhes do best_result_entry
    ```

---

<!-- **Slide 14: Salvando os Resultados** -->

## Registrando Tudo

*   Todos os resultados (K-Fold, Teste Final, Melhor Geral) são salvos em um arquivo de texto para referência futura.
    ```python
    output_filename = "evaluation_results.txt"
    try:
        with open(output_filename, "w") as f:
            f.write("--- K-Fold Cross-Validation Summary ---\n")
            # ... Escreve resumo do K-Fold ...

            f.write("--- Final Evaluation Results on Test Set ---\n")
            # ... Escreve resultados detalhados do teste ...

            f.write("--- Overall Best Result on Test Set ---\n")
            # ... Escreve detalhes do melhor modelo geral ...
        print(f"Results successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving results: {e}")
    ```

---

<!-- **Slide 15: Conclusão** -->

## Resumo do Processo

1.  **Preparação:** Configuração, dados e transformações.
2.  **Extração de Features:** ViTs transformam imagens em vetores.
3.  **K-Fold:** Otimização de hiperparâmetros dos classificadores sobre as features de treino.
4.  **Avaliação Final:** Teste dos melhores modelos (Extrator + Classificador + Params) no conjunto de teste.
5.  **Análise e Relatório:** Métricas detalhadas, matrizes de confusão e salvamento dos resultados.

**Benefícios:** Avaliação robusta e comparativa de diferentes abordagens, com otimização de hiperparâmetros.

