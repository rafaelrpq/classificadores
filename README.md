- Subdividir o conjunto de treino original subconjuntos de treino e validação, usando kfold com k = 5. Desse modo, é possível obter 5 medidas de desempenho (uma por validação) para cada configuração (método alternativo) avaliada. 
Obs.: Cada método alternativo será composto de um extrator de características + um classificador + um hiperparâmetro para o classificador. 

- Como extrator de características, avaliar o CLS Token de diferentes modelos ViT disponíveis no HugginFace

- Como classificador, avaliar SVM linear, MLP, Random Forest e KNN. 

- Cada classificador deve ser avaliado com 5 hiperparâmetros distintos (por exemplo, KNN com K = 1, 3, 5, 7 e 9). 

- Selecionar a configuração ideal com base nos resultados de validação. 

```mermaid
graph TD
    %% Subgraph: Inicialização e Carregamento de Dados
    subgraph A[Configuração e Preparação de Dados]
        A0(Início do Pipeline) --> A1[Definir Hiperparâmetros de Classificadores]
        A1 --> A2[Configurar Dispositivo GPU/CPU]
        A2 --> A3[Definir Transformações para Imagens]
        A3 --> A4[Clonar Repositório do Dataset CTCB]

        A4 --> A5{Diretório 'Test' Existe?}
        A5 -- Sim --> A6[Carregar Datasets de Treino e Teste separados]
        A5 -- Não --> A7[Carregar Dataset Completo diretório 'Train']
        A7 --> A8[Dividir Dataset em Treino/Teste Stratificado]
        A8 --> A6
        A6 --> A9[Verificar Classes e Criar DataLoaders]
    end

    %% Subgraph: Extração de Características
    subgraph B[Extração de Características ViT]
        A9 --> B0[Inicializar Extratores de Características ViT DINO, ViT-Base, ViT-Large]
        B0 --> B1{Para Cada Extrator ViT}
        B1 --> B5(Características Extraídas Prontas)
        B1 --> B2[Extrair Características de Imagens do Conjunto de Treino COMPLETO X_train_full, y_train_full]
        B2 --> B3[Extrair Características de Imagens do Conjunto de Teste COMPLETO X_test_full, y_test_full]
        B3 --> B4[Armazenar Características Extraídas]
        B4 --> B1
    end

    %% Subgraph: Validação Cruzada K-Fold para Otimização de Hiperparâmetros
    subgraph C[Validação Cruzada K-Fold]
        B5 --> C0[Inicializar KFold 5 Folds, embaralhado]
        C1 --> C10(Resultados K-Fold Coletados)
        C0 --> C1{Para Cada Extrator de Características}
        C1 --> C2{Para Cada Fold Treino/Validação do Conjunto de Treino COMPLETO}
        C2 --> C3[Dividir Características de Treino em Fold de Treino e Fold de Validação]
        C3 --> C4{Para Cada Tipo de Classificador SVM, MLP, RF, KNN}
        C4 --> C5{Para Cada Combinação de Hiperparâmetros}
        C5 --> C6[Instanciar Classificador com Parâmetros Atuais]
        C6 --> C7[Treinar Classificador no Fold de Treino]
        C7 --> C8[Fazer Previsões no Fold de Validação]
        C8 --> C9[Avaliar Métricas Acurácia Ponderada, Precisão, Recall, F1, CM e Armazenar Resultados]
        C9 --> C5
        C5 --> C4
        C4 --> C2
        C2 --> C1
    end

    %% Subgraph: Seleção de Hiperparâmetros
    subgraph D[Seleção dos Melhores Hiperparâmetros]
        C10 --> D0[Aggregar Resultados K-Fold Média das Métricas por Extrator/Classificador/Parâmetro]
        D0 --> D1[Selecionar Melhores Hiperparâmetros com Base na Média da Acurácia Ponderada por Amostra]
        D1 --> D2(Melhores Modelos por Extrator/Classificador Definidos)
    end

    %% Subgraph: Avaliação Final no Conjunto de Teste
    subgraph E[Avaliação Final no Teste]
        D2 --> E0{Para Cada Extrator de Características}
        E0 --> E7(Avaliação Final Concluída)
        E0 --> E1{Para Cada Tipo de Classificador}
        E1 --> E2[Recuperar Melhores Parâmetros Selecionados]
        E2 --> E3[Instanciar Classificador com Melhores Parâmetros]
        E3 --> E4[Treinar Classificador no Conjunto de Treino COMPLETO X_train_full, y_train_full]
        E4 --> E5[Fazer Previsões no Conjunto de Teste X_test_full]
        E5 --> E6[Avaliar Métricas Finais Acurácia Ponderada, Precisão, Recall, F1, CM e Armazenar Resultados]
        E6 --> E1
        E1 --> E0
    end

    %% Subgraph: Exibição e Salvamento de Resultados
    subgraph F[Exibição e Salvamento de Resultados]
        E7 --> F0[Exibir Detalhes dos Resultados Finais no Console]
        F0 --> F1[Plotar Matrizes de Confusão para Cada Modelo Avaliado]
        F1 --> F2[Identificar e Exibir o Melhor Modelo Geral Baseado na Acurácia Ponderada no Teste]
        F2 --> F3[Salvar Todos os Resultados em 'evaluation_results.txt']
        F3 --> F4(Fim do Pipeline)
    end
```


- [ ] explicar fcnn
- [ ] reprodizir fcnn
- [ ] fine tuning ??
- [ ] https://huggingface.co/docs/transformers/model_doc/mobilevit