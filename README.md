# 🤖 Classificação de Discurso de Ódio (Hate Speech) em Português

Este projeto implementa um pipeline de Machine Learning (ML) focado na classificação automática de textos (presumivelmente tweets) para a detecção de Discurso de Ódio ("Hate Speech") na língua portuguesa.

O fluxo de trabalho abrange desde o pré-processamento robusto de dados textuais até o treinamento, vetorização e avaliação de um modelo de classificação.

## ⚙️ Tecnologias e Bibliotecas

| Categoria | Biblioteca | Uso Principal |
| :--- | :--- | :--- |
| **Data Science** | `pandas`, `numpy` | Manipulação e organização do dataset. |
| **NLP** | `nltk`, `spaCy` (`pt_core_news_sm`), `unidecode` | Tokenização, remoção de stopwords, lematização e tratamento de acentuação. |
| **Vetorização** | `TfidfVectorizer`, `CountVectorizer` | Conversão de texto em representações numéricas (features). |
| **Machine Learning** | `sklearn` | Divisão de dados, treinamento de modelos e avaliação de métricas. |
| **Visualização** | `matplotlib`, `seaborn` | Visualização de resultados e Matriz de Confusão. |
| **Modelos** | `ExtraTreesClassifier` | Classificador principal utilizado para o treinamento. |

## 📑 Estrutura do Pipeline

O script segue as seguintes etapas:

### 1. Pré-processamento e Limpeza de Dados

O objetivo é padronizar e limpar o texto antes da vetorização:
* **Remoção de Colunas:** Seleção das colunas `Hate.speech` (rótulo) e `text` (texto).
* **Normalização:** Conversão para minúsculas e remoção de acentos (`unidecode`).
* **Filtros:** Remoção de caracteres especiais e da *stopword* comum `rt`.
* **Lematização:** O texto é lematizado (`spacy.load('pt_core_news_sm')`) para reduzir as palavras à sua forma base, aumentando a eficácia do modelo.
* **Stopwords:** Remoção de *stopwords* em português (`nltk.corpus.stopwords.words('portuguese')`).

### 2. Vetorização (Feature Engineering)

Os textos limpos e lematizados são convertidos em vetores de características para alimentar o modelo de ML:
* **`CountVectorizer`:** Usado para criar uma matriz de frequência de termos (`df_count`).
* **`TfidfVectorizer`:** Também preparado para análise, que calcula a importância de cada termo (Term Frequency-Inverse Document Frequency) (`df_tfidf`).

### 3. Divisão e Estratificação dos Dados

O dataset é dividido em três conjuntos distintos com a proporção das classes preservada (`stratify`):
* **Treinamento:** Para o aprendizado do modelo.
* **Validação (Vad):** Para ajuste inicial ou teste durante o desenvolvimento.
* **Teste:** Para a avaliação final do desempenho do modelo em dados não vistos (30% do total).

### 4. Treinamento e Avaliação do Modelo

O modelo `ExtraTreesClassifier` é escolhido para a tarefa de classificação:
* **Treinamento:** O modelo é treinado usando as features geradas pelo `CountVectorizer` sobre o texto de treino.
* **Validação Cruzada (K-Fold):** Utilizada com $k=5$ para avaliar a estabilidade e robustez do modelo durante o treinamento.

### 5. Métricas de Desempenho

O desempenho é medido no conjunto de Teste, utilizando métricas ponderadas (`average='weighted'`) para lidar com o possível desbalanceamento de classes:

| Métrica | Função (Python) |
| :--- | :--- |
| **Acurácia** | `accuracy_score` |
| **Precisão** | `precision_score` |
| **Revocação (Recall)** | `recall_score` |
| **F1-Score** | `f1_score` |

Além disso, o script gera uma **Matriz de Confusão** visualizada por um *heatmap* (`sns.heatmap`) para entender onde o modelo está cometendo erros (Falsos Positivos e Falsos Negativos).

## 🚀 Como Executar

1.  **Instalar Dependências:**
    ```bash
    pip install pandas nltk scikit-learn matplotlib seaborn unidecode spacy
    python -m spacy download pt_core_news_sm
    ```

2.  **Preparar o Dataset:**
    * Certifique-se de que o arquivo de dados (`tweetscomhierarquia.csv`) está no mesmo diretório do script.
    * O arquivo deve conter as colunas `Hate.speech` e `text`.

3.  **Executar o Script:**
    ```bash
    python TwitterHateSpeechDetection.ipynb
    ```
    
# Realização:
  + Ciência da Computação da FEI. 
  + Desenvolvedor: Guilherme Marcato Mendes Justiça
  + Orientadora: Prof. Dra. Leila Cristina C. Bergamasco - Orientadora, coordenadora e chefe do departamento curso  
  + Em parceria com : Ministério Público Federal
