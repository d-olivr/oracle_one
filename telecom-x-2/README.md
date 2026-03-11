# 📊 TelecomX – Análise de Evasão de Clientes (Churn)

> Projeto de análise exploratória de dados e modelagem preditiva para identificar clientes com risco de cancelamento na empresa TelecomX.

---

## 📌 Objetivo

Desenvolver um pipeline completo de ciência de dados que permita à TelecomX:

1. Compreender o perfil dos clientes que cancelam o serviço
2. Identificar as variáveis com maior impacto na evasão
3. Prever quais clientes têm maior risco de churn com modelos de machine learning

---

## 🔧 Como Executar

### Pré-requisitos

- Python 3.8+
- pip

### Instalação das dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execução

```bash
python analise_churn.py
```

Os gráficos serão salvos automaticamente na pasta `outputs/`.

---

## 📋 Etapas do Pipeline

### 1. Carregamento e Achatamento dos Dados
O dataset está em formato JSON aninhado (com sub-objetos `customer`, `phone`, `internet`, `account`). O script realiza o *flatten* para um DataFrame tabular.

### 2. Limpeza e Pré-processamento

| Ação | Justificativa |
|------|---------------|
| Remoção da coluna `customerID` | Identificador único sem valor preditivo — pode causar overfitting |
| Conversão de `TotalCharges` para numérico | Estava como string no JSON |
| Preenchimento de `TotalCharges` nulo com 0 | Clientes com `tenure=0` (sem cobrança acumulada) |
| Remoção de linhas com `Churn` nulo | Não é possível treinar sem o target definido |

### 3. Balanceamento de Classes

- **73,4%** dos clientes não evadiram
- **26,6%** evadiram

O dataset é **desbalanceado**. Ambos os modelos utilizam `class_weight='balanced'` para compensar esse desequilíbrio, evitando que o modelo aprenda a prever apenas a classe majoritária.

### 4. One-Hot Encoding

15 variáveis categóricas foram transformadas em 26 colunas binárias via `pd.get_dummies()`, resultando em um DataFrame com 46 features no total. Essa técnica é necessária pois algoritmos de ML operam com valores numéricos.

### 5. Matriz de Correlação

As variáveis com maior correlação com `Churn`:

| Variável | Correlação |
|----------|-----------|
| `Contract_Month-to-month` | +0.40 |
| `tenure` | -0.35 |
| `OnlineSecurity_No` | +0.34 |
| `TechSupport_No` | +0.34 |
| `InternetService_Fiber optic` | +0.31 |
| `Contract_Two year` | -0.30 |
| `PaymentMethod_Electronic check` | +0.30 |

---

## 🤖 Modelos de Machine Learning

### Modelo 1 – Regressão Logística

**Justificativa:** Modelo linear interpretável, padrão para classificação binária. Permite analisar diretamente os coeficientes de cada variável para entender o impacto na probabilidade de evasão.

**Normalização:** ✅ Necessária (StandardScaler). A Regressão Logística otimiza por gradiente e é sensível à escala das features. Variáveis em escalas muito diferentes (ex: `tenure` em meses vs. `TotalCharges` em reais) distorcem os coeficientes sem normalização.

**Resultados:**

| Métrica | Não Evadiu | Evadiu |
|---------|-----------|--------|
| Precision | 0.91 | 0.51 |
| Recall | 0.73 | 0.80 |
| F1-Score | 0.81 | 0.62 |
| **AUC-ROC** | — | **0.8446** |

**Variáveis mais relevantes (coeficientes):**
- `tenure` → coeficiente negativo forte: mais tempo = menor risco de evasão
- `MonthlyCharges` → negativo após normalização (correlacionado com tenure)
- `InternetService_Fiber optic` → coeficiente positivo: fibra aumenta risco
- `Contract_Two year` → coeficiente negativo: contratos longos retêm clientes

---

### Modelo 2 – Random Forest

**Justificativa:** Modelo de ensemble baseado em múltiplas árvores de decisão. Captura relações não-lineares, é robusto a outliers e fornece importância de features de forma nativa. Não requer normalização pois as divisões nas árvores são baseadas em limiares relativos, independentes da escala.

**Normalização:** ❌ Não necessária.

**Resultados:**

| Métrica | Não Evadiu | Evadiu |
|---------|-----------|--------|
| Precision | 0.88 | 0.56 |
| Recall | 0.80 | 0.70 |
| F1-Score | 0.84 | 0.62 |
| **AUC-ROC** | — | **0.8450** |

**Variáveis mais importantes (feature importance):**
1. `tenure` (13,1%) – tempo de contrato é o preditor mais poderoso
2. `Contract_Month-to-month` (11,3%) – contratos mensais elevam o risco
3. `TotalCharges` (10,4%) – total gasto acumulado
4. `MonthlyCharges` (7,8%) – valor mensal da fatura
5. `Contract_Two year` (5,9%) – contratos longos reduzem o risco

---

## 📈 Comparação dos Modelos

| Modelo | AUC-ROC | Normalização | Interpretabilidade |
|--------|---------|-------------|-------------------|
| Regressão Logística | 0.8446 | Sim | Alta (coeficientes) |
| Random Forest | 0.8450 | Não | Média (feature importance) |

Ambos os modelos alcançam **AUC-ROC ≈ 0.845**, indicando boa capacidade discriminativa. O Random Forest tem leve vantagem em recall para a classe minoritária, enquanto a Regressão Logística é preferível quando a interpretabilidade é prioridade.

---

## 💡 Conclusões e Recomendações

### Perfil do Cliente com Alto Risco de Evasão:
- Contrato **mês a mês**
- Serviço de **fibra óptica** sem add-ons de segurança/suporte
- Pagamento via **cheque eletrônico**
- Pouco tempo de contrato (**tenure < 12 meses**)
- **Sem parceiro ou dependentes**

### Ações Recomendadas:
1. **Incentivar contratos anuais/bianuais** com descontos progressivos
2. **Ofertar pacotes de OnlineSecurity e TechSupport** para clientes de fibra
3. **Programa de fidelização** para clientes nos primeiros 12 meses
4. **Alertas automáticos** para clientes com perfil de alto risco identificado pelo modelo
5. **Revisar a experiência** de clientes com pagamento via cheque eletrônico

---

## 📚 Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** – manipulação de dados
- **NumPy** – operações numéricas
- **Matplotlib / Seaborn** – visualizações
- **Scikit-learn** – modelos de machine learning e métricas

