# 📡 TelecomX — Análise de Evasão de Clientes (Churn)

> Projeto de análise exploratória de dados (EDA) com foco na identificação de padrões de churn na base de clientes da empresa fictícia **TelecomX**.

---

## 📋 Descrição do Projeto

Este projeto foi desenvolvido como parte de um desafio de análise de dados. O objetivo é aplicar o pipeline **ETL (Extração → Transformação → Carga e Análise)** sobre dados de clientes de uma operadora de telecomunicações, identificando os principais fatores associados à **evasão de clientes (churn)**.


## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou Google Colab

### Instalação das dependências

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Executar o notebook

```bash
jupyter notebook TelecomX_BR.ipynb
```

Ou faça o upload do `.ipynb` e do `.json` diretamente no **Google Colab**.

---

## 📦 Pipeline ETL

### 1. 📌 Extração
- Carregamento do arquivo JSON (simulando chamada à API da TelecomX)
- Conversão para DataFrame do Pandas via `pd.json_normalize()`

### 2. 🔧 Transformação
- Achatamento das colunas aninhadas (`customer.*`, `internet.*`, `account.*`)
- Tratamento de valores ausentes em `TotalCharges` (preenchimento por mediana)
- Conversão de tipos de dados (string → numérico)
- Codificação binária da variável-alvo `Churn` (Yes→1 / No→0)
- Remoção de registros duplicados

### 3. 📊 Carga e Análise
- Estatísticas descritivas (média, mediana, desvio padrão, quartis)
- Análise de distribuição de churn
- Correlação entre variáveis numéricas e churn
- Visualizações por: contrato, tipo de internet, método de pagamento, perfil demográfico

---

## 📈 Principais Insights

| Fator | Observação |
|---|---|
| **Taxa de Churn Geral** | ~26–27% dos clientes saíram |
| **Tipo de Contrato** | Contratos mensais: ~42% de churn vs. bianuais: ~3% |
| **Internet Fibra Óptica** | Maior taxa de evasão (~41%) |
| **Tempo de Contrato** | Clientes com churn ficaram ~10 meses; sem churn, ~37 meses |
| **Método de Pagamento** | Cheque eletrônico associado ao maior churn |
| **Perfil Idoso** | Clientes idosos: ~41% de churn vs. ~23% dos demais |

---

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📄 Dicionário de Dados

Consulte o arquivo [`TelecomX_dicionario.md`](./TelecomX_dicionario.md) para a descrição completa de cada coluna do dataset.

---

## 👤 Autor

Desenvolvido como projeto de conclusão de curso — Análise de Dados.
