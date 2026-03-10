# 🛒 AluraStoreBr – Análise de Desempenho das Lojas

> **Challenge 1 – Data Science | Alura**
>
> Auxiliar o Senhor João a decidir qual loja da sua rede **Alura Store** deve ser vendida para iniciar um novo empreendimento.

---

## 📋 Descrição do Projeto

Este projeto analisa dados de vendas, avaliações e desempenho das **4 lojas fictícias** da Alura Store com o objetivo de identificar a loja com menor eficiência e apresentar uma recomendação baseada em dados.

---

## 📊 Métricas Analisadas

| Métrica | Descrição |
|---|---|
| **Faturamento Total** | Soma de todos os preços de venda por loja |
| **Vendas por Categoria** | Quantidade de itens vendidos em cada categoria |
| **Média de Avaliação** | Média das avaliações dos clientes (escala 1–5) |
| **Produtos mais/menos vendidos** | Produto com maior e menor número de vendas |
| **Frete Médio** | Média do custo de frete por loja |

---

## 📈 Gráficos Gerados

1. **Barras** – Faturamento total por loja
2. **Barras Horizontais** – Média de avaliação por loja
3. **Barras Agrupadas** – Vendas por categoria em cada loja
4. **Pizza** – Distribuição do frete médio
5. **Dispersão** – Faturamento × Avaliação (visão geral de eficiência)

---

## 🏆 Recomendação Final

> Com base nas análises realizadas, recomenda-se que o Senhor João venda a **Loja 4**.

A Loja 4 apresenta:
- **Menor faturamento total** da rede (R$ 1.384.497,58)
- **Avaliação de clientes** abaixo das melhores lojas
- **Ticket médio menor**, indicando mix de produtos de menor valor agregado
- **Pior posicionamento** no gráfico de dispersão (faturamento × avaliação)

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas matplotlib
```

### Execução

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/AluraStoreBr.git
   cd AluraStoreBr
   ```

2. Abra o notebook:
   ```bash
   jupyter notebook AluraStoreBr.ipynb
   ```

3. Execute todas as células (`Kernel > Restart & Run All`)

---

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 👤 Autor

Desenvolvido como parte do **Challenge 1 de Data Science** da [Alura](https://www.alura.com.br/).
