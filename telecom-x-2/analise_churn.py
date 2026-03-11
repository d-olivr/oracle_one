"""
TelecomX - Análise de Evasão de Clientes (Churn)
=================================================
Autor: Análise de Dados
Descrição: Pipeline completo de análise e modelagem preditiva para
           identificar clientes com risco de evasão na TelecomX.
"""

# ============================================================
# 1. IMPORTAÇÕES
# ============================================================
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import os

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})
PALETTE = {"Não Evadiu": "#4C72B0", "Evadiu": "#DD8452"}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. CARREGAMENTO E FLATTEN DO JSON
# ============================================================
print("=" * 60)
print("ETAPA 1 – CARREGAMENTO DOS DADOS")
print("=" * 60)

with open("TelecomX_Data.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

records = []
for r in raw:
    flat = {
        "customerID":       r.get("customerID"),
        "Churn":            r.get("Churn"),
        "gender":           r["customer"].get("gender"),
        "SeniorCitizen":    r["customer"].get("SeniorCitizen"),
        "Partner":          r["customer"].get("Partner"),
        "Dependents":       r["customer"].get("Dependents"),
        "tenure":           r["customer"].get("tenure"),
        "PhoneService":     r["phone"].get("PhoneService"),
        "MultipleLines":    r["phone"].get("MultipleLines"),
        "InternetService":  r["internet"].get("InternetService"),
        "OnlineSecurity":   r["internet"].get("OnlineSecurity"),
        "OnlineBackup":     r["internet"].get("OnlineBackup"),
        "DeviceProtection": r["internet"].get("DeviceProtection"),
        "TechSupport":      r["internet"].get("TechSupport"),
        "StreamingTV":      r["internet"].get("StreamingTV"),
        "StreamingMovies":  r["internet"].get("StreamingMovies"),
        "Contract":         r["account"].get("Contract"),
        "PaperlessBilling": r["account"].get("PaperlessBilling"),
        "PaymentMethod":    r["account"].get("PaymentMethod"),
        "MonthlyCharges":   r["account"]["Charges"].get("Monthly"),
        "TotalCharges":     r["account"]["Charges"].get("Total"),
    }
    records.append(flat)

df = pd.DataFrame(records)
print(f"\nShape inicial: {df.shape}")
print(f"Colunas: {list(df.columns)}\n")
print(df.head(3).to_string())

# ============================================================
# 3. LIMPEZA E PRÉ-PROCESSAMENTO
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 2 – LIMPEZA E PRÉ-PROCESSAMENTO")
print("=" * 60)

# Conversão de tipos
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

print(f"\nValores nulos antes da limpeza:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Preencher TotalCharges nulo com 0 (clientes novos, tenure=0)
df["TotalCharges"].fillna(0, inplace=True)
df.dropna(inplace=True)  # remover linhas com outros nulos residuais

print(f"\nValores nulos após a limpeza:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string() or 'Nenhum'}")

# Remover coluna sem valor preditivo
df.drop(columns=["customerID"], inplace=True)
print("\n→ Coluna 'customerID' removida (identificador único sem valor preditivo).")

# Converter Churn em binário
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.dropna(subset=["Churn"], inplace=True)  # remover registros sem Churn definido
df["Churn"] = df["Churn"].astype(int)
print("→ 'Churn' convertido para binário (1=Evadiu, 0=Ficou).")
print(f"  Linhas removidas por Churn nulo: {7267 - len(df)}")

print(f"\nShape após limpeza: {df.shape}")

# ============================================================
# 4. ANÁLISE EXPLORATÓRIA – DISTRIBUIÇÃO DE CHURN
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 3 – ANÁLISE EXPLORATÓRIA")
print("=" * 60)

churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100

print("\nDistribuição de Churn:")
print(f"  Não Evadiu (0): {churn_counts[0]:>5}  ({churn_pct[0]:.1f}%)")
print(f"  Evadiu     (1): {churn_counts[1]:>5}  ({churn_pct[1]:.1f}%)")
print(f"\n→ Desequilíbrio de classes: {churn_pct[0]:.1f}% vs {churn_pct[1]:.1f}%")
print("  O dataset é desbalanceado. Modelos que ignoram isso tendem a prever")
print("  majoritariamente a classe dominante. Usaremos class_weight='balanced'")
print("  nos modelos para mitigar esse efeito.")

# Gráfico – Distribuição de Churn
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribuição de Evasão de Clientes (Churn)", fontsize=14, fontweight="bold")

labels = ["Não Evadiu", "Evadiu"]
colors = [PALETTE["Não Evadiu"], PALETTE["Evadiu"]]

axes[0].bar(labels, [churn_counts[0], churn_counts[1]], color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Contagem Absoluta")
axes[0].set_ylabel("Número de Clientes")
for i, v in enumerate([churn_counts[0], churn_counts[1]]):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

axes[1].pie(
    [churn_counts[0], churn_counts[1]],
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Proporção (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribuicao_churn.png", bbox_inches="tight")
plt.close()
print(f"\n→ Gráfico salvo: 01_distribuicao_churn.png")

# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 4 – CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS (ONE-HOT ENCODING)")
print("=" * 60)

cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nVariáveis categóricas identificadas ({len(cat_cols)}):")
for c in cat_cols:
    print(f"  {c}: {df[c].unique().tolist()}")

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Churn já é int (0/1), remover NaN residuais e garantir tipo
df_encoded.dropna(subset=["Churn"], inplace=True)
df_encoded["Churn"] = df_encoded["Churn"].astype(int)

print(f"\nShape antes do encoding: {df.shape}")
print(f"Shape após  o encoding:  {df_encoded.shape}")
print(f"→ {df_encoded.shape[1] - df.shape[1]} novas colunas binárias criadas.")

# ============================================================
# 6. MATRIZ DE CORRELAÇÃO
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 5 – MATRIZ DE CORRELAÇÃO")
print("=" * 60)

# Correlação com Churn (top 15)
corr_churn = df_encoded.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
print("\nTop 15 variáveis com maior correlação com Churn:")
print(corr_churn.head(15).to_string())

# Heatmap – correlações numéricas originais
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
corr_num = df_encoded[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_num, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 11}
)
ax.set_title("Matriz de Correlação – Variáveis Numéricas", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_matriz_correlacao.png", bbox_inches="tight")
plt.close()
print(f"\n→ Gráfico salvo: 02_matriz_correlacao.png")

# Heatmap top-20 correlações com Churn
top20 = corr_churn.head(20).index.tolist() + ["Churn"]
corr_top = df_encoded[top20].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    corr_top, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.3, ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Top 20 Variáveis – Correlação com Churn", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlacao_top20.png", bbox_inches="tight")
plt.close()
print(f"→ Gráfico salvo: 03_correlacao_top20.png")

# ============================================================
# 7. VISUALIZAÇÕES EXPLORATÓRIAS ADICIONAIS
# ============================================================

# Churn por tipo de contrato
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Perfil de Evasão por Variáveis-Chave", fontsize=13, fontweight="bold")

contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
axes[0].bar(contract_churn.index, contract_churn.values * 100,
            color=[PALETTE["Evadiu"], "#aec7e8", PALETTE["Não Evadiu"]], edgecolor="white")
axes[0].set_title("Taxa de Churn por Tipo de Contrato")
axes[0].set_ylabel("% de Evasão")
axes[0].set_xticklabels(contract_churn.index, rotation=10)
for i, v in enumerate(contract_churn.values):
    axes[0].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontweight="bold")

# Churn por Internet
internet_churn = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
axes[1].bar(internet_churn.index, internet_churn.values * 100,
            color=[PALETTE["Evadiu"], "#aec7e8", PALETTE["Não Evadiu"]], edgecolor="white")
axes[1].set_title("Taxa de Churn por Serviço de Internet")
axes[1].set_ylabel("% de Evasão")
for i, v in enumerate(internet_churn.values):
    axes[1].text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontweight="bold")

# Tenure por Churn
df.boxplot(column="tenure", by="Churn", ax=axes[2],
           boxprops=dict(color="#333"), medianprops=dict(color="red", linewidth=2))
axes[2].set_title("Tempo de Contrato (tenure) por Churn")
axes[2].set_xlabel("Churn (0=Ficou, 1=Evadiu)")
axes[2].set_ylabel("Meses")
plt.suptitle("")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_analise_exploratoria.png", bbox_inches="tight")
plt.close()
print(f"→ Gráfico salvo: 04_analise_exploratoria.png")

# ============================================================
# 8. PREPARAÇÃO PARA MODELAGEM
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 6 – PREPARAÇÃO PARA MODELAGEM")
print("=" * 60)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisão treino/teste (80/20, estratificado):")
print(f"  Treino: {X_train.shape[0]} amostras")
print(f"  Teste:  {X_test.shape[0]} amostras")

# Normalização para Regressão Logística
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n→ Normalização (StandardScaler) aplicada para a Regressão Logística.")
print("  Justificativa: algoritmos baseados em gradiente e distância são")
print("  sensíveis à escala das variáveis. O Random Forest não precisa,")
print("  pois divide features por limiares e não por distância.")

# ============================================================
# 9. MODELO 1 – REGRESSÃO LOGÍSTICA
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 7 – MODELO 1: REGRESSÃO LOGÍSTICA")
print("=" * 60)
print("""
Justificativa: Modelo linear interpretável, amplamente utilizado para
classificação binária. Permite analisar os coeficientes de cada variável,
tornando fácil entender o impacto de cada feature na probabilidade de evasão.
Requer normalização pois é sensível à escala das features.
""")

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_scaled, y_train)

y_pred_lr  = lr.predict(X_test_scaled)
y_prob_lr  = lr.predict_proba(X_test_scaled)[:, 1]

print("Relatório de Classificação – Regressão Logística:")
print(classification_report(y_test, y_pred_lr, target_names=["Não Evadiu", "Evadiu"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}")

# Coeficientes mais relevantes
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": lr.coef_[0]
}).sort_values("Coeficiente", key=abs, ascending=False)

print("\nTop 15 variáveis por magnitude do coeficiente:")
print(coef_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top15_lr = coef_df.head(15).sort_values("Coeficiente")
colors_lr = [PALETTE["Evadiu"] if v > 0 else PALETTE["Não Evadiu"] for v in top15_lr["Coeficiente"]]
ax.barh(top15_lr["Feature"], top15_lr["Coeficiente"], color=colors_lr, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Regressão Logística – Top 15 Coeficientes", fontsize=13, fontweight="bold")
ax.set_xlabel("Coeficiente (log-odds)")
patch_pos = mpatches.Patch(color=PALETTE["Evadiu"],    label="↑ Aumenta risco de evasão")
patch_neg = mpatches.Patch(color=PALETTE["Não Evadiu"], label="↓ Reduz risco de evasão")
ax.legend(handles=[patch_pos, patch_neg])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_lr_coeficientes.png", bbox_inches="tight")
plt.close()
print(f"\n→ Gráfico salvo: 05_lr_coeficientes.png")

# ============================================================
# 10. MODELO 2 – RANDOM FOREST
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 8 – MODELO 2: RANDOM FOREST")
print("=" * 60)
print("""
Justificativa: Modelo de ensemble baseado em árvores de decisão. Robusto a
outliers, não requer normalização e captura relações não-lineares entre
variáveis. Fornece importância de features de forma nativa, facilitando a
interpretação. Adequado para datasets desbalanceados com class_weight.
""")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("Relatório de Classificação – Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=["Não Evadiu", "Evadiu"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# Importância das features
feat_imp = pd.DataFrame({
    "Feature":    X.columns,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False)

print("\nTop 15 variáveis por importância (Random Forest):")
print(feat_imp.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top15_rf = feat_imp.head(15).sort_values("Importância")
ax.barh(top15_rf["Feature"], top15_rf["Importância"],
        color=PALETTE["Não Evadiu"], edgecolor="white")
ax.set_title("Random Forest – Top 15 Variáveis por Importância", fontsize=13, fontweight="bold")
ax.set_xlabel("Importância (redução de impureza)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_rf_importancia.png", bbox_inches="tight")
plt.close()
print(f"\n→ Gráfico salvo: 06_rf_importancia.png")

# ============================================================
# 11. COMPARAÇÃO – CURVAS ROC
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 9 – COMPARAÇÃO DE MODELOS (CURVA ROC)")
print("=" * 60)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_lr, tpr_lr, label=f"Regressão Logística (AUC = {auc_lr:.3f})",
        color=PALETTE["Não Evadiu"], linewidth=2)
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})",
        color=PALETTE["Evadiu"], linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório (AUC = 0.500)")
ax.set_xlabel("Taxa de Falsos Positivos")
ax.set_ylabel("Taxa de Verdadeiros Positivos")
ax.set_title("Curva ROC – Comparação dos Modelos", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_curvas_roc.png", bbox_inches="tight")
plt.close()
print(f"→ Gráfico salvo: 07_curvas_roc.png")

# Matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrizes de Confusão", fontsize=13, fontweight="bold")

for ax, y_pred, title in zip(
    axes,
    [y_pred_lr, y_pred_rf],
    ["Regressão Logística", "Random Forest"]
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não Evadiu", "Evadiu"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_matrizes_confusao.png", bbox_inches="tight")
plt.close()
print(f"→ Gráfico salvo: 08_matrizes_confusao.png")

# ============================================================
# 12. RESUMO FINAL
# ============================================================
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)

summary = pd.DataFrame({
    "Modelo": ["Regressão Logística", "Random Forest"],
    "AUC-ROC": [f"{auc_lr:.4f}", f"{auc_rf:.4f}"],
    "Normalização": ["Sim (StandardScaler)", "Não necessária"],
})
print(summary.to_string(index=False))

print("""
CONCLUSÕES:
-----------
1. O dataset possui ~26% de evasão, configurando desbalanceamento moderado.
   Ambos os modelos foram configurados com class_weight='balanced'.

2. A Regressão Logística identifica como principais fatores de evasão:
   contratos mensais, fibra óptica sem serviços de segurança e suporte,
   e cobrança eletrônica sem paperless. Tempo de contrato longo (tenure)
   e contratos anuais/bianuais reduzem o risco.

3. O Random Forest confirma esses achados e destaca MonthlyCharges, tenure
   e TotalCharges como as variáveis contínuas mais preditivas.

4. Recomendações estratégicas:
   • Incentivar contratos de longo prazo (1-2 anos) com benefícios
   • Ofertar pacotes de OnlineSecurity e TechSupport para clientes de fibra
   • Criar programas de fidelização para clientes com menos de 12 meses
   • Monitorar clientes com conta mensal + fibra + pagamento eletrônico
""")

print("=" * 60)
print("Análise concluída! Gráficos salvos na pasta 'outputs/'.")
print("=" * 60)
