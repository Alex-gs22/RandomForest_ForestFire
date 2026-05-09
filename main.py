import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PASO 1 - Carga del dataset
df = pd.read_csv("forestfires.csv")

# PASO 2 — Identificacion de X e y

# Con el long1p se comprime la distribución de ley de potencias de "area" para que el modelo aprenda
# de todos por igual, no solo de los incendios extremos
df["log_area"] = np.log1p(df["area"])

# Aplicamos One Hot Encoding en month y day porque no tienen orden numerico util
orden_meses = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
orden_dias  = ["mon","tue","wed","thu","fri","sat","sun"]
df["month"] = pd.Categorical(df["month"], categories=orden_meses, ordered=False)
df["day"]   = pd.Categorical(df["day"],   categories=orden_dias,  ordered=False)
df_enc = pd.get_dummies(df, columns=["month", "day"], drop_first=False)
# Determinamos que "rain" no aporta información útil (casi siempre es 0) y lo eliminamos
df_enc = df_enc.drop(columns=["rain"])

X = df_enc.drop(columns=["area", "log_area"])
y = df_enc["log_area"]

# PASO 3 — Division del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train: {X_train.shape[0]} filas  |  Test: {X_test.shape[0]} filas\n")

# PASO 4 - Preprocesamiento
# Random Forest usa umbrales sobre valores crudos por lo que la normalización no es requerida
# El resto del preprocesamiento se hizo antes de la división (log-transform y one-hot encoding)

# PASO 5 — Entrenamiento del modelo

# Modelo base como referencia
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
r2_base = r2_score(y_test, rf_base.predict(X_test))
print(f"Modelo base  R²: {r2_base:.4f}\n")

# param_grid es el espacio de hiperparametros a explorar
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [None, 10, 20],
    "max_features"     : ["sqrt", 0.33],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
}

# GridSearchCV realiza una búsqueda exhaustiva sobre param_grid.
grid_search = GridSearchCV(
    estimator  = RandomForestRegressor(bootstrap=True, random_state=42, n_jobs=-1),
    param_grid = param_grid,
    cv         = KFold(n_splits=5, shuffle=True, random_state=42),
    scoring    = "neg_root_mean_squared_error",
    n_jobs     = -1,
    verbose    = 1,
    refit      = True
)

# como resultado obtenemos el mejor modelo encontrado y sus hiperparámetros optimos, evaluados por RMSE en validación cruzada
print("GridSearchCV esta buscando los mejores hiperparámetros...")
grid_search.fit(X_train, y_train)
rf_final = grid_search.best_estimator_

print(f"Mejores hiperparámetros : {grid_search.best_params_}")
print(f"Mejor RMSE (CV train)   : {-grid_search.best_score_:.4f}\n")

# PASO 6 — Evaluacion con metricas adecuadas

y_pred_log = rf_final.predict(X_test)
y_real_ha  = np.expm1(y_test.values)
y_pred_ha  = np.expm1(y_pred_log)
residuos   = y_test.values - y_pred_log

mae_log  = mean_absolute_error(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log   = r2_score(y_test, y_pred_log)
mae_ha   = mean_absolute_error(y_real_ha, y_pred_ha)
rmse_ha  = np.sqrt(mean_squared_error(y_real_ha, y_pred_ha))

# top 5 features por importancia, útil para saber qué variables manda el modelo
top_features = pd.Series(
    rf_final.feature_importances_, index=X_train.columns
).sort_values(ascending=False).head(5)

# resumen visual en terminal: métricas, mejora vs base y features principales
# sencillo
print("Resultados del modelo final:")
print(f"R² (log scale)  : {r2_log:.4f}")
print(f"RMSE (log scale): {rmse_log:.4f}")
print(f"MAE  (log scale): {mae_log:.4f}")
print(" ")
print("Escala real (hectareas quemadas):")
print(f"RMSE (ha)       : {rmse_ha:.2f} ha")
print(f"MAE  (ha)       : {mae_ha:.2f} ha")
print(" ")
print(f"Modelo Base → {r2_base:.4f} ")
print(f"Modelo Final → {r2_log:.4f} ")
print(" ")
print(f"Mejora R² vs base: {r2_log - r2_base:.4f}")

print("  Top 5 features más importantes:            ")
for i, (feat, val) in enumerate(top_features.items(), 1):
    print(f"    {i}. {feat:<10s}  →  {val:.4f}")

# PASO 7 — Graficación
# Aquí si nos tiro paro mi amigo Claude no se lo voy a negar

fig1, ax = plt.subplots(figsize=(6, 6))
fig1.suptitle("RandomForestRegressor — Forest Fires UCI", fontsize=13, fontweight="bold")

sc = ax.scatter(
    y_test, y_pred_log,
    c=y_real_ha, cmap="YlOrRd",
    vmin=0, vmax=np.percentile(y_real_ha, 95),
    s=45, alpha=0.8, edgecolors="white", linewidths=0.4
)
plt.colorbar(sc, ax=ax, label="Hectáreas reales quemadas", shrink=0.85)
lim = [min(y_test.min(), y_pred_log.min()) - 0.2,
       max(y_test.max(), y_pred_log.max()) + 0.2]
ax.plot(lim, lim, color="#0F6E56", linewidth=1.6,
        linestyle="--", label="Predicción perfecta")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("log(area + 1)  real")
ax.set_ylabel("log(area + 1)  predicho")
ax.set_title("Predichos vs Reales", fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.04, 0.95,
        f"R²   = {r2_log:.3f}\nRMSE = {rmse_log:.3f}\nMAE  = {mae_log:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

plt.tight_layout()
plt.savefig("grafica1_predichos_vs_reales.png", dpi=180, bbox_inches="tight")
plt.show()
print("Guardado: grafica1_predichos_vs_reales.png")

# ── Gráfica 2: Distribución del área quemada — punto crítico

fig2, ax2 = plt.subplots(figsize=(6, 5))
fig2.suptitle("RandomForestRegressor — Forest Fires UCI", fontsize=13, fontweight="bold")

area_no_cero = df["area"][df["area"] > 0].values
area_sorted  = np.sort(area_no_cero)[::-1]
ccdf = np.arange(1, len(area_sorted) + 1) / len(area_sorted)  # P(X >= x)

ax2.loglog(area_sorted, ccdf,
           "o", color="#185FA5", markersize=4, alpha=0.7, label="Incendios reales")

# Ajuste de ley de potencias (línea recta en log-log)
log_x = np.log(area_sorted)
log_y = np.log(ccdf)
coef  = np.polyfit(log_x, log_y, 1)
x_fit = np.logspace(np.log10(area_sorted.min()), np.log10(area_sorted.max()), 100)
y_fit = np.exp(coef[1]) * x_fit ** coef[0]
ax2.loglog(x_fit, y_fit, "--", color="#993C1D", linewidth=1.8,
           label=f"Ley de potencias  α ≈ {coef[0]:.2f}")

ax2.set_xlabel("Área quemada (ha)  — escala log")
ax2.set_ylabel("P(X ≥ x)  — escala log")
ax2.set_title("Distribución de ley de potencias\n(firma del punto crítico)", fontweight="bold")
ax2.legend(fontsize=9)
ax2.text(0.97, 0.95,
         f"n = {len(area_no_cero)} incendios\n(excluye area = 0 ha)",
         transform=ax2.transAxes, va="top", ha="right", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

plt.tight_layout()
plt.savefig("grafica2_ley_de_potencias.png", dpi=180, bbox_inches="tight")
plt.show()
print("Guardado: grafica2_ley_de_potencias.png")

# ── Gráfica 3: Mapa de calor del parque (cuadrícula X/Y)

fig3, ax3 = plt.subplots(figsize=(6, 5))
fig3.suptitle("RandomForestRegressor — Forest Fires UCI", fontsize=13, fontweight="bold")

grid_area = df.groupby(["X", "Y"])["area"].mean().reset_index()
pivot     = grid_area.pivot(index="Y", columns="X", values="area")
pivot     = pivot.reindex(index=sorted(pivot.index, reverse=True))  # Y crece hacia arriba

im = ax3.imshow(
    pivot.values,
    cmap="YlOrRd", aspect="auto",
    interpolation="nearest"
)
plt.colorbar(im, ax=ax3, label="Área promedio quemada (ha)", shrink=0.85)

ax3.set_xticks(range(len(pivot.columns)))
ax3.set_yticks(range(len(pivot.index)))
ax3.set_xticklabels(pivot.columns)
ax3.set_yticklabels(pivot.index)

# anotar cada celda con el valor para facilitar lectura
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax3.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=7.5, color="black" if val < 30 else "white")

ax3.set_xlabel("Coordenada X del parque")
ax3.set_ylabel("Coordenada Y del parque")
ax3.set_title("Zonas críticas del parque\n(área promedio quemada por celda)", fontweight="bold")

plt.tight_layout()
plt.savefig("grafica3_zonas_criticas.png", dpi=180, bbox_inches="tight")
plt.show()
print("Guardado: grafica3_zonas_criticas.png")