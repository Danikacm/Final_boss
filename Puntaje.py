import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_excel(r"C:\Users\danie\OneDrive\Escritorio\bases\FinalBoss\DataBases\modified\Basededatosgrados-limpiada.xlsx")


cols_venc = ['0-30', '31-60', '61-90', '91-120', '121+']
df[cols_venc] = df[cols_venc].fillna(0)

# Calcular riesgo ponderado
pesos = {'0-30': 1, '31-60': 1.5, '61-90': 2, '91-120': 2.5, '121+': 3}
df['Riesgo'] = sum(df[col] * peso for col, peso in pesos.items())

# Puntaje = Saldo - Riesgo
df['Puntaje'] = df['SaldoCartera'] - df['Riesgo']

# Etiqueta personalizada según puntaje
def etiquetar(puntaje):
    if puntaje < 0:
        return "Alta"
    elif puntaje < 10000:
        return "Media"
    else:
        return "Baja"

df['EtiquetaRiesgo'] = df['Puntaje'].apply(etiquetar)


X = df[['Puntaje', 'SaldoCartera']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df_temp = df.copy()
df_temp['Grupo'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_temp, x='SaldoCartera', y='Puntaje', hue='Grupo', palette='viridis')
plt.title("Clustering de Empresas por Riesgo (Visualización)")
plt.xlabel("Saldo en Cartera")
plt.ylabel("Puntaje de Riesgo")
plt.legend(title="Grupo de Riesgo")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df_temp, x='Grupo', y='Puntaje', palette='pastel')
plt.title('Distribución del Puntaje por Grupo')
plt.xlabel('Grupo de Riesgo')
plt.ylabel('Puntaje')
plt.tight_layout()
plt.show()


archivo_salida = r"C:\Users\danie\OneDrive\Escritorio\bases\FinalBoss\DataBases\modified\EmpresasConClasificacionRiesgo.xlsx"
df.to_excel(archivo_salida, index=False)

print(f"\n✅ Archivo generado correctamente con {len(df)} registros (sin columna 'Grupo'):")
print(f"→ {archivo_salida}")
