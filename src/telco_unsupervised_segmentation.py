import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Numeric features for clustering
feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
num = df[feature_cols].dropna().copy()

scaler = StandardScaler()
X_full = scaler.fit_transform(num)

# Compare model variations on a representative sample
rng = np.random.RandomState(42)
sample_idx = rng.choice(X_full.shape[0], 1500, replace=False)
X = X_full[sample_idx]

results = []

for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(X)
    results.append({
        "Model": f"K-Means (k={k})",
        "Silhouette": round(float(silhouette_score(X, labels)), 3),
    })

for k in [3, 4]:
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = ag.fit_predict(X)
    results.append({
        "Model": f"Hierarchical (Ward, k={k})",
        "Silhouette": round(float(silhouette_score(X, labels)), 3),
    })

for eps in [0.5, 0.7]:
    db = DBSCAN(eps=eps, min_samples=20)
    labels = db.fit_predict(X)
    mask = labels != -1
    valid_clusters = len(set(labels[mask]))
    sil = ""
    if valid_clusters >= 2:
        sil = round(float(silhouette_score(X[mask], labels[mask])), 3)
    results.append({
        "Model": f"DBSCAN (eps={eps})",
        "Silhouette": sil,
    })

comparison_df = pd.DataFrame(results)
print("\nModel comparison:")
print(comparison_df)

# Final model: K-Means with 3 clusters
km_final = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=100)
full_labels = km_final.fit_predict(X_full)

analysis_df = df.loc[num.index, ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Contract", "InternetService", "PaymentMethod", "Churn"]].copy()
analysis_df["Cluster"] = full_labels

summary_df = analysis_df.groupby("Cluster").agg(
    customers=("Cluster", "size"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charges=("MonthlyCharges", "mean"),
    avg_total_charges=("TotalCharges", "mean"),
    pct_senior=("SeniorCitizen", "mean"),
    churn_rate=("Churn", lambda s: (s == "Yes").mean())
).round(2)

modes_df = analysis_df.groupby("Cluster").agg(
    top_contract=("Contract", lambda s: s.mode().iat[0]),
    top_internet=("InternetService", lambda s: s.mode().iat[0]),
    top_payment=("PaymentMethod", lambda s: s.mode().iat[0]),
)

cluster_profiles = summary_df.join(modes_df)
print("\nCluster profiles:")
print(cluster_profiles)

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_full)

plt.figure(figsize=(7.5, 5.5))
for cluster_id in sorted(np.unique(full_labels)):
    mask = full_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, alpha=0.5, label=f"Cluster {cluster_id}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer segments from K-Means (k=3)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("telco_clusters_pca.png", dpi=180)
plt.show()
