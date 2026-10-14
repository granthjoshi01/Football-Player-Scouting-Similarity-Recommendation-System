"""Core ML + Recommender logic for the Football Scouting system."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

BASE = ["goals","shots","shots_on_target","xg","assists","xa","key_passes","passes_completed",
        "pass_accuracy","progressive_passes","progressive_carries","dribbles","tackles",
        "interceptions","clearances","blocks","aerials_won","touches","recoveries","crosses"]
ENGINEERED = ["goal_conversion","shot_accuracy","finishing_vs_xg","attacking_output",
              "defensive_actions","progression","creativity"]
FEATURES = BASE + ENGINEERED

def load_data(path="data/players.csv"):
    return pd.read_csv(path)

# ---------- 1. FEATURE ENGINEERING ----------
def engineer_features(df):
    d = df.copy()
    d["goal_conversion"]   = d.goals / d.shots.replace(0, np.nan)
    d["shot_accuracy"]     = d.shots_on_target / d.shots.replace(0, np.nan)
    d["finishing_vs_xg"]   = d.goals - d.xg
    d["attacking_output"]  = d.goals + d.assists
    d["defensive_actions"] = d.tackles + d.interceptions + d.clearances + d.blocks
    d["progression"]       = d.progressive_passes + d.progressive_carries
    d["creativity"]        = d.key_passes + 2 * d.xa
    d[["goal_conversion","shot_accuracy"]] = d[["goal_conversion","shot_accuracy"]].fillna(0).clip(0, 1)
    return d.dropna(subset=BASE).reset_index(drop=True)

class ScoutingEngine:
    def __init__(self, df, k_range=range(4, 11), pca_variance=0.90, seed=42):
        self.seed = seed
        self.df = engineer_features(df)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[FEATURES])           # z-scored features
        self._fit_pca(pca_variance)
        self._fit_clusters(k_range)
        self._fit_classifier()
        self.pct = self.df.groupby("position")[FEATURES].rank(pct=True) * 100  # percentile in role
        self._knn = NearestNeighbors(metric="euclidean").fit(self.X)

    # ---------- 2. PCA ----------
    def _fit_pca(self, var):
        self.pca_full = PCA().fit(self.X)
        self.pca = PCA(n_components=var, random_state=self.seed).fit(self.X)
        self.Z = self.pca.transform(self.X)
        self.pca2 = PCA(n_components=2, random_state=self.seed).fit(self.X)
        self.coords2d = self.pca2.transform(self.X)
        self.df["pc1"], self.df["pc2"] = self.coords2d[:,0], self.coords2d[:,1]

    # ---------- 3. K-MEANS (playing styles) ----------
    def _fit_clusters(self, k_range):
        self.k_scores = {k: silhouette_score(self.Z, KMeans(k, n_init=10, random_state=self.seed).fit_predict(self.Z))
                         for k in k_range}
        self.k = max(self.k_scores, key=self.k_scores.get)
        self.kmeans = KMeans(self.k, n_init=10, random_state=self.seed).fit(self.Z)
        self.df["cluster"] = self.kmeans.labels_
        z = pd.DataFrame(self.X, columns=FEATURES); z["cluster"] = self.df.cluster
        self.cluster_profile = z.groupby("cluster")[FEATURES].mean()
        names = {}
        for c, row in self.cluster_profile.iterrows():
            top = row.sort_values(ascending=False).index[:2]
            pos = self.df[self.df.cluster == c].position.mode()[0]
            names[c] = f"{pos} · " + " + ".join(t.replace("_"," ") for t in top)
        self.cluster_names = names
        self.df["style"] = self.df.cluster.map(names)

    # ---------- 4. RANDOM FOREST (role classification) ----------
    def _fit_classifier(self):
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.df.position, test_size=.25,
                                              stratify=self.df.position, random_state=self.seed)
        self.rf = RandomForestClassifier(n_estimators=300, random_state=self.seed, n_jobs=-1).fit(Xtr, ytr)
        pred = self.rf.predict(Xte)
        self.classes = list(self.rf.classes_)
        self.rf_acc = accuracy_score(yte, pred)
        self.rf_cm = confusion_matrix(yte, pred, labels=self.classes)
        self.rf_report = classification_report(yte, pred)
        self.importances = pd.Series(self.rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    def predict_role(self, name):
        i = self._idx(name)
        p = self.rf.predict_proba(self.X[[i]])[0]
        return dict(zip(self.classes, p))

    # ---------- 5. CONTENT-BASED RECOMMENDER ----------
    def _idx(self, name):
        return int(self.df.index[self.df.player == name][0])

    def similar(self, name, n=10, method="cosine", same_position=False, max_age=None, max_value=None):
        """Top-N most similar players. method: 'cosine' | 'knn'."""
        i = self._idx(name)
        if method == "cosine":
            score = cosine_similarity(self.X[[i]], self.X)[0]
        else:  # KNN: euclidean distance -> similarity in (0,1]
            dist, _ = self._knn.kneighbors(self.X[[i]], n_neighbors=len(self.df))
            d = np.empty(len(self.df)); d[_[0]] = dist[0]
            score = 1 / (1 + d)
        out = self.df.assign(similarity=score * 100).drop(index=i)
        if same_position: out = out[out.position == self.df.loc[i, "position"]]
        if max_age is not None: out = out[out.age <= max_age]
        if max_value is not None: out = out[out.market_value_m <= max_value]
        return out.sort_values("similarity", ascending=False).head(n)

    # ---------- 6. SCOUTING MODE (requirement-based) ----------
    def scout(self, positions=None, age_range=(16, 40), min_minutes=0, max_value=None,
              attributes=None, n=10):
        """attributes: {feature: weight}. Score = weighted mean of in-role percentiles."""
        attributes = attributes or {}
        m = self.df.age.between(*age_range) & (self.df.minutes >= min_minutes)
        if positions: m &= self.df.position.isin(positions)
        if max_value is not None: m &= self.df.market_value_m <= max_value
        cand = self.df[m].copy()
        if cand.empty: return cand
        if attributes:
            w = pd.Series(attributes, dtype=float)
            cand["fit_score"] = (self.pct.loc[cand.index, w.index] * w).sum(axis=1) / w.sum()
        else:
            cand["fit_score"] = 0.0
        return cand.sort_values("fit_score", ascending=False).head(n)

    # ---------- 7. EVALUATION OF THE RECOMMENDER ----------
    def precision_at_k(self, k=10, method="cosine", sample=200):
        """Fraction of the top-k recommendations sharing the query's ground-truth style."""
        label = "true_style" if "true_style" in self.df else "cluster"
        idx = np.random.default_rng(self.seed).choice(len(self.df), min(sample, len(self.df)), replace=False)
        hits = []
        for i in idx:
            rec = self.similar(self.df.player[i], n=k, method=method)
            hits.append((rec[label] == self.df.loc[i, label]).mean())
        return float(np.mean(hits))
