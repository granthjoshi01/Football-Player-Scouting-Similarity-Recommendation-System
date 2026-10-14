# ⚽ Football Player Scouting & Similarity Recommendation System

One project covering **Machine Learning** and **Recommender Systems**.

## Run it
```bash
pip install -r requirements.txt
python generate_data.py        # creates data/players.csv (synthetic, 900 players)
streamlit run app.py
```
**Using real data:** export per-90 stats (e.g. from FBref / Kaggle "Football Players Stats") to a CSV with the
columns in `scout.py → BASE` plus `player, club, position (FW/MF/DF), age, minutes, market_value_m`.
Save as `data/players.csv` or upload it in the app sidebar. (`true_style` is optional.)

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic dataset of 9 player archetypes (Poacher, Winger, Playmaker, Ball-Playing CB...) |
| `scout.py` | All ML + recommender logic (`ScoutingEngine`) |
| `app.py` | Streamlit dashboard (4 modes) |

## Pipeline → which subject it counts for
| Step | Where | Subject |
|---|---|---|
| Feature engineering (goal conversion, shot accuracy, finishing vs xG, defensive actions, progression, creativity) | `engineer_features` | ML |
| Standardisation (z-score) | `StandardScaler` | ML + RS |
| PCA (keep 90% variance) + 2-D map | `_fit_pca` | ML |
| K-Means, k chosen by silhouette → playing styles | `_fit_clusters` | ML |
| Random Forest: classify position (FW/MF/DF), feature importance, confusion matrix | `_fit_classifier` | ML |
| Content-based filtering: player = feature vector | `similar` | RS |
| Cosine similarity and KNN, Top-N list, filters | `similar` | RS |
| Requirement-based scouting (weighted percentile score) | `scout` | RS |
| Precision@k evaluation (cosine vs KNN) | `precision_at_k` | RS |

## Results on the included dataset
- Random Forest position accuracy ≈ **97%**
- K-Means best k = 6 (silhouette ≈ 0.32), PCA: 20+7 features → 9 components
- Precision@10: cosine ≈ **94%**, KNN ≈ **94%**

## Report / viva tips
1. **Why cosine?** It compares the *shape* of a profile, ignoring magnitude. Combined with z-scoring it's a fair style comparison.
2. **Why cosine vs KNN gives similar results:** after standardising, both use the same feature space; cosine ignores length, Euclidean KNN doesn't.
3. **Cold start / limits:** content-based filtering needs no user history but can't recommend based on tastes of other users, only on stats.
4. **Synthetic data caveat:** state clearly that results are high partly because the data is generated from clean archetypes; with real FBref data expect lower accuracy (~80-90%).
5. **Possible extensions:** weight features per role, add GK, hybrid score (similarity + budget), SHAP for explanations, t-SNE/UMAP.
