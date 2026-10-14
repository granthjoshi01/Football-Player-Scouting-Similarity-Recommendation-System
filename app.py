import streamlit as st, pandas as pd, plotly.express as px, plotly.graph_objects as go
from scout import ScoutingEngine, load_data, FEATURES, BASE

st.set_page_config(page_title="Football Scout AI", page_icon="⚽", layout="wide")

@st.cache_resource
def get_engine(file):
    df = pd.read_csv(file) if file else load_data()
    return ScoutingEngine(df)

up = st.sidebar.file_uploader("Optional: upload your own players CSV", type="csv")
E = get_engine(up)
df = E.df
st.sidebar.title("⚽ Football Scout AI")
page = st.sidebar.radio("Mode", ["🏠 Overview", "👤 Player Profile & Similar Players", "🔍 Scouting Mode", "🧠 Model Insights"])

SHOW = ["player","club","position","age","style","market_value_m"]
RADAR = ["goals","assists","key_passes","progressive_passes","dribbles","tackles","interceptions","aerials_won"]

def radar(names):
    fig = go.Figure()
    for n in names:
        i = df.index[df.player == n][0]
        fig.add_trace(go.Scatterpolar(r=E.pct.loc[i, RADAR].values, theta=RADAR, fill="toself", name=n))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=420,
                      title="Percentile vs players in same position")
    return fig

# ---------------- OVERVIEW ----------------
if page.startswith("🏠"):
    st.title("Football Player Scouting & Similarity Recommender")
    c = st.columns(4)
    c[0].metric("Players", len(df)); c[1].metric("Playing styles (K-Means)", E.k)
    c[2].metric("Role classifier accuracy", f"{E.rf_acc:.1%}"); c[3].metric("PCA components", E.pca.n_components_)
    fig = px.scatter(df, x="pc1", y="pc2", color="style", hover_name="player",
                     hover_data=["position","age","club"], title="Player map (PCA 2D) coloured by discovered playing style")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df[SHOW + ["minutes"]], width="stretch", height=300)

# ---------------- PLAYER PROFILE ----------------
elif page.startswith("👤"):
    st.title("Player Profile & Similar Players")
    name = st.selectbox("Select a player", sorted(df.player))
    p = df[df.player == name].iloc[0]
    l, r = st.columns([1, 1.2])
    with l:
        st.subheader(name); st.write(f"**{p.club}** · {p.position} · age {p.age} · €{p.market_value_m}M")
        st.write(f"**Playing style:** {p['style']}")
        probs = E.predict_role(name)
        st.write("**Random Forest role probabilities:**")
        st.bar_chart(pd.Series(probs))
    with r:
        st.plotly_chart(radar([name]), width="stretch")
    st.divider(); st.subheader("Similar players")
    a, b, c2, d = st.columns(4)
    n = a.slider("How many", 3, 20, 10); method = b.selectbox("Algorithm", ["cosine", "knn"])
    same = c2.checkbox("Same position only", True); mx = d.number_input("Max value (€M, 0 = any)", 0.0, 200.0, 0.0)
    rec = E.similar(name, n, method, same, max_value=mx or None)
    st.dataframe(rec[SHOW + ["similarity"]].style.format({"similarity": "{:.1f}"}), width="stretch")
    pick = st.multiselect("Compare on radar", rec.player.tolist(), default=rec.player.tolist()[:2])
    if pick: st.plotly_chart(radar([name] + pick), width="stretch")

# ---------------- SCOUTING ----------------
elif page.startswith("🔍"):
    st.title("Scouting Mode")
    with st.form("scout"):
        a, b = st.columns(2)
        pos = a.multiselect("Position", ["FW", "MF", "DF"], ["FW"])
        age = a.slider("Age range", 16, 40, (18, 25))
        mins = b.number_input("Minimum minutes played", 0, 3400, 900)
        val = b.number_input("Max market value (€M, 0 = any)", 0.0, 200.0, 0.0)
        attrs = st.multiselect("Desired attributes", FEATURES, ["dribbles", "progressive_carries", "goals"])
        weights = {x: st.slider(f"Importance: {x}", 1, 5, 3, key=x) for x in attrs}
        n = st.slider("Results", 3, 25, 10)
        go_ = st.form_submit_button("Find players")
    if go_:
        res = E.scout(pos, age, mins, val or None, weights, n)
        if res.empty: st.warning("No players match these filters.")
        else:
            st.dataframe(res[SHOW + ["minutes", "fit_score"]].style.format({"fit_score": "{:.1f}"}), width="stretch")
            if weights:
                st.plotly_chart(px.bar(res, x="fit_score", y="player", orientation="h", title="Fit score (weighted in-role percentile)")
                                .update_yaxes(autorange="reversed"), width="stretch")

# ---------------- MODEL INSIGHTS ----------------
else:
    st.title("Model Insights")
    t1, t2, t3, t4 = st.tabs(["PCA", "K-Means", "Random Forest", "Recommender evaluation"])
    with t1:
        ev = E.pca_full.explained_variance_ratio_.cumsum()
        st.plotly_chart(px.line(x=range(1, len(ev) + 1), y=ev, markers=True, labels=dict(x="components", y="cumulative variance"),
                                title=f"{E.pca.n_components_} components keep ≥90% variance"), width="stretch")
    with t2:
        ks = pd.Series(E.k_scores)
        st.plotly_chart(px.line(x=ks.index, y=ks.values, markers=True, labels=dict(x="k", y="silhouette"),
                                title=f"Silhouette score → best k = {E.k}"), width="stretch")
        st.write("Average z-score profile of each style:")
        prof = E.cluster_profile.rename(index=E.cluster_names)
        st.plotly_chart(px.imshow(prof[BASE], aspect="auto", color_continuous_scale="RdBu_r", zmin=-2, zmax=2), width="stretch")
    with t3:
        st.metric("Test accuracy", f"{E.rf_acc:.1%}")
        st.plotly_chart(px.imshow(pd.DataFrame(E.rf_cm, index=E.classes, columns=E.classes), text_auto=True,
                                  labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"), width="stretch")
        st.plotly_chart(px.bar(E.importances.head(12)[::-1], orientation="h", title="Top feature importances"), width="stretch")
    with t4:
        k = st.slider("k", 3, 20, 10)
        res = pd.DataFrame({"Precision@k": {m: E.precision_at_k(k, m) for m in ["cosine", "knn"]}})
        st.dataframe(res.style.format("{:.1%}"))
        st.caption("Precision@k = share of the top-k recommended players that share the query player's ground-truth style.")
