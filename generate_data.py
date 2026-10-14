"""Generate a realistic SYNTHETIC football dataset (per-90 stats) for the project.

Why synthetic? Real data (FBref / Kaggle) needs manual download. This script lets the
project run out-of-the-box. To use real data, export a CSV with the same columns
(see FEATURES in scout.py) and save it as data/players.csv.
"""
import numpy as np, pandas as pd

COLS = ["goals","shots","shots_on_target","xg","assists","xa","key_passes","passes_completed",
        "pass_accuracy","progressive_passes","progressive_carries","dribbles","tackles",
        "interceptions","clearances","blocks","aerials_won","touches","recoveries","crosses"]

# archetype -> (position, mean per-90 values in COLS order)
ARCH = {
 "Poacher":        ("FW",[.65,3.2,1.5,.6,.10,.10,.8,18,72,1.5,1.0,.8,.4,.2,.3,.2,1.2,35,3,.3]),
 "Target Man":     ("FW",[.50,2.6,1.1,.5,.12,.10,.7,17,70,1.2,.5,.4,.5,.3,.8,.3,4.5,32,3.5,.5]),
 "Winger":         ("FW",[.35,2.5,1.0,.3,.30,.25,2.0,28,78,3.5,3.8,3.2,.9,.5,.4,.3,.7,50,4.5,2.5]),
 "Playmaker":      ("MF",[.15,1.4,.5,.15,.30,.30,2.6,55,86,7.0,2.0,1.6,1.4,.9,.6,.4,.6,75,5.5,1.2]),
 "Box-to-Box":     ("MF",[.15,1.3,.5,.12,.15,.12,1.2,45,84,4.5,2.8,1.3,2.6,1.3,1.0,.7,1.6,68,7.5,.6]),
 "Defensive Mid":  ("MF",[.04,.7,.2,.04,.06,.06,.7,52,88,4.0,1.0,.6,3.0,1.8,1.6,1.0,2.2,70,8.5,.3]),
 "Ball-Playing CB":("DF",[.03,.5,.15,.03,.04,.04,.4,68,91,6.5,1.3,.3,1.6,1.3,3.6,1.2,2.8,80,6.0,.1]),
 "Stopper CB":     ("DF",[.04,.5,.2,.04,.02,.02,.2,40,80,1.5,.4,.2,1.8,1.4,5.5,1.8,4.8,52,6.5,.05]),
 "Full-Back":      ("DF",[.05,.8,.3,.05,.18,.15,1.4,42,82,4.0,2.8,1.4,2.4,1.4,2.2,1.1,1.5,62,6.5,2.2]),
}
FIRST = "Luca Marco Diego Mateo Jonas Ivan Omar Kofi Hugo Rafa Leo Nico Sami Tariq Yuki Andre Felix Pablo Erik Bruno Kian Ravi Arjun Tomas Jules Theo Enzo Idris Mason".split()
LAST = "Silva Costa Rossi Weber Novak Haddad Okafor Mendes Ferrari Larsen Petrov Santos Moreau Khan Ito Vega Duarte Kovacs Brandt Ortiz Sato Nasser Bauer Lopez Reyes Ricci Diallo Horvat Fischer Almeida".split()
CLUBS = [f"{c} FC" for c in "Northgate Riverside Eastport Highland Redwood Bluebay Ironfield Stonebridge Westmere Goldcoast Silverlake Oakham Kingsway Harborne Falcon Crestview Lakeside Ashford Valemont Sunhaven".split()]

def generate(n=900, seed=42):
    rng = np.random.default_rng(seed)
    names = list(ARCH)
    rows = []
    for i in range(n):
        a = names[i % len(names)]
        pos, mu = ARCH[a][0], np.array(ARCH[a][1], float)
        if rng.random() < .25:                       # hybrid players blur archetype borders
            b = names[rng.integers(len(names))]; w = rng.uniform(.15,.35)
            mu = (1-w)*mu + w*np.array(ARCH[b][1], float)
        quality = rng.normal(1, .18)                  # overall ability
        v = mu * quality * rng.lognormal(0, .22, len(mu))
        d = dict(zip(COLS, v))
        d["pass_accuracy"] = float(np.clip(mu[8] + rng.normal(0,3.5), 55, 97))
        d["shots_on_target"] = min(d["shots_on_target"], d["shots"])
        age = int(np.clip(rng.normal(26, 4.5), 17, 38))
        mins = int(rng.uniform(900, 3400))
        value = max(.3, quality**4 * 12 * (1.25 if age < 24 else 1 if age < 30 else .6) * rng.lognormal(0,.3))
        rows.append(dict(player=f"{rng.choice(FIRST)} {rng.choice(LAST)}", club=rng.choice(CLUBS),
                         position=pos, age=age, minutes=mins, market_value_m=round(value,1),
                         true_style=a, **{k: round(float(x),3) for k,x in d.items()}))
    df = pd.DataFrame(rows)
    df["player"] = df["player"] + df.groupby("player").cumcount().map(lambda c: "" if c==0 else f" {c+1}")
    return df

if __name__ == "__main__":
    df = generate(); df.to_csv("data/players.csv", index=False)
    print(df.shape); print(df.position.value_counts().to_dict())
