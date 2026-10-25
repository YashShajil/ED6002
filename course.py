"""
Dark Store Location Optimisation
=================================

Solves the capacitated facility-location problem shown in the slides:

    minimise   sum_i sum_j d_ij * x_ij  +  sum_j F_j * y_j
    subject to sum_j x_ij == D_i                      (Constraint 1: demand satisfaction)   for all i in I
               sum_i x_ij <= C_j * y_j                 (Constraint 2: facility capacity)      for all j in J
               y_j in {0,1},  x_ij >= 0

Entities
    I : customer (demand) locations
    J : candidate dark-store locations
Parameters
    d_ij : distance from candidate j to customer i (km)
    D_i  : daily demand (orders) at customer location i
    F_j  : fixed cost of opening a dark store at location j
    C_j  : daily capacity (max orders/day) of dark store at location j
Decision variables
    y_j : 1 if a dark store is opened at location j, else 0
    x_ij: share of demand at customer i served from dark store j (orders/day)

The raw dataset (AI_for_DS_Sample_Data.csv) only contains ~21,000 individual
customer pings (lat, long, points/demand-weight) -- it does not contain
candidate dark-store sites, fixed costs or capacities, since a real facility
network isn't just "given". This script therefore:
  1. Cleans the raw data and aggregates it into a manageable number of
     demand nodes (I) using weighted K-Means clustering.
  2. Proposes a set of candidate dark-store sites (J) at demand hot-spots
     using weighted K-Means with a different number of clusters.
  3. Derives synthetic-but-reasonable capacities (C_j) and fixed costs (F_j)
     for each candidate from local demand density.
  4. Builds & solves the exact MILP shown in the slides with OR-Tools.
  5. Produces plots to visualise the network, utilisation, and costs.

All of the "derive candidates / capacities / costs" logic is clearly marked
below -- swap it out for your real site list, capacities and costs if you
have them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# --------------------------------------------------------------------------
# 0. CONFIG -- tune the size of the problem here
# --------------------------------------------------------------------------
DATA_PATH = "D:\IITM\Academics\9th Sem\AI for DS Sample Data.csv"

N_DEMAND_NODES = 150          # size of aggregated set I (customer clusters)
N_CANDIDATE_SITES = 35        # size of candidate set J (dark-store options)

CAPACITY_MIN = 500            # smallest allowed dark-store daily capacity
CAPACITY_MAX = 1500           # largest allowed dark-store daily capacity
CAPACITY_SLACK = 1.35         # ensure total candidate capacity >= this x total demand
                               # (guarantees the equality demand constraint stays feasible)

FIXED_COST_BASE = 3500.0      # base daily-equivalent fixed cost of a store
FIXED_COST_PER_CAPACITY = 2.2   # extra fixed cost per unit of capacity
FIXED_COST_NOISE = 0.15       # +/- 15% random noise on fixed cost

OUTLIER_LOW_Q, OUTLIER_HIGH_Q = 0.01, 0.99  # trim mis-geocoded pings

# --------------------------------------------------------------------------
# 1. LOAD & CLEAN THE RAW CUSTOMER DATA
# --------------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
raw = raw.rename(columns={"points": "demand"})

lat_lo, lat_hi = raw["lat"].quantile([OUTLIER_LOW_Q, OUTLIER_HIGH_Q])
lon_lo, lon_hi = raw["long"].quantile([OUTLIER_LOW_Q, OUTLIER_HIGH_Q])
clean = raw[
    raw["lat"].between(lat_lo, lat_hi) & raw["long"].between(lon_lo, lon_hi)
].copy()

print(f"Loaded {len(raw):,} raw customer pings.")
print(f"Dropped {len(raw) - len(clean):,} pings outside the "
      f"[{OUTLIER_LOW_Q:.0%}, {OUTLIER_HIGH_Q:.0%}] coordinate range as likely bad geocodes.")
print(f"Working with {len(clean):,} pings, total demand = {clean['demand'].sum():,.1f} orders/day.\n")

# --------------------------------------------------------------------------
# 2. BUILD DEMAND NODES (set I) -- weighted K-Means aggregation
# --------------------------------------------------------------------------
coords = clean[["lat", "long"]].to_numpy()
weights = clean["demand"].to_numpy()

demand_km = KMeans(n_clusters=N_DEMAND_NODES, random_state=RNG_SEED, n_init=10)
demand_labels = demand_km.fit_predict(coords, sample_weight=weights)

demand_nodes = pd.DataFrame(demand_km.cluster_centers_, columns=["lat", "long"])
demand_nodes["demand"] = (
    pd.Series(weights).groupby(demand_labels).sum().reindex(range(N_DEMAND_NODES)).fillna(0).to_numpy()
)
# drop empty clusters (can happen rarely)
demand_nodes = demand_nodes[demand_nodes["demand"] > 0].reset_index(drop=True)
demand_nodes.index.name = "i"

I = demand_nodes.index.tolist()
D = demand_nodes["demand"].to_dict()

print(f"Aggregated raw pings into {len(demand_nodes)} demand nodes "
      f"(total demand preserved = {demand_nodes['demand'].sum():,.1f}).")

# --------------------------------------------------------------------------
# 3. BUILD CANDIDATE DARK-STORE SITES (set J)
# --------------------------------------------------------------------------
site_km = KMeans(n_clusters=N_CANDIDATE_SITES, random_state=RNG_SEED + 1, n_init=10)
site_labels = site_km.fit_predict(coords, sample_weight=weights)

candidates = pd.DataFrame(site_km.cluster_centers_, columns=["lat", "long"])
local_demand = pd.Series(weights).groupby(site_labels).sum().reindex(range(N_CANDIDATE_SITES)).fillna(0)
candidates["local_demand"] = local_demand.to_numpy()
candidates.index.name = "j"

# --- derive capacity from local demand density: busier neighbourhoods get
#     bigger candidate stores, scaled into a realistic [MIN, MAX] range ---
raw_cap = candidates["local_demand"].clip(lower=1)
candidates["capacity"] = (
    CAPACITY_MIN + (raw_cap - raw_cap.min()) / (raw_cap.max() - raw_cap.min()) * (CAPACITY_MAX - CAPACITY_MIN)
).round().astype(int)

# safety net: guarantee the demand-satisfaction equality constraint can
# always be met (total candidate capacity, if all opened, >= slack x demand)
required_total = demand_nodes["demand"].sum() * CAPACITY_SLACK
if candidates["capacity"].sum() < required_total:
    scale = required_total / candidates["capacity"].sum()
    candidates["capacity"] = (candidates["capacity"] * scale).round().astype(int)

# --- derive fixed cost: bigger / busier sites cost more to run, + noise ---
noise = 1 + rng.uniform(-FIXED_COST_NOISE, FIXED_COST_NOISE, size=len(candidates))
candidates["fixed_cost"] = (
    (FIXED_COST_BASE + FIXED_COST_PER_CAPACITY * candidates["capacity"]) * noise
).round(1)

J = candidates.index.tolist()
C = candidates["capacity"].to_dict()
F = candidates["fixed_cost"].to_dict()

print(f"Proposed {len(candidates)} candidate dark-store sites.")
print(f"Total candidate capacity = {candidates['capacity'].sum():,} "
      f"vs total demand = {demand_nodes['demand'].sum():,.1f} "
      f"(x{candidates['capacity'].sum() / demand_nodes['demand'].sum():.2f} slack -> demand constraint stays feasible).\n")

# --------------------------------------------------------------------------
# 4. DISTANCE MATRIX d_ij (km, haversine)
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

d = {}
for i in I:
    for j in J:
        d[i, j] = haversine_km(
            demand_nodes.loc[i, "lat"], demand_nodes.loc[i, "long"],
            candidates.loc[j, "lat"], candidates.loc[j, "long"],
        )

# --------------------------------------------------------------------------
# 5. BUILD & SOLVE THE MILP WITH OR-TOOLS
# --------------------------------------------------------------------------
solver = pywraplp.Solver.CreateSolver("CBC")
if solver is None:
    raise RuntimeError("The OR-Tools CBC MILP solver is unavailable in this Python environment.")

# decision variables
y = {j: solver.BoolVar(f"y_{j}") for j in J}                                   # open dark store j?
x = {(i, j): solver.NumVar(0.0, solver.infinity(), f"x_{i}_{j}") for i in I for j in J}  # demand share

# Constraint 1: demand satisfaction -- sum_j x_ij = D_i, for all i
for i in I:
    solver.Add(sum(x[i, j] for j in J) == D[i], f"demand_satisfaction_{i}")

# Constraint 2: facility capacity -- sum_i x_ij <= C_j * y_j, for all j
for j in J:
    solver.Add(sum(x[i, j] for i in I) <= C[j] * y[j], f"facility_capacity_{j}")

# Objective: minimise sum d_ij*x_ij + sum F_j*y_j
solver.Minimize(
    sum(d[i, j] * x[i, j] for i in I for j in J) + sum(F[j] * y[j] for j in J)
)

status = solver.Solve()
if status != pywraplp.Solver.OPTIMAL:
    raise RuntimeError(f"The solver did not find an optimal solution (status code: {status}).")

print("Solver status: OPTIMAL")
print(f"Total cost (transport + fixed) = {solver.Objective().Value():,.2f}")

# --------------------------------------------------------------------------
# 6. EXTRACT & SUMMARISE THE SOLUTION
# --------------------------------------------------------------------------
opened = [j for j in J if y[j].solution_value() > 0.5]
candidates["opened"] = candidates.index.isin(opened)

flows = pd.DataFrame(
    [(i, j, x[i, j].solution_value()) for i in I for j in J if x[i, j].solution_value() > 1e-6],
    columns=["i", "j", "flow"],
)

total_transport_cost = float(flows.apply(lambda r: d[r.i, r.j] * r.flow, axis=1).sum())
total_fixed_cost = float(candidates.loc[opened, "fixed_cost"].sum())

utilisation = flows.groupby("j")["flow"].sum().reindex(opened).fillna(0)
candidates["served_demand"] = candidates.index.map(utilisation).fillna(0)
candidates["utilisation_pct"] = np.where(
    candidates["opened"], candidates["served_demand"] / candidates["capacity"] * 100, np.nan
)

# each demand node's "assigned" store = the one serving it the most
assigned = flows.sort_values("flow", ascending=False).drop_duplicates("i").set_index("i")["j"]
demand_nodes["assigned_site"] = demand_nodes.index.map(assigned)

print(f"\nOpened {len(opened)} of {len(candidates)} candidate sites: {opened}")
print(f"  Transport cost : {total_transport_cost:,.2f}")
print(f"  Fixed cost     : {total_fixed_cost:,.2f}")
print(f"  Demand served  : {flows['flow'].sum():,.1f} / {demand_nodes['demand'].sum():,.1f}")

print("\nOpened-site summary:")
print(
    candidates.loc[opened, ["lat", "long", "capacity", "served_demand", "utilisation_pct", "fixed_cost"]]
    .sort_values("served_demand", ascending=False)
    .round(1)
)

# --------------------------------------------------------------------------
# 7. PLOTS
# --------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ---- 7a. Network map: demand nodes, candidate sites, and assignment lines
fig, ax = plt.subplots(figsize=(9, 8))

# faint assignment lines demand-node -> its opened store
cmap = plt.get_cmap("tab20", len(opened))
site_color = {j: cmap(k) for k, j in enumerate(opened)}

for i, row in demand_nodes.dropna(subset=["assigned_site"]).iterrows():
    j = int(row["assigned_site"])
    ax.plot(
        [row["long"], candidates.loc[j, "long"]],
        [row["lat"], candidates.loc[j, "lat"]],
        color=site_color[j], alpha=0.15, linewidth=0.6, zorder=1,
    )

# demand nodes, sized by demand, colored by assigned store
node_colors = [site_color.get(int(s), "grey") if pd.notna(s) else "grey"
               for s in demand_nodes["assigned_site"]]
ax.scatter(
    demand_nodes["long"], demand_nodes["lat"],
    s=demand_nodes["demand"] * 3 + 5, c=node_colors,
    alpha=0.8, edgecolors="white", linewidths=0.3, zorder=2, label="Demand node",
)

# candidate sites: opened (star) vs not opened (x)
closed = candidates[~candidates["opened"]]
ax.scatter(closed["long"], closed["lat"], marker="x", s=70, c="black",
           linewidths=1.3, zorder=3, label="Candidate site (not opened)")
ax.scatter(
    candidates.loc[opened, "long"], candidates.loc[opened, "lat"],
    marker="*", s=candidates.loc[opened, "capacity"] * 0.9 + 120,
    c=[site_color[j] for j in opened], edgecolors="black", linewidths=1.0,
    zorder=4, label="Opened dark store",
)

ax.set_title("Dark Store Network: Opened Sites & Demand Assignment")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markersize=8, label="Demand node (size = demand)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=14, label="Opened dark store (size = capacity)"),
    Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=8, label="Candidate site (not opened)"),
]
ax.legend(handles=legend_elems, loc="best", frameon=True)
fig.tight_layout()
fig.savefig("plot_1_network_map.png")

# ---- 7b. Facility utilisation bar chart
fig, ax = plt.subplots(figsize=(9, 5))
util = candidates.loc[opened].sort_values("utilisation_pct", ascending=False)
bars = ax.bar(util.index.astype(str), util["utilisation_pct"], color=[site_color[j] for j in util.index])
ax.axhline(100, color="red", linestyle="--", linewidth=1, label="Capacity limit")
ax.set_xlabel("Candidate site ID")
ax.set_ylabel("Utilisation (%)")
ax.set_title("Capacity Utilisation of Opened Dark Stores")
ax.set_ylim(0, max(105, util["utilisation_pct"].max() * 1.1))
for b, (_, row) in zip(bars, util.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"{row['served_demand']:.0f}/{row['capacity']:.0f}",
            ha="center", va="bottom", fontsize=7)
ax.legend()
fig.tight_layout()
fig.savefig("plot_2_utilisation.png")

# ---- 7c. Cost breakdown
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].pie(
    [total_transport_cost, total_fixed_cost],
    labels=[f"Transport cost\n{total_transport_cost:,.0f}", f"Fixed cost\n{total_fixed_cost:,.0f}"],
    autopct="%1.1f%%", startangle=90, colors=["#4C72B0", "#DD8452"],
)
axes[0].set_title("Total Cost Breakdown")

axes[1].bar(candidates.loc[opened].sort_values("fixed_cost", ascending=False).index.astype(str),
            candidates.loc[opened].sort_values("fixed_cost", ascending=False)["fixed_cost"],
            color="#DD8452")
axes[1].set_xlabel("Candidate site ID")
axes[1].set_ylabel("Fixed cost")
axes[1].set_title("Fixed Cost by Opened Site")

fig.tight_layout()
fig.savefig("plot_3_cost_breakdown.png")

# ---- 7d. Candidate sites: opened vs not opened, cost/capacity trade-off
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(closed["capacity"], closed["fixed_cost"], marker="x", s=70, c="black", label="Not opened")
ax.scatter(candidates.loc[opened, "capacity"], candidates.loc[opened, "fixed_cost"],
           marker="*", s=180, c=[site_color[j] for j in opened], edgecolors="black", label="Opened")
for j in candidates.index:
    ax.annotate(str(j), (candidates.loc[j, "capacity"], candidates.loc[j, "fixed_cost"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Candidate capacity (orders/day)")
ax.set_ylabel("Fixed cost")
ax.set_title("Candidate Sites: Capacity vs Fixed Cost")
ax.legend()
fig.tight_layout()
fig.savefig("plot_4_candidate_tradeoff.png")

print("\nSaved plots: plot_1_network_map.png, plot_2_utilisation.png, "
      "plot_3_cost_breakdown.png, plot_4_candidate_tradeoff.png")