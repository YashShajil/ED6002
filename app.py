# -*- coding: utf-8 -*-
"""
Dark Store Location Optimisation – Streamlit App
=================================================

Enhanced MILP model (based on course.py) with:
  • Maximum-distance constraint between dark store and customer
  • Interactive Streamlit UI with inputs for:
      - number of customer demand nodes (aggregated)
      - number of candidate dark-store sites
      - maximum allowed distance (km)
  • Results:
      1. KPI summary table
      2. Interactive Folium map
      3. Profit vs. allowed-open-stores trade-off plot (for 10 000 customers
         and 50 candidate sites, at k ∈ {2, 5, 10, 20, 25})

Geography: Chennai Metropolitan Area (lat 12.78–13.23, lon 80.02–80.35)

Assumptions (all clearly stated in the UI):
  • Revenue per order delivered = ₹80
  • Transport (delivery) cost rate = ₹12 per km per order
  • Daily-equivalent fixed cost of opening a store derived from local demand
    density (same formula as course.py)
  • Capacity of each candidate site derived from local demand density
  • Demand (orders/day) at each node = proportional to number of customer pings
"""

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp
import io

# ─────────────────────────────────────────────────────────────────────────────
# Constants / assumptions
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42

# Chennai bounding box
CHENNAI_LAT_MIN, CHENNAI_LAT_MAX = 12.78, 13.23
CHENNAI_LON_MIN, CHENNAI_LON_MAX = 80.02, 80.35

REVENUE_PER_ORDER = 80.0        # ₹ per order delivered
TRANSPORT_COST_PER_KM = 12.0    # ₹ per km per order (last-mile delivery)

CAPACITY_MIN = 300              # orders/day
CAPACITY_MAX = 1500             # orders/day
CAPACITY_SLACK = 1.35           # ensure total capacity ≥ slack × total demand

FIXED_COST_BASE = 3500.0        # ₹/day base fixed cost
FIXED_COST_PER_CAPACITY = 2.2   # ₹/day per unit capacity
FIXED_COST_NOISE = 0.10         # ±10% noise


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def generate_chennai_data(n_customers: int, n_candidates: int, seed: int = RNG_SEED):
    """
    Synthetically generate customer pings uniformly within the Chennai bounding
    box, then aggregate into demand nodes and candidate dark-store sites.
    Returns (demand_nodes, candidates, d_matrix) DataFrames / dict.
    """
    rng = np.random.default_rng(seed)

    # --- synthetic customer pings (uniform over Chennai bbox) ---
    lats = rng.uniform(CHENNAI_LAT_MIN, CHENNAI_LAT_MAX, n_customers)
    lons = rng.uniform(CHENNAI_LON_MIN, CHENNAI_LON_MAX, n_customers)
    demands = rng.integers(1, 10, n_customers).astype(float)

    coords = np.column_stack([lats, lons])

    # --- aggregate into demand nodes ---
    n_nodes = min(n_customers, max(20, n_customers // 100))
    km_d = KMeans(n_clusters=n_nodes, random_state=seed, n_init=10)
    labels_d = km_d.fit_predict(coords, sample_weight=demands)
    demand_nodes = pd.DataFrame(km_d.cluster_centers_, columns=["lat", "lon"])
    demand_nodes["demand"] = pd.Series(demands).groupby(labels_d).sum().reindex(range(n_nodes)).fillna(0).values
    demand_nodes = demand_nodes[demand_nodes["demand"] > 0].reset_index(drop=True)
    demand_nodes.index.name = "i"

    # --- candidate sites ---
    km_c = KMeans(n_clusters=n_candidates, random_state=seed + 1, n_init=10)
    labels_c = km_c.fit_predict(coords, sample_weight=demands)
    candidates = pd.DataFrame(km_c.cluster_centers_, columns=["lat", "lon"])
    local_dem = pd.Series(demands).groupby(labels_c).sum().reindex(range(n_candidates)).fillna(0)
    candidates["local_demand"] = local_dem.values
    candidates.index.name = "j"

    # capacity
    raw_cap = candidates["local_demand"].clip(lower=1)
    candidates["capacity"] = (
        CAPACITY_MIN + (raw_cap - raw_cap.min()) / (raw_cap.max() - raw_cap.min() + 1e-9) * (CAPACITY_MAX - CAPACITY_MIN)
    ).round().astype(int)

    # safety slack
    total_demand = demand_nodes["demand"].sum()
    required = total_demand * CAPACITY_SLACK
    if candidates["capacity"].sum() < required:
        scale = required / candidates["capacity"].sum()
        candidates["capacity"] = (candidates["capacity"] * scale).round().astype(int)

    # fixed cost
    noise = 1 + rng.uniform(-FIXED_COST_NOISE, FIXED_COST_NOISE, size=len(candidates))
    candidates["fixed_cost"] = (
        (FIXED_COST_BASE + FIXED_COST_PER_CAPACITY * candidates["capacity"]) * noise
    ).round(1)

    # distance matrix
    I = demand_nodes.index.tolist()
    J = candidates.index.tolist()
    d = {}
    for i in I:
        for j in J:
            d[i, j] = haversine_km(
                demand_nodes.loc[i, "lat"], demand_nodes.loc[i, "lon"],
                candidates.loc[j, "lat"], candidates.loc[j, "lon"],
            )

    return demand_nodes, candidates, d


def solve_milp(demand_nodes, candidates, d, max_dist_km, max_open=None):
    """
    Solve the capacitated facility-location MILP with the added max-distance
    constraint.  Returns (status_str, solution_dict_or_None).

    Enhanced model:
        minimise   Σ_i Σ_j  transport_cost_per_km * d_ij * x_ij  +  Σ_j F_j * y_j
        subject to
            Σ_j x_ij == D_i                              (demand satisfaction)
            Σ_i x_ij <= C_j * y_j                        (capacity)
            x_ij <= D_i * z_ij                           (link flow to binary indicator)
            z_ij <= y_j                                  (only assign to open stores)
            z_ij = 0 if d_ij > max_dist_km               (MAX DISTANCE CONSTRAINT)
            Σ_j y_j <= max_open   (optional, for trade-off plot)
            y_j ∈ {0,1},  x_ij >= 0,  z_ij ∈ {0,1}

    Profit = Revenue - Transport cost - Fixed cost
    """
    I = demand_nodes.index.tolist()
    J = candidates.index.tolist()
    D = demand_nodes["demand"].to_dict()
    C = candidates["capacity"].to_dict()
    F = candidates["fixed_cost"].to_dict()

    # feasibility pre-check: for each demand node, at least one candidate must
    # be within max_dist_km; otherwise the problem is infeasible
    reachable = {i: [j for j in J if d[i, j] <= max_dist_km] for i in I}
    infeasible_nodes = [i for i in I if not reachable[i]]
    if infeasible_nodes:
        return "INFEASIBLE", None

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return "ERROR", None

    infinity = solver.infinity()

    # decision variables
    y = {j: solver.BoolVar(f"y_{j}") for j in J}
    x = {(i, j): solver.NumVar(0.0, infinity, f"x_{i}_{j}") for i in I for j in J}

    # Constraint 1: demand satisfaction
    for i in I:
        solver.Add(sum(x[i, j] for j in J) == D[i])

    # Constraint 2: capacity
    for j in J:
        solver.Add(sum(x[i, j] for i in I) <= C[j] * y[j])

    # Constraint 3 (NEW): max distance — zero out flow if distance exceeds limit
    for i in I:
        for j in J:
            if d[i, j] > max_dist_km:
                solver.Add(x[i, j] == 0.0)

    # Constraint 4 (optional): limit number of open stores
    if max_open is not None:
        solver.Add(sum(y[j] for j in J) <= max_open)

    # Objective: minimise transport + fixed costs
    transport_obj = sum(TRANSPORT_COST_PER_KM * d[i, j] * x[i, j] for i in I for j in J)
    fixed_obj = sum(F[j] * y[j] for j in J)
    solver.Minimize(transport_obj + fixed_obj)

    solver.SetTimeLimit(120_000)  # 2-minute time limit
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return "INFEASIBLE", None

    status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE (time limit)"

    opened = [j for j in J if y[j].solution_value() > 0.5]
    flows = [
        (i, j, x[i, j].solution_value())
        for i in I for j in J
        if x[i, j].solution_value() > 1e-6
    ]
    flows_df = pd.DataFrame(flows, columns=["i", "j", "flow"])

    total_transport_cost = float(
        flows_df.apply(lambda r: TRANSPORT_COST_PER_KM * d[int(r.i), int(r.j)] * r.flow, axis=1).sum()
    )
    total_fixed_cost = float(candidates.loc[opened, "fixed_cost"].sum())
    total_revenue = float(demand_nodes["demand"].sum() * REVENUE_PER_ORDER)
    profit = total_revenue - total_transport_cost - total_fixed_cost

    utilisation = flows_df.groupby("j")["flow"].sum().reindex(opened).fillna(0)
    candidates_copy = candidates.copy()
    candidates_copy["opened"] = candidates_copy.index.isin(opened)
    candidates_copy["served_demand"] = candidates_copy.index.map(utilisation).fillna(0)
    candidates_copy["utilisation_pct"] = np.where(
        candidates_copy["opened"],
        candidates_copy["served_demand"] / candidates_copy["capacity"] * 100,
        np.nan,
    )

    assigned = (
        flows_df.sort_values("flow", ascending=False)
        .drop_duplicates("i")
        .set_index("i")["j"]
    )
    demand_nodes_copy = demand_nodes.copy()
    demand_nodes_copy["assigned_site"] = demand_nodes_copy.index.map(assigned)

    sol = {
        "status": status_str,
        "opened": opened,
        "flows_df": flows_df,
        "total_transport_cost": total_transport_cost,
        "total_fixed_cost": total_fixed_cost,
        "total_revenue": total_revenue,
        "profit": profit,
        "candidates": candidates_copy,
        "demand_nodes": demand_nodes_copy,
        "d": d,
        "n_stores": len(opened),
        "n_candidates": len(J),
        "total_demand": demand_nodes["demand"].sum(),
    }
    return status_str, sol


def build_kpi_table(sol):
    """Return a DataFrame of KPIs from the solution dict."""
    kpis = [
        ("Stores opened", f"{sol['n_stores']} / {sol['n_candidates']}"),
        ("Total demand (orders/day)", f"{sol['total_demand']:,.1f}"),
        ("Revenue (₹/day)", f"₹{sol['total_revenue']:,.0f}"),
        ("Transport cost (₹/day)", f"₹{sol['total_transport_cost']:,.0f}"),
        ("Fixed cost (₹/day)", f"₹{sol['total_fixed_cost']:,.0f}"),
        ("Total cost (₹/day)", f"₹{sol['total_transport_cost'] + sol['total_fixed_cost']:,.0f}"),
        ("Profit (₹/day)", f"₹{sol['profit']:,.0f}"),
        ("Avg transport cost / order (₹)", f"₹{sol['total_transport_cost'] / sol['total_demand']:.2f}"),
        ("Solver status", sol["status"]),
    ]
    return pd.DataFrame(kpis, columns=["KPI", "Value"])


def build_folium_map(sol):
    """Build a Folium interactive map showing the solution."""
    demand_nodes = sol["demand_nodes"]
    candidates = sol["candidates"]
    opened = sol["opened"]
    d = sol["d"]

    # palette for stores
    colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
        "#ffd8b1", "#000075", "#a9a9a9", "#ffe119", "#000000",
    ]
    store_color = {j: colors[k % len(colors)] for k, j in enumerate(opened)}

    center_lat = (CHENNAI_LAT_MIN + CHENNAI_LAT_MAX) / 2
    center_lon = (CHENNAI_LON_MIN + CHENNAI_LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

    # assignment lines
    for _, row in demand_nodes.dropna(subset=["assigned_site"]).iterrows():
        j = int(row["assigned_site"])
        color = store_color.get(j, "#888888")
        folium.PolyLine(
            locations=[
                [row["lat"], row["lon"]],
                [candidates.loc[j, "lat"], candidates.loc[j, "lon"]],
            ],
            color=color,
            weight=1.0,
            opacity=0.4,
        ).add_to(m)

    # demand nodes
    for i, row in demand_nodes.iterrows():
        j = row.get("assigned_site")
        color = store_color.get(int(j), "#888888") if pd.notna(j) else "#888888"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(3, min(12, row["demand"] / demand_nodes["demand"].max() * 12)),
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Demand node {i}<br>Demand: {row['demand']:.0f} orders/day<br>Assigned store: {int(j) if pd.notna(j) else 'None'}",
        ).add_to(m)

    # candidate sites (not opened)
    closed = candidates[~candidates["opened"]]
    for j, row in closed.iterrows():
        folium.RegularPolygonMarker(
            location=[row["lat"], row["lon"]],
            number_of_sides=4,
            radius=8,
            color="#444444",
            fill=False,
            tooltip=f"Candidate {j} (not opened)<br>Capacity: {row['capacity']} orders/day<br>Fixed cost: ₹{row['fixed_cost']:,.0f}/day",
        ).add_to(m)

    # opened dark stores (stars)
    for j in opened:
        row = candidates.loc[j]
        color = store_color[j]
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
            tooltip=(
                f"<b>Dark Store {j}</b><br>"
                f"Capacity: {row['capacity']} orders/day<br>"
                f"Served: {row['served_demand']:.0f} orders/day<br>"
                f"Utilisation: {row['utilisation_pct']:.1f}%<br>"
                f"Fixed cost: ₹{row['fixed_cost']:,.0f}/day"
            ),
        ).add_to(m)

    return m


@st.cache_data(show_spinner=False)
def compute_tradeoff(n_customers, n_candidates, max_dist_km, k_values, seed=RNG_SEED):
    """Compute profit for each max_open value in k_values."""
    demand_nodes, candidates, d = generate_chennai_data(n_customers, n_candidates, seed)
    results = []
    for k in k_values:
        _, sol = solve_milp(demand_nodes, candidates, d, max_dist_km, max_open=k)
        if sol is not None:
            results.append({"max_open": k, "profit": sol["profit"], "stores_opened": sol["n_stores"]})
        else:
            results.append({"max_open": k, "profit": None, "stores_opened": None})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Store Location Optimiser – Chennai",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Dark Store Location Optimisation – Chennai")
st.caption(
    "Capacitated Facility Location MILP with Maximum-Distance Constraint | "
    "Engineering Design Optimisation Assignment 1"
)

with st.expander("📋 Model Description & Assumptions", expanded=False):
    st.markdown("""
    ### Model
    **Minimise:** `Σ transport_cost_rate × d_ij × x_ij + Σ F_j × y_j`

    **Subject to:**
    1. Demand satisfaction: `Σ_j x_ij = D_i` for all customers *i*
    2. Capacity: `Σ_i x_ij ≤ C_j × y_j` for all candidate sites *j*
    3. **Max-distance (new):** `x_ij = 0` if `d_ij > max_dist` — no customer can be served from a store that is farther than the specified maximum distance
    4. Integrality: `y_j ∈ {0,1}`, `x_ij ≥ 0`

    ### Assumptions
    | Parameter | Value |
    |---|---|
    | Revenue per order | ₹80 |
    | Transport cost rate | ₹12 / km / order |
    | Base fixed cost | ₹3,500 / day |
    | Fixed cost per capacity unit | ₹2.2 / day |
    | Capacity range | 300 – 1,500 orders/day |
    | Geography | Chennai Metropolitan Area |
    | Customer distribution | Synthetic, uniform across Chennai bbox |

    Candidate site capacities and fixed costs are derived from local demand density (same approach as course.py).
    """)

# ─── Sidebar inputs ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Problem Parameters")

n_customers = st.sidebar.slider(
    "Number of customers",
    min_value=100, max_value=10_000, value=500, step=100,
    help="Total synthetic customer pings to generate across Chennai",
)
n_candidates = st.sidebar.slider(
    "Dark store candidate locations",
    min_value=5, max_value=50, value=15, step=5,
    help="Number of potential dark store sites",
)
max_dist_km = st.sidebar.slider(
    "Maximum distance (km)",
    min_value=1.0, max_value=30.0, value=8.0, step=0.5,
    help="Maximum allowable distance between a dark store and the customers it serves",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Trade-off Analysis (fixed)")
st.sidebar.markdown(
    "Uses **10,000 customers** and **50 candidates** regardless of the sliders above."
)
run_tradeoff = st.sidebar.button("Run Trade-off Analysis", use_container_width=True)

# ─── Main panel ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Optimisation", "📈 Trade-off Plot"])

with tab1:
    with st.spinner("Generating data and solving MILP…"):
        demand_nodes, candidates, d = generate_chennai_data(n_customers, n_candidates)
        status_str, sol = solve_milp(demand_nodes, candidates, d, max_dist_km)

    if sol is None:
        st.error(
            f"⚠️ **Infeasible**: With **max distance = {max_dist_km} km** and the current "
            f"candidate locations, some demand nodes cannot be reached by any open store. "
            "Try increasing the maximum distance or the number of candidate locations."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stores Opened", f"{sol['n_stores']} / {sol['n_candidates']}")
        col2.metric("Total Revenue (₹/day)", f"₹{sol['total_revenue']:,.0f}")
        col3.metric("Total Cost (₹/day)", f"₹{sol['total_transport_cost'] + sol['total_fixed_cost']:,.0f}")
        col4.metric("Profit (₹/day)", f"₹{sol['profit']:,.0f}")

        st.subheader("1️⃣ KPI Summary")
        kpi_df = build_kpi_table(sol)
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        st.subheader("Opened Store Details")
        opened_detail = sol["candidates"][sol["candidates"]["opened"]][
            ["lat", "lon", "capacity", "served_demand", "utilisation_pct", "fixed_cost"]
        ].copy()
        opened_detail.columns = ["Lat", "Lon", "Capacity", "Served Demand", "Utilisation %", "Fixed Cost (₹/day)"]
        opened_detail["Utilisation %"] = opened_detail["Utilisation %"].round(1)
        opened_detail["Fixed Cost (₹/day)"] = opened_detail["Fixed Cost (₹/day)"].round(0)
        st.dataframe(opened_detail, use_container_width=True)

        st.subheader("2️⃣ Solution Map")
        st.caption("🔴 = Opened dark store | ◆ = Candidate (not opened) | Circle = Demand node (size ∝ demand)")
        m = build_folium_map(sol)
        st_folium(m, width="100%", height=550)

with tab2:
    st.subheader("📈 Profit vs. Number of Allowed Dark Stores")
    st.markdown(
        "Fixed setup: **10,000 customers**, **50 candidate sites**, "
        f"maximum distance = **{max_dist_km} km**. "
        "Stores allowed to open: 2, 5, 10, 20, 25."
    )

    if run_tradeoff:
        k_values = [2, 5, 10, 20, 25]
        with st.spinner("Running trade-off analysis (this may take 1–3 minutes)…"):
            tradeoff_df = compute_tradeoff(
                n_customers=10_000,
                n_candidates=50,
                max_dist_km=max_dist_km,
                k_values=k_values,
            )

        st.dataframe(
            tradeoff_df.rename(columns={
                "max_open": "Max Stores Allowed",
                "profit": "Profit (₹/day)",
                "stores_opened": "Stores Actually Opened",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        valid = tradeoff_df.dropna(subset=["profit"])
        ax.plot(valid["max_open"], valid["profit"] / 1e6, marker="o", linewidth=2.5,
                color="#4363d8", markersize=8)
        ax.fill_between(valid["max_open"], valid["profit"] / 1e6, alpha=0.15, color="#4363d8")
        for _, row in valid.iterrows():
            ax.annotate(
                f"₹{row['profit'] / 1e6:.2f}M",
                (row["max_open"], row["profit"] / 1e6),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
            )
        ax.set_xlabel("Maximum number of dark stores allowed to open", fontsize=11)
        ax.set_ylabel("Daily Profit (₹ Millions)", fontsize=11)
        ax.set_title(
            f"Profit vs. Store Count\n(10,000 customers, 50 candidates, max dist = {max_dist_km} km)",
            fontsize=12,
        )
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        infeasible = tradeoff_df[tradeoff_df["profit"].isna()]
        if not infeasible.empty:
            st.warning(
                f"⚠️ Infeasible for k = {infeasible['max_open'].tolist()} — "
                "too few stores to serve all demand within the distance limit. "
                "Try increasing the max distance."
            )
    else:
        st.info("👈 Click **Run Trade-off Analysis** in the sidebar to generate this plot.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Engineering Design Optimisation | Assignment 1 | "
    "Dark Store Location Problem | Chennai Metropolitan Area"
)
