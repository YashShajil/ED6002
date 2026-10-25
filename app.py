# -*- coding: utf-8 -*-
"""
Dark Store Location Optimisation – Streamlit App
=================================================

Enhanced MILP model with:
  • Maximum-distance constraint between dark store and customer (d_ij <= max_dist)
  • Real UP / Varanasi customer data loaded directly from 'AI for DS Sample Data.csv'
  • Interactive Streamlit UI with configurable economic parameters (Revenue, Transport rate)
  • Results:
      1. KPIs in a table (Revenue, Transport Cost, Fixed Cost, Net Profit)
      2. Detailed solution on an interactive OpenStreetMap (no API key required)
      3. Trade-off plot: Profit vs. allowed dark stores (k in {2, 5, 10, 20, 25})
         for 10,000 customers and 50 candidate locations.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ortools.linear_solver import pywraplp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & ASSUMPTIONS
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42
DATA_FILE = os.path.join(os.path.dirname(__file__), "AI for DS Sample Data.csv")

FIXED_COST_BASE = 3500.0        # Rs./day base fixed cost of opening a dark store
FIXED_COST_PER_CAPACITY = 2.2   # Rs./day per unit of capacity
FIXED_COST_NOISE = 0.10         # +/-10% realistic variation in rent/costs
OUTLIER_LOW_Q, OUTLIER_HIGH_Q = 0.01, 0.99


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clean_dataset():
    """Load and clean the customer pings from the assignment CSV."""
    if not os.path.exists(DATA_FILE):
        df = pd.read_csv("AI for DS Sample Data.csv")
    else:
        df = pd.read_csv(DATA_FILE)

    df = df.rename(columns={"points": "demand"})
    
    # Trim mis-geocoded outlier pings (as done in course.py)
    lat_lo, lat_hi = df["lat"].quantile([OUTLIER_LOW_Q, OUTLIER_HIGH_Q])
    lon_lo, lon_hi = df["long"].quantile([OUTLIER_LOW_Q, OUTLIER_HIGH_Q])
    clean = df[
        df["lat"].between(lat_lo, lat_hi) & df["long"].between(lon_lo, lon_hi)
    ].copy().reset_index(drop=True)
    return clean


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two sets of lat/lon coordinates."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def build_network_data(n_customers: int, n_candidates: int, seed: int = RNG_SEED):
    """
    Subsamples n_customers from the cleaned CSV, clusters them into demand nodes,
    and proposes n_candidates candidate dark-store sites with demand-proportional capacities.
    """
    clean_df = load_clean_dataset()
    n_sample = min(n_customers, len(clean_df))
    sample_df = clean_df.sample(n=n_sample, random_state=seed).copy()

    coords = sample_df[["lat", "long"]].to_numpy()
    weights = sample_df["demand"].to_numpy()

    # 1. Aggregate customer pings into demand nodes (set I)
    n_nodes = min(120, max(25, n_sample // 80))
    km_d = KMeans(n_clusters=n_nodes, random_state=seed, n_init=10)
    labels_d = km_d.fit_predict(coords, sample_weight=weights)

    demand_nodes = pd.DataFrame(km_d.cluster_centers_, columns=["lat", "lon"])
    demand_nodes["demand"] = (
        pd.Series(weights).groupby(labels_d).sum().reindex(range(n_nodes)).fillna(0).values
    )
    demand_nodes = demand_nodes[demand_nodes["demand"] > 0].reset_index(drop=True)
    demand_nodes.index.name = "i"

    # 2. Build candidate dark-store sites (set J) using weighted K-Means
    km_c = KMeans(n_clusters=n_candidates, random_state=seed + 1, n_init=10)
    labels_c = km_c.fit_predict(coords, sample_weight=weights)

    candidates = pd.DataFrame(km_c.cluster_centers_, columns=["lat", "lon"])
    local_dem = (
        pd.Series(weights).groupby(labels_c).sum().reindex(range(n_candidates)).fillna(0).values
    )
    candidates["local_demand"] = local_dem
    candidates.index.name = "j"

    total_demand = demand_nodes["demand"].sum()

    # 3. Derive store capacities:
    # Sized so that opening subsets of stores (e.g. k=5, 10, 25) can handle demand
    raw_cap = candidates["local_demand"].clip(lower=1)
    base_cap = (total_demand / max(2, n_candidates * 0.30))
    candidates["capacity"] = (
        base_cap * (0.8 + 0.6 * (raw_cap - raw_cap.min()) / (raw_cap.max() - raw_cap.min() + 1e-9))
    ).round().astype(int)

    # 4. Derive fixed costs: base cost + capacity-scaled component + random noise
    rng = np.random.default_rng(seed)
    noise = 1.0 + rng.uniform(-FIXED_COST_NOISE, FIXED_COST_NOISE, size=len(candidates))
    candidates["fixed_cost"] = (
        (FIXED_COST_BASE + FIXED_COST_PER_CAPACITY * candidates["capacity"]) * noise
    ).round(1)

    # 5. Precompute pairwise distance matrix d_ij
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


# ─────────────────────────────────────────────────────────────────────────────
# MILP OPTIMIZATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def solve_milp(demand_nodes, candidates, d, max_dist_km, revenue_per_order=80.0, transport_rate=1.5, max_open=None):
    """
    Solves the Capacitated Facility Location Problem with a Maximum Distance Constraint.
    """
    I = demand_nodes.index.tolist()
    J = candidates.index.tolist()
    D = demand_nodes["demand"].to_dict()
    C = candidates["capacity"].to_dict()
    F = candidates["fixed_cost"].to_dict()
    total_demand = sum(D.values())

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return "SOLVER_UNAVAILABLE", None

    infinity = solver.infinity()

    # Decision variables
    y = {j: solver.BoolVar(f"y_{j}") for j in J}
    x = {(i, j): solver.NumVar(0.0, infinity, f"x_{i}_{j}") for i in I for j in J}
    u = {i: solver.NumVar(0.0, infinity, f"u_{i}") for i in I}  # Unserved demand slack

    # Constraint 1: Demand satisfaction (with unserved slack variable)
    for i in I:
        solver.Add(sum(x[i, j] for j in J) + u[i] == D[i])

    # Constraint 2: Facility capacity
    for j in J:
        solver.Add(sum(x[i, j] for i in I) <= C[j] * y[j])

    # Constraint 3 (ASSIGNMENT REQUIREMENT): Max distance limit
    for i in I:
        for j in J:
            if d[i, j] > max_dist_km:
                solver.Add(x[i, j] == 0.0)

    # Constraint 4: Optional limit on number of open stores
    if max_open is not None:
        solver.Add(sum(y[j] for j in J) <= max_open)

    # Penalty on unserved demand ensures the model serves all reachable demand
    UNSERVED_PENALTY = 500.0

    transport_cost_expr = sum(transport_rate * d[i, j] * x[i, j] for i in I for j in J)
    fixed_cost_expr = sum(F[j] * y[j] for j in J)
    unserved_penalty_expr = sum(UNSERVED_PENALTY * u[i] for i in I)

    solver.Minimize(transport_cost_expr + fixed_cost_expr + unserved_penalty_expr)
    solver.SetTimeLimit(60_000)

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return "INFEASIBLE", None

    # Extract solution
    opened = [j for j in J if y[j].solution_value() > 0.5]
    flows = [
        (i, j, x[i, j].solution_value())
        for i in I for j in J
        if x[i, j].solution_value() > 1e-4
    ]
    flows_df = pd.DataFrame(flows, columns=["i", "j", "flow"])

    total_served = flows_df["flow"].sum() if not flows_df.empty else 0.0
    total_transport_cost = float(
        flows_df.apply(lambda r: transport_rate * d[int(r.i), int(r.j)] * r.flow, axis=1).sum()
    ) if not flows_df.empty else 0.0
    
    total_fixed_cost = float(candidates.loc[opened, "fixed_cost"].sum()) if opened else 0.0
    total_revenue = float(total_served * revenue_per_order)
    profit = total_revenue - total_transport_cost - total_fixed_cost

    utilisation = flows_df.groupby("j")["flow"].sum().reindex(opened).fillna(0) if not flows_df.empty else pd.Series(0, index=opened)
    candidates_copy = candidates.copy()
    candidates_copy["opened"] = candidates_copy.index.isin(opened)
    candidates_copy["served_demand"] = candidates_copy.index.map(utilisation).fillna(0)
    candidates_copy["utilisation_pct"] = np.where(
        candidates_copy["opened"],
        candidates_copy["served_demand"] / candidates_copy["capacity"] * 100.0,
        np.nan,
    )

    demand_nodes_copy = demand_nodes.copy()
    if not flows_df.empty:
        assigned = (
            flows_df.sort_values("flow", ascending=False)
            .drop_duplicates("i")
            .set_index("i")["j"]
        )
        demand_nodes_copy["assigned_site"] = demand_nodes_copy.index.map(assigned)
    else:
        demand_nodes_copy["assigned_site"] = np.nan

    sol = {
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
        "opened": opened,
        "n_stores": len(opened),
        "n_candidates": len(J),
        "total_demand": total_demand,
        "served_demand": total_served,
        "unserved_demand": total_demand - total_served,
        "total_revenue": total_revenue,
        "total_transport_cost": total_transport_cost,
        "total_fixed_cost": total_fixed_cost,
        "profit": profit,
        "candidates": candidates_copy,
        "demand_nodes": demand_nodes_copy,
        "d": d,
    }
    return "OPTIMAL", sol


def build_kpi_table(sol):
    """Return a styled KPI DataFrame."""
    coverage_pct = (sol["served_demand"] / sol["total_demand"] * 100.0) if sol["total_demand"] > 0 else 0
    kpis = [
        ("Stores Opened", f"{sol['n_stores']} of {sol['n_candidates']} candidates"),
        ("Total Demand", f"{sol['total_demand']:,.1f} orders/day"),
        ("Demand Served", f"{sol['served_demand']:,.1f} orders/day ({coverage_pct:.1f}%)"),
        ("Demand Unserved", f"{sol['unserved_demand']:,.1f} orders/day"),
        ("Daily Revenue", f"₹ {sol['total_revenue']:,.0f}"),
        ("Transport (Delivery) Cost", f"₹ {sol['total_transport_cost']:,.0f}"),
        ("Fixed Operating Cost", f"₹ {sol['total_fixed_cost']:,.0f}"),
        ("Total Daily Cost", f"₹ {sol['total_transport_cost'] + sol['total_fixed_cost']:,.0f}"),
        ("Net Profit", f"₹ {sol['profit']:,.0f} / day (₹ {sol['profit']/1e5:.2f} Lakhs)"),
        ("Solver Status", sol["status"]),
    ]
    return pd.DataFrame(kpis, columns=["Key Performance Indicator", "Value"])


def build_folium_map(sol, dark_mode=False):
    """Build an interactive Folium map showing the network solution."""
    demand_nodes = sol["demand_nodes"]
    candidates = sol["candidates"]
    opened = sol["opened"]

    center_lat = demand_nodes["lat"].mean()
    center_lon = demand_nodes["lon"].mean()

    # OpenStreetMap tiles (standard light vs dark CSS-filtered basemap)
    if dark_mode:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="&copy; OpenStreetMap contributors &copy; CARTO",
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="OpenStreetMap",
        )

    colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
        "#ffd8b1", "#000075", "#a9a9a9", "#ffe119", "#000000",
    ]
    store_color = {j: colors[idx % len(colors)] for idx, j in enumerate(opened)}

    # 1. Assignment lines: customer node -> assigned opened store
    for _, row in demand_nodes.dropna(subset=["assigned_site"]).iterrows():
        j = int(row["assigned_site"])
        color = store_color.get(j, "#888888")
        folium.PolyLine(
            locations=[
                [row["lat"], row["lon"]],
                [candidates.loc[j, "lat"], candidates.loc[j, "lon"]],
            ],
            color=color,
            weight=1.2,
            opacity=0.35,
        ).add_to(m)

    # 2. Demand nodes (circles sized by demand)
    for i, row in demand_nodes.iterrows():
        j = row.get("assigned_site")
        color = store_color.get(int(j), "#999999") if pd.notna(j) else "#999999"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(3, min(10, row["demand"] / demand_nodes["demand"].max() * 10)),
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=(
                f"<b>Demand Node {i}</b><br>"
                f"Demand: {row['demand']:.0f} orders/day<br>"
                f"Assigned Store: {int(j) if pd.notna(j) else 'Unserved'}"
            ),
        ).add_to(m)

    # 3. Candidate sites (closed = black squares)
    closed = candidates[~candidates["opened"]]
    for j, row in closed.iterrows():
        folium.RegularPolygonMarker(
            location=[row["lat"], row["lon"]],
            number_of_sides=4,
            radius=6,
            color="#333333",
            fill=False,
            tooltip=(
                f"<b>Candidate {j} (Not Opened)</b><br>"
                f"Capacity: {row['capacity']} orders/day<br>"
                f"Fixed cost: ₹{row['fixed_cost']:,.0f}/day"
            ),
        ).add_to(m)

    # 4. Opened dark stores (prominent markers)
    for j in opened:
        row = candidates.loc[j]
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
            tooltip=(
                f"<b>Opened Dark Store {j}</b><br>"
                f"Served: {row['served_demand']:.0f} / {row['capacity']} orders/day<br>"
                f"Utilisation: {row['utilisation_pct']:.1f}%<br>"
                f"Fixed Cost: ₹{row['fixed_cost']:,.0f}/day"
            ),
        ).add_to(m)

    return m


@st.cache_data(show_spinner=False)
def compute_tradeoff(n_customers, n_candidates, max_dist_km, revenue_per_order, transport_rate, k_values, seed=RNG_SEED):
    """Computes net profit for each maximum allowed store count k."""
    demand_nodes, candidates, d = build_network_data(n_customers, n_candidates, seed)
    results = []
    for k in k_values:
        _, sol = solve_milp(demand_nodes, candidates, d, max_dist_km, revenue_per_order=revenue_per_order, transport_rate=transport_rate, max_open=k)
        if sol is not None:
            results.append({
                "max_open": k,
                "profit": sol["profit"],
                "stores_opened": sol["n_stores"],
                "demand_served": sol["served_demand"],
            })
        else:
            results.append({
                "max_open": k,
                "profit": None,
                "stores_opened": None,
                "demand_served": None,
            })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Store Location Optimiser",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Dark Store Location Optimisation System")
st.caption(
    "Capacitated Facility Location Model with Distance Constraint | "
    "ED6002 Engineering Design Optimisation Assignment 1"
)

clean_dataset = load_clean_dataset()
total_rows_csv = len(clean_dataset)

with st.expander("📖 Problem Background, Capacity Decision & Formulation", expanded=False):
    st.markdown(f"""
    ### 1. How Store Capacities are Decided
    - **Candidate Locations ($J$):** Proposed at dense customer demand hot-spots using **Weighted K-Means Clustering**.
    - **Capacity Sizing ($C_j$):** Sites located in denser, higher-demand clusters are assigned higher daily capacity, while suburban/sparse locations receive smaller capacity.
    - **Fixed Operating Cost ($F_j$):** Larger stores carry higher fixed costs due to rent, staff, and storage:
      $$F_j = (\\text{{Base Cost ₹3,500}} + 2.2 \\times C_j) \\times (1 \\pm 10\\%)$$

    ### 2. Mathematical Formulation
    $$\\min \\sum_{{i \\in I}} \\sum_{{j \\in J}} c \\cdot d_{{ij}} \\cdot x_{{ij}} + \\sum_{{j \\in J}} F_j \\cdot y_j + \\text{{Penalty}} \\cdot \\sum_{{i \\in I}} u_i$$

    **Subject to:**
    1. **Demand Satisfaction:** $\\sum_{{j \\in J}} x_{{ij}} + u_i = D_i \\quad \\forall i \\in I$
    2. **Facility Capacity:** $\\sum_{{i \\in I}} x_{{ij}} \\le C_j \\cdot y_j \\quad \\forall j \\in J$
    3. **Maximum Distance Limit (Assignment Enhancement):**
       $$x_{{ij}} = 0 \\quad \\text{{if }} d_{{ij}} > d_{{\\max}}$$
       *No customer can be served from a dark store farther than $d_{{\\max}}$ km.*
    4. **Store Budget Limit (Trade-off Analysis):** $\\sum_{{j \\in J}} y_j \\le k$
    5. **Binary & Non-negativity:** $y_j \\in \\{{0,1\\}}, \\; x_{{ij}} \\ge 0, \\; u_i \\ge 0$
    """)

# ─── Sidebar Controls ────────────────────────────────────────────────────────
st.sidebar.header("🎨 Appearance")
theme_choice = st.sidebar.radio(
    "Select Theme:",
    options=["☀️ Light Mode", "🌙 Dark Mode"],
    index=0,
    horizontal=True,
    help="Switch between Light and Dark mode styling",
)
dark_mode = (theme_choice == "🌙 Dark Mode")

if dark_mode:
    # Explicit CSS for complete dark mode styling
    st.markdown("""
        <style>
            .stApp {
                background-color: #0e1117 !important;
                color: #e6edf3 !important;
            }
            header[data-testid="stHeader"] {
                background-color: #0e1117 !important;
            }
            [data-testid="stSidebar"] {
                background-color: #161b22 !important;
                color: #e6edf3 !important;
            }
            [data-testid="stSidebar"] * {
                color: #e6edf3 !important;
            }
            [data-testid="stMetricValue"] {
                color: #58a6ff !important;
            }
            [data-testid="stMetricLabel"] {
                color: #8b949e !important;
            }
            .stDataFrame {
                background-color: #161b22 !important;
            }
            p, h1, h2, h3, h4, span, label {
                color: #e6edf3 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    plt.style.use("dark_background")
else:
    # Explicit CSS for clean light mode styling
    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff !important;
                color: #1a1a1a !important;
            }
            header[data-testid="stHeader"] {
                background-color: #ffffff !important;
            }
            [data-testid="stSidebar"] {
                background-color: #f8f9fa !important;
                color: #1a1a1a !important;
            }
            [data-testid="stSidebar"] * {
                color: #1a1a1a !important;
            }
            [data-testid="stMetricValue"] {
                color: #0969da !important;
            }
            [data-testid="stMetricLabel"] {
                color: #57606a !important;
            }
            p, h1, h2, h3, h4, span, label {
                color: #1a1a1a !important;
            }
        </style>
    """, unsafe_allow_html=True)
    plt.style.use("default")

st.sidebar.header("⚙️ Network Inputs")

n_customers = st.sidebar.slider(
    "Number of Customers (sampled from CSV)",
    min_value=500,
    max_value=total_rows_csv,
    value=10000,
    step=500,
    help=f"Select sample size from the {total_rows_csv:,} total customer pings in the CSV",
)

n_candidates = st.sidebar.slider(
    "Dark Store Candidate Locations",
    min_value=5,
    max_value=60,
    value=35,
    step=5,
    help="Number of candidate dark store sites proposed via demand clustering",
)

max_dist_km = st.sidebar.slider(
    "Maximum Allowable Distance (km)",
    min_value=5.0,
    max_value=60.0,
    value=30.0,
    step=2.5,
    help="Maximum distance constraint between any customer and their fulfilling dark store",
)

st.sidebar.header("💰 Economic Parameters")

revenue_per_order = st.sidebar.number_input(
    "Revenue per Order (₹)",
    min_value=40.0,
    max_value=300.0,
    value=100.0,
    step=5.0,
    help="Average revenue generated per delivered order",
)

transport_rate = st.sidebar.number_input(
    "Transport Cost Rate (₹/km/order)",
    min_value=0.5,
    max_value=15.0,
    value=1.5,
    step=0.25,
    help="Per-km delivery cost per order (course benchmark ~1.0 - 2.0)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Trade-off Analysis")
st.sidebar.write("Evaluate profit for **10,000 customers** and **50 candidate stores** across $k \\in [2, 5, 10, 20, 25]$:")
run_tradeoff = st.sidebar.button("Run Trade-off Analysis", use_container_width=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Optimisation Results", "📈 Trade-off Plot"])

with tab1:
    with st.spinner("Clustering customer pings & solving MILP..."):
        demand_nodes, candidates, d = build_network_data(n_customers, n_candidates)
        status_str, sol = solve_milp(demand_nodes, candidates, d, max_dist_km, revenue_per_order=revenue_per_order, transport_rate=transport_rate)

    if sol is None or sol["n_stores"] == 0:
        st.error(
            f"No stores could be opened within {max_dist_km} km. Try increasing the maximum distance."
        )
    else:
        # Top KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stores Opened", f"{sol['n_stores']} / {sol['n_candidates']}")
        c2.metric("Daily Revenue", f"₹{sol['total_revenue']:,.0f}")
        c3.metric("Total Daily Cost", f"₹{sol['total_transport_cost'] + sol['total_fixed_cost']:,.0f}")
        c4.metric("Net Daily Profit", f"₹{sol['profit']:,.0f}", f"₹{sol['profit']/1e5:.2f} Lakhs")

        st.subheader("1️⃣ KPI Summary Table")
        st.dataframe(build_kpi_table(sol), use_container_width=True, hide_index=True)

        st.subheader("Opened Dark Stores Breakdown")
        opened_df = sol["candidates"][sol["candidates"]["opened"]][
            ["lat", "lon", "capacity", "served_demand", "utilisation_pct", "fixed_cost"]
        ].copy()
        opened_df.columns = ["Latitude", "Longitude", "Capacity (orders/day)", "Served (orders/day)", "Utilisation %", "Fixed Cost (₹/day)"]
        opened_df["Utilisation %"] = opened_df["Utilisation %"].round(1)
        opened_df["Fixed Cost (₹/day)"] = opened_df["Fixed Cost (₹/day)"].round(0)
        st.dataframe(opened_df, use_container_width=True)

        st.subheader("2️⃣ Network Solution on Map")
        st.caption("🔴 Red Marker = Opened Dark Store | ◾ Black Diamond = Unopened Candidate | 🔵 Circle = Demand Node")
        folium_map = build_folium_map(sol, dark_mode=dark_mode)
        st_folium(folium_map, width="100%", height=550)

with tab2:
    st.subheader("📈 Trade-off: Profit vs. Allowed Dark Store Count")
    st.markdown(
        f"**Specification:** Evaluated on **10,000 customers** and **50 dark store candidate locations** "
        f"with maximum distance = **{max_dist_km} km** for up to **2, 5, 10, 20, 25** open stores."
    )

    if run_tradeoff:
        k_values = [2, 5, 10, 20, 25]
        with st.spinner("Computing optimal solution for each k in [2, 5, 10, 20, 25]..."):
            tradeoff_df = compute_tradeoff(
                n_customers=10000,
                n_candidates=50,
                max_dist_km=max_dist_km,
                revenue_per_order=revenue_per_order,
                transport_rate=transport_rate,
                k_values=k_values,
            )

        display_df = tradeoff_df.copy()
        display_df.columns = ["Max Stores Allowed (k)", "Net Daily Profit (₹)", "Stores Opened", "Demand Served"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        valid = tradeoff_df.dropna(subset=["profit"]).copy()
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x_vals = valid["max_open"].astype(float).to_numpy()
            y_vals = (valid["profit"].astype(float) / 1e5).to_numpy()  # in Lakhs

            ax.plot(x_vals, y_vals, marker="o", linewidth=2.5, color="#1f77b4", markersize=8)
            ax.fill_between(x_vals, 0, y_vals, alpha=0.15, color="#1f77b4")

            for x, y in zip(x_vals, y_vals):
                ax.annotate(
                    f"₹{y:.2f}L",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xlabel("Maximum Number of Dark Stores Allowed to Open (k)", fontsize=11)
            ax.set_ylabel("Daily Net Profit (₹ Lakhs)", fontsize=11)
            ax.set_title(
                f"Profit vs. Allowed Dark Stores Trade-off\n(10,000 customers, 50 candidates, max dist = {max_dist_km} km)",
                fontsize=12,
            )
            ax.set_xticks(k_values)
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.info("👈 Click **Run Trade-off Analysis** in the sidebar to compute and plot the profit trade-off curve.")

st.markdown("---")
st.caption("ED6002: Engineering Design Optimisation | Assignment 1 | IIT Madras")
