# 🛒 Dark Store Location Optimisation System

**Assignment 1 | ED6002 Engineering Design Optimisation | IIT Madras**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Problem Overview

A **dark store** is a dedicated fulfilment centre / mini-warehouse operating exclusively for online order delivery with no walk-in retail shoppers. 

The objective of this project is to develop and deploy a full-stack **Streamlit application** that solves the **Capacitated Facility Location Problem (CFLP)** with an added operational enhancement:
> **Maximum Distance Constraint:** No customer order can be fulfilled from a dark store that exceeds a user-specified maximum allowable distance threshold ($d_{\max}$).

---

## 🧮 Mathematical Model

### Decision Variables

| Variable | Type | Meaning |
|---|---|---|
| $y_j$ | Binary $\{0, 1\}$ | $1$ if dark store candidate site $j$ is opened; $0$ otherwise |
| $x_{ij}$ | Continuous $\ge 0$ | Volume of demand at node $i$ fulfilled by dark store $j$ (orders/day) |
| $u_i$ | Continuous $\ge 0$ | Unserved demand at node $i$ (penalised to maximise coverage) |

### Objective Function

$$\min \quad \sum_{i \in I} \sum_{j \in J} c \cdot d_{ij} \cdot x_{ij} \;+\; \sum_{j \in J} F_j \cdot y_j \;+\; \text{Penalty} \cdot \sum_{i \in I} u_i$$

Where:
- $c$ = Transport cost rate (₹/km/order)
- $d_{ij}$ = Great-circle distance between customer node $i$ and store $j$ (km, Haversine formula)
- $F_j$ = Fixed daily operational and lease cost for store $j$ (₹/day)
- $\text{Penalty}$ = Penalty multiplier for unserved demand (₹500/order)

### Constraints

| # | Formulation | Description |
|---|---|---|
| **C1** | $\sum_{j \in J} x_{ij} + u_i = D_i \quad \forall i \in I$ | **Demand Balance:** Customer demand is fulfilled or flagged as unserved |
| **C2** | $\sum_{i \in I} x_{ij} \le C_j \cdot y_j \quad \forall j \in J$ | **Capacity Constraint:** Orders processed only by open stores up to capacity $C_j$ |
| **C3** | $x_{ij} = 0 \quad \text{if } d_{ij} > d_{\max}$ | **Maximum Distance Constraint (Enhanced):** Disallows assignment beyond $d_{\max}$ |
| **C4** | $\sum_{j \in J} y_j \le k$ | **Budget / Store Limit:** Allows at most $k$ stores to open (trade-off analysis) |
| **C5** | $y_j \in \{0, 1\}, \; x_{ij} \ge 0, \; u_i \ge 0$ | **Integrality & Non-negativity** |

### Net Profit Metric

$$\text{Net Daily Profit} = \text{Revenue} - \text{Transport Cost} - \text{Fixed Store Cost}$$

$$\text{Profit} = \left( \sum_{i} \sum_{j} x_{ij} \times \text{Revenue per Order} \right) - \left( \sum_{i} \sum_{j} c \cdot d_{ij} \cdot x_{ij} \right) - \left( \sum_{j} F_j \cdot y_j \right)$$

---

## 🗺️ Geography & Dataset

- **Source File:** `AI for DS Sample Data.csv` (20,999 raw customer coordinate pings).
- **Region:** Eastern Uttar Pradesh cluster (Varanasi, Prayagraj, Jaunpur, Mirzapur, Ghazipur) spanning Lat 24.2°–26.5° N, Lon 81.6°–85.9° E.
- **Data Preprocessing:**
  1. Trims geocoding coordinate outliers (1st to 99th percentile clipping).
  2. Aggregates customer pings into demand nodes using **Weighted K-Means** (weighted by the `points` column).
  3. Identifies candidate dark-store sites at regional demand hot-spots.

---

## ⚙️ Capacity Sizing & Fixed Cost Derivation

Since dark store candidate sites are proposed from geospatial demand clusters:
1. **Candidate Capacity ($C_j$):** Proportional to the local demand density captured around candidate $j$ during clustering, scaled between minimum and maximum facility capacity.
2. **Fixed Operational Cost ($F_j$):** Derived using a baseline lease cost plus a capacity-scaling component:
   $$F_j = (\text{Base Cost ₹3,500} + 2.2 \times C_j) \times (1 \pm 10\% \text{ noise})$$
   Larger stores incur higher fixed overheads (rent, staffing, cold storage).

---

## 🖥️ Streamlit Application Features

### 1. Interactive Inputs
- **Number of Customers:** Dynamic slider sampling from 500 up to all 20,169 cleaned customer pings.
- **Candidate Store Locations:** 5 to 60 candidate sites.
- **Maximum Distance Limit:** 5.0 to 60.0 km.
- **Economic Controls:** Configurable Revenue per order (default ₹100) and Transport cost rate (default ₹1.5/km/order).
- **Appearance Switch:** One-click toggle between **☀️ Light Mode** and **🌙 Dark Mode** (includes custom dark-inverted Leaflet map styling with zero API key dependencies).

### 2. Output & Visualisation
- **KPI Summary Table:** Real-time metrics including total demand, served vs. unserved demand, daily revenue, transport cost, fixed costs, net profit, and solver optimality status.
- **Interactive OpenStreetMap:** Rendered via Folium with:
  - 🔴 Red markers for opened dark stores (with popup metrics).
  - ◾ Black diamond markers for unopened candidates.
  - 🔵 Demand nodes sized proportionally to daily order volume.
  - Connecting assignment rays depicting customer-to-facility allocation.
- **Trade-off Analysis ($k \in \{2, 5, 10, 20, 25\}$):**
  - Evaluates profit for **10,000 customers** across **50 candidate locations**.
  - Displays structured comparison table and publication-ready trade-off plot.

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.11+
- Git

### Steps

```bash
# 1. Clone repository
git clone https://github.com/YashShajil/ED6002.git
cd ED6002

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run app.py
```

The app will start locally at **`http://localhost:8501`**.

---

## ☁️ Deployment on Streamlit Cloud

1. Push latest code to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New app**, select:
   - **Repository:** `YashShajil/ED6002`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

---

## 📁 Repository Structure

```
ED6002/
├── app.py                      # Main Streamlit web application
├── AI for DS Sample Data.csv   # Customer demand coordinate dataset
├── requirements.txt            # Python dependencies (ortools, streamlit, folium, etc.)
├── .streamlit/
│   └── config.toml             # Headless server & theme configuration
├── course.py                   # Original course reference script
└── README.md                   # Project documentation
```

---

## 📚 References
- Google OR-Tools Linear Solver documentation: https://developers.google.com/optimization
- Streamlit framework documentation: https://docs.streamlit.io
- Daskin, M. S. (2013). *Network and Discrete Location: Models, Algorithms, and Applications*. John Wiley & Sons.

---

*Department of Engineering Design | Indian Institute of Technology Madras*
