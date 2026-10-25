# 🛒 Dark Store Location Optimisation – Chennai

**Assignment 1 | ED6002 Engineering Design Optimisation | IIT Madras**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yashsh-ed6002-app-dark-store.streamlit.app)

---

## 📌 Problem Statement

A **dark store** is a retail fulfilment centre (mini-warehouse) that serves only online orders — no walk-in customers. The goal is to decide:

- **Which candidate sites to open** as dark stores
- **How to assign customers** to open stores

…so that **total cost (transport + fixed) is minimised** while satisfying:
1. All customer demand is fully met
2. No store exceeds its capacity
3. *(New constraint)* No customer is served from a store **farther than a specified maximum distance**

---

## 🧮 Mathematical Model

### Decision Variables
| Variable | Type | Meaning |
|---|---|---|
| $y_j$ | Binary {0,1} | 1 if dark store at candidate site *j* is opened |
| $x_{ij}$ | Continuous ≥ 0 | Orders/day served to customer *i* from store *j* |

### Objective

$$\text{Minimise} \quad \sum_i \sum_j c \cdot d_{ij} \cdot x_{ij} \;+\; \sum_j F_j \cdot y_j$$

where $c$ = transport cost rate (₹/km/order), $d_{ij}$ = distance (km), $F_j$ = daily fixed cost of site $j$.

### Constraints

| # | Constraint | Description |
|---|---|---|
| C1 | $\sum_j x_{ij} = D_i \;\forall i$ | All demand satisfied |
| C2 | $\sum_i x_{ij} \le C_j \cdot y_j \;\forall j$ | Capacity, only if open |
| C3 | $x_{ij} = 0 \;\text{if}\; d_{ij} > d_{\max}$ | **Maximum distance (new)** |
| C4 | $y_j \in \{0,1\},\; x_{ij} \ge 0$ | Domain |

### Profit

$$\text{Profit} = \text{Revenue} - \text{Transport Cost} - \text{Fixed Cost}$$

---

## 🗺️ Geography

All customer and candidate-site data is **synthetically generated** within the **Chennai Metropolitan Area** bounding box:

| | Lat | Lon |
|---|---|---|
| South-West | 12.78° N | 80.02° E |
| North-East | 13.23° N | 80.35° E |

---

## ⚙️ Assumptions

| Parameter | Value | Rationale |
|---|---|---|
| Revenue per order | ₹80 | ~20% margin on ₹400 AOV |
| Transport cost rate | ₹12 / km / order | Last-mile delivery vehicle cost |
| Base fixed cost | ₹3,500 / day | Dark store rent + staff + ops |
| Fixed cost per capacity unit | ₹2.2 / day | Scales with store size |
| Store capacity range | 300–1,500 orders/day | Typical quick-commerce dark store |
| Capacity slack factor | 1.35× | Ensures feasibility of demand satisfaction |

---

## 🖥️ App Features

### Tab 1 – Optimisation
- **Inputs** (sidebar):
  - Number of customers (100–10,000)
  - Candidate dark store locations (5–50)
  - Maximum distance km (1.0–30.0 km)
- **Outputs:**
  - KPI summary table (Revenue, Transport Cost, Fixed Cost, Profit, Utilisation, Solver Status)
  - Interactive Folium map (demand nodes, opened stores, assignment lines)

### Tab 2 – Trade-off Plot
- Fixed: **10,000 customers**, **50 candidate sites**
- Sweeps max stores allowed: **k ∈ {2, 5, 10, 20, 25}**
- Shows daily Profit (₹ millions) vs. allowed store count

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11+
- pip

### Install & Run

```bash
# Clone the repo
git clone https://github.com/YashShajil/ED6002.git
cd ED6002

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `ortools` | Google OR-Tools MILP solver (CBC) |
| `folium` + `streamlit-folium` | Interactive map rendering |
| `scikit-learn` | K-Means clustering for demand aggregation |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib` | Trade-off plot |

---

## 📁 File Structure

```
ED6002/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # App theme & server settings
├── course.py                 # Original class demo (reference)
└── README.md                 # This file
```

---

## 📚 References

- Daskin, M. S. (1995). *Network and Discrete Location: Models, Algorithms, and Applications*. Wiley.
- Google OR-Tools Documentation: https://developers.google.com/optimization
- Streamlit Documentation: https://docs.streamlit.io

---

*IIT Madras | ED6002 Engineering Design Optimisation | 2026*
