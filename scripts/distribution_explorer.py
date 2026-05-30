import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="M5 Distribution Explorer", layout="wide")

st.title("📊 Tweedie vs. Negative Binomial Explorer")
st.markdown("""
This app illustrates the two probability distributions used in our M5 forecasting model. 
These are chosen specifically because they can handle **discrete counts** and **intermittent zeros**.
""")

# --- Sidebar Controls ---
st.sidebar.header("Global Parameters")
mu = st.sidebar.slider("Mean Sales (μ)", 0.1, 20.0, 2.0, help="The average expected sales per day.")
num_samples = st.sidebar.select_slider("Number of Simulations", options=[1000, 5000, 10000, 50000], value=10000)

st.sidebar.markdown("---")

# --- Negative Binomial Section ---
st.sidebar.header("Negative Binomial Parameters")
alpha = st.sidebar.slider("Shape/Dispersion (α)", 0.1, 10.0, 1.0, 
                         help="Lower α means higher variance (more 'bursty' sales).")

# --- Tweedie Section ---
st.sidebar.header("Tweedie Parameters")
p = st.sidebar.slider("Power (p)", 1.01, 1.99, 1.5, 
                     help="1.0=Poisson, 1.5=Compound Poisson-Gamma, 2.0=Gamma. For M5, 1.5 is standard.")
phi = st.sidebar.slider("Dispersion (φ)", 0.1, 5.0, 1.0, 
                       help="Scales the overall variance.")

# --- M5 Case Studies ---
st.markdown("---")
st.header("🎯 M5 Dataset Case Studies")
st.write("How do these parameters map to real Walmart products?")

case_study = st.selectbox("Select a Product Persona", [
    "Custom (Manual Sliders)",
    "Slow-Moving Hobby Item (Intermittent)",
    "High-Volume Grocery Item (Overdispersed)",
    "The 'SNAP Day' Shift (Promotion Effect)"
])

if case_study == "Slow-Moving Hobby Item (Intermittent)":
    mu, alpha, p, phi = 0.3, 0.5, 1.2, 1.0
    st.success("📝 **Context**: This item (e.g., a specific craft kit) sells maybe once or twice a week. Note the massive **Zero Spike** in both graphs.")
elif case_study == "High-Volume Grocery Item (Overdispersed)":
    mu, alpha, p, phi = 8.0, 2.0, 1.6, 1.5
    st.success("📝 **Context**: This item (e.g., milk or bread) sells every day, but volume varies wildly. Note the **Long Tail** reaching out to 20+ units.")
elif case_study == "The 'SNAP Day' Shift (Promotion Effect)":
    mu, alpha, p, phi = 12.0, 5.0, 1.5, 0.8
    st.success("📝 **Context**: During a SNAP event, demand spikes and becomes more 'predictable' (lower dispersion). The distribution is 'taller' and shifted right.")
else:
    st.info("Use the sidebar sliders to create your own scenario.")

# --- Simulation Logic (Re-run with case study params if selected) ---
if case_study != "Custom (Manual Sliders)":
    nb_data = get_nbinom_samples(mu, alpha, num_samples)
    tw_data = get_tweedie_samples(mu, p, phi, num_samples)

# --- Visualization ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Negative Binomial (Discrete)")
    st.info(f"**Variance**: {mu + (mu**2/alpha):.2f} (Target: {mu:.2f})")
    
    fig, ax = plt.subplots()
    bins = np.arange(0, max(nb_data.max(), 10) + 1) - 0.5
    ax.hist(nb_data, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Probability")
    ax.set_title(f"NB(μ={mu}, α={alpha})")
    
    # Highlight Zero Probability
    zeros_pct = (nb_data == 0).mean() * 100
    ax.annotate(f"Zero Sales: {zeros_pct:.1f}%", xy=(0, 0.1), xytext=(2, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    st.pyplot(fig)
    
    st.latex(r"P(Y=y) = \frac{\Gamma(y+\alpha)}{\Gamma(y+1)\Gamma(\alpha)} \left(\frac{\alpha}{\alpha+\mu}\right)^\alpha \left(\frac{\mu}{\alpha+\mu}\right)^y")

with col2:
    st.subheader("Tweedie (Zero-Inflated Continuous)")
    st.info(f"**Variance**: {phi * (mu**p):.2f} (Target: {mu:.2f})")
    
    fig, ax = plt.subplots()
    # Tweedie has a spike at 0, then a continuous tail
    ax.hist(tw_data[tw_data > 0], bins=30, density=True, color='salmon', edgecolor='black', alpha=0.7)
    
    # Manually add the zero spike bar for visualization
    zeros_pct_tw = (tw_data == 0).mean()
    ax.bar(0, zeros_pct_tw, width=0.5, color='red', label='Zero Spike')
    
    ax.set_xlabel("Units Sold (Continuous Proxy)")
    ax.set_ylabel("Density / Probability")
    ax.set_title(f"Tweedie(μ={mu}, p={p}, φ={phi})")
    ax.legend()
    st.pyplot(fig)
    
    st.latex(r"Var(Y) = \phi \cdot \mu^p")
    st.write("*Note: Red bar is the point mass at exactly 0.0*")

# --- REAL M5 DATA SECTION ---
st.markdown("---")
st.header("🏠 Reality Check: Actual Walmart Sales")

@st.cache_data
def load_sample_m5():
    path = Path("m5-forecasting-accuracy/sales_train_evaluation.csv")
    if not path.exists():
        return None
    # Load just a few rows to illustrate
    df = pd.read_csv(path, nrows=50)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    return df, day_cols

m5_df, m5_days = load_sample_m5()

if m5_df is not None:
    st.write("Let's look at a real item from the dataset.")
    item_idx = st.selectbox("Select an Item ID", range(len(m5_df)), 
                           format_func=lambda i: f"{m5_df.iloc[i]['item_id']} in {m5_df.iloc[i]['store_id']}")
    
    item_sales = m5_df.iloc[item_idx][m5_days].values.astype(float)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("The Raw Time Series")
        st.write("This is what the model sees: jagged, intermittent demand.")
        fig_ts, ax_ts = plt.subplots(figsize=(10, 3))
        ax_ts.plot(item_sales[-365:], color="black", linewidth=0.8)
        ax_ts.set_title("Last 365 Days of Sales")
        ax_ts.set_ylabel("Units")
        st.pyplot(fig_ts)

    with col_b:
        st.subheader("The 'Reality' Histogram")
        st.write("Notice the massive spike at 0.")
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(item_sales, bins=int(item_sales.max())+1, density=True, color="gray", alpha=0.5)
        ax_hist.set_title("Distribution of All History")
        st.pyplot(fig_hist)

    st.info(f"💡 **The Goal of DeepAR**: Instead of drawing a straight line through that jagged mess, the model predicts a **new version of that gray histogram** for every single day in the future.")

    # Compare with Normal Distribution
    st.subheader("Why standard Linear Regression fails here")
    mean_real = item_sales.mean()
    std_real = item_sales.std()
    
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
    x_range = np.linspace(-5, max(item_sales.max(), 10), 500)
    
    # Real Histogram
    ax_comp.hist(item_sales, bins=int(item_sales.max())+1, density=True, color="gray", alpha=0.3, label="Actual Walmart Data")
    
    # Normal Fit (Linear Regression Assumption)
    norm_pdf = stats.norm.pdf(x_range, mean_real, std_real)
    ax_comp.plot(x_range, norm_pdf, 'r--', label="Normal Dist (Linear Regression Assumption)")
    
    # NB Fit
    nb_prob = mean_real / (mean_real + (std_real**2 - mean_real)) if std_real**2 > mean_real else 0.9
    nb_alpha = (mean_real**2) / (std_real**2 - mean_real) if std_real**2 > mean_real else 10.0
    nb_pdf = stats.nbinom.pmf(np.round(x_range), nb_alpha, nb_alpha/(nb_alpha+mean_real))
    ax_comp.plot(x_range, nb_pdf, 'b-', label="Negative Binomial (Our Model)")
    
    ax_comp.axvline(0, color='black', linestyle='-')
    ax_comp.set_xlim(-3, item_sales.max() + 2)
    ax_comp.legend()
    st.pyplot(fig_comp)
    
    st.error("❌ **Observe the Red Line**: See how it predicts probability for **negative sales**? That's what happens when you use standard models. The **Blue Line** (Our Model) respects the 'Wall of Zero'.")

# --- THE LIKELIHOOD EXPLAINER ---
st.markdown("---")
st.header("🧠 The Paradox of the Single Point")
st.write("""
A common question: *'If I only have ONE sales number for tomorrow (e.g., 5 units), how can you say it follows a whole distribution?'*
""")

col_ex1, col_ex2 = st.columns([1, 2])

with col_ex1:
    actual_val = st.number_input("Actual Sale Observed", min_value=0, max_value=20, value=5)
    st.write(f"Assume the model predicted the **Tweedie** curve on the right for this specific day.")

with col_ex2:
    # Plot the distribution and the specific point
    fig_lik, ax_lik = plt.subplots(figsize=(8, 4))
    x_range = np.linspace(0, 20, 500)
    
    # Generate a theoretical Tweedie-like PDF for illustration
    # (Using a simple Gamma+Spike proxy for visual clarity)
    lambda_p = (mu**(2-p)) / (phi * (2-p))
    zeros_pct_th = np.exp(-lambda_p)
    
    # Continuous part
    alpha_g = (2-p) / (p-1)
    theta_g = phi * (p-1) * (mu**(p-1))
    # Approximation of Tweedie PDF for visualization
    pdf = stats.gamma.pdf(x_range, alpha_g, scale=theta_g) * (1-zeros_pct_th)
    
    ax_lik.plot(x_range, pdf, color="salmon", linewidth=2, label="Predicted 'Cloud of Possibility'")
    ax_lik.bar(0, zeros_pct_th, width=0.5, color='red', alpha=0.5)
    
    # Plot the Actual Point
    ax_lik.scatter([actual_val], [0], color="black", s=100, zorder=5, label="Actual Sale (Single Point)")
    
    # Draw the Likelihood Hook
    lik_height = stats.gamma.pdf(actual_val, alpha_g, scale=theta_g) * (1-zeros_pct_th) if actual_val > 0 else zeros_pct_th
    ax_lik.vlines(actual_val, 0, lik_height, colors="blue", linestyles="--", label="Likelihood (Height)")
    ax_lik.scatter([actual_val], [lik_height], color="blue", s=50)

    ax_lik.set_title("How the model is 'Graded'")
    ax_lik.legend()
    st.pyplot(fig_lik)

st.info(f"""
**How to explain this to stakeholders:**
1. **The Curve** is our 'Forecast'. It represents all possible outcomes.
2. **The Point (X)** is what actually happened.
3. **The Goal**: Training is simply the model trying to **raise the blue dashed line**. 
   - If the line is tall, the model 'predicted' the outcome well.
   - If the line is short, the model was 'surprised' and must change its internal rules to better fit this type of day next time.
""")
