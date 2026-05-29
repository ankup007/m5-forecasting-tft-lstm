# M5 DeepAR: Features, Scaling, and Loss Functions

This document provides a technical reference for the feature engineering and model architecture implemented in this repository, heavily inspired by the **3rd place M5 competition solution** (Team Kuai).

---

## 1. Target Scaling & Normalization

To handle the extreme range of demand across Walmart's hierarchy, we use **Mean Non-Zero Scaling**. Each time series $i$ is assigned a unique scale factor $s_i$.

### Scale Calculation
The scale $s_i$ is the average of all non-zero sales in the training history:
$$s_i = \max\left(1.0, \text{mean}(\{y_{i,t} \mid y_{i,t} > 0\})\right)$$

### Normalization Logic
During training and inference, the model operates on scaled targets $\tilde{y}$:
$$\tilde{y}_{i,t} = \frac{y_{i,t}}{s_i}$$
Final predictions are mapped back to units by multiplying by $s_i$. The model also receives $\log(1+s_i)$ as a static input to preserve information about the absolute volume of the series.

---

## 2. Feature Engineering

Features are categorized into **Static Categories**, **Temporal Categories**, **Known Covariates**, and **Dynamic Recurrent Features**.

### A. Static Categories (Embeddings)
These features are constant for a series and are embedded using the heuristic $dim = \min(50, \frac{C+1}{2})$ where $C$ is cardinality.
- `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

### B. Temporal Categories (Dynamic Event Embeddings)
Unlike binary flags, these features use a shared embedding table and are looked up at every time step $t$.
- `event_name_1_id`: Unique ID for each holiday/event (e.g., SuperBowl, Christmas).
- `event_type_1_id`: Unique ID for event categories (e.g., Sporting, National).

### C. Known Covariates (Normalized & Cyclic)
All calendar features are normalized to the range $[-0.5, 0.5]$ to ensure stable LSTM gradients.

| Feature | Transformation | Range / Center |
| :--- | :--- | :--- |
| **Cyclic Weekday** | $\sin / \cos(2\pi \cdot \text{wday} / 7)$ | $[-1.0, 1.0]$ |
| **Cyclic Month** | $\sin / \cos(2\pi \cdot \text{month} / 12)$ | $[-1.0, 1.0]$ |
| **Normalized Year** | $(\text{year} - 2013.5) / 5$ | $[-0.5, 0.5]$ |
| **Normalized Week** | $(\text{week} - 27.0) / 52$ | $[-0.5, 0.5]$ |
| **Normalized Day** | $(\text{day} - 16.0) / 30$ | $[-0.5, 0.5]$ |
| **Weekend Flag** | $1.0$ if Sat/Sun, else $0.0$ | $\{0, 1\}$ |
| **SNAP Flag** | Binary state-specific SNAP indicator | $\{0, 1\}$ |

### D. Advanced Price Ratios
- **Log Price**: $\log(1 + \text{sell\_price})$.
- **Relative to Max**: $(\frac{\text{price}}{\text{max\_historical\_price}}) - 0.5$. Captures current discount levels.
- **Relative to Dept**: $(\frac{\text{price}}{\text{mean\_dept\_price\_on\_day}}) - 1.0$. Captures price competitiveness within the category.

---

## 3. Dynamic Recurrent Features

These features are calculated "on-the-fly" inside the LSTM loop. They depend on the previous prediction $\hat{y}_{t-1}$.

### Scale-Aware Zero Counter
Tracks intermittent demand droughts. To be robust against "noisy" mean forecasts, we use a **0.5 unit threshold**:
$$
\text{counter}_t = 
\begin{cases} 
\text{counter}_{t-1} + 1 & \text{if } (\hat{y}_{t-1} \cdot s_i) < 0.5 \\
0 & \text{if } (\hat{y}_{t-1} \cdot s_i) \ge 0.5
\end{cases}
$$
This ensures the counter only resets if the model predicts at least one "half-unit" of sale.

### Rolling Averages
The model maintains a 28-day history buffer to compute dynamic statistics:
- **Rolling Mean 7**: $\text{mean}(\hat{y}_{t-7:t-1})$.
- **Rolling Mean 28**: $\text{mean}(\hat{y}_{t-28:t-1})$.

---

## 4. Loss Functions

### Negative Binomial (Default)
Used for discrete count data. It models the mean $\mu$ and a shape parameter $\alpha$ (alpha).
$$\text{NLL}(y; \mu, \alpha) = -\left[ \log \Gamma(y+\alpha) - \log \Gamma(\alpha) - \log \Gamma(y+1) + \alpha \log\left(\frac{\alpha}{\alpha+\mu}\right) + y \log\left(\frac{\mu}{\alpha+\mu}\right) \right]$$

### Tweedie Deviance Loss
The Tweedie distribution is ideal for zero-inflated continuous data. It has a point mass at zero and a continuous distribution for $y > 0$. We use a fixed power $\rho \in (1, 2)$ (typically $1.5$) and dispersion $\phi=1.0$.

The deviance loss (ignoring constants) is:
$$\text{Loss}(y; \mu) = \frac{1}{\phi} \left( \frac{\mu^{2-\rho}}{2-\rho} - \frac{y \mu^{1-\rho}}{1-\rho} \right)$$

*Note: For $\rho=1.5$, this simplifies to:*
$$\text{Loss}(y; \mu) = 2 \mu^{0.5} + 2 y \mu^{-0.5}$$

---

## 5. Model Input Structure

The LSTM receives a concatenated vector $\mathbf{x}_t$ at each step:
$$\mathbf{x}_t = [ \tilde{y}_{t-1}, \text{roll7}_t, \text{roll28}_t, \text{zero\_counter}_t, \mathbf{covariates}_t, \mathbf{event\_emb}_t, \mathbf{static\_emb}_t, \log(1+s_i) ]$$
