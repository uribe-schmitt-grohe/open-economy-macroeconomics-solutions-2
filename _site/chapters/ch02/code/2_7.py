# 1. Simulate the economy for 100 years.
import numpy as np

# Set seed for reproducibility
np.random.seed(123)

# Model parameters
rho = 0.9            # AR(1) coefficient for output
y_bar = 1.0          # Steady‐state level of output
sigma_eps = 0.03     # Standard deviation of output shock
r = 0.1              # World interest rate
beta = 1 / (1 + r)   # Discount factor

# Pre‐allocate array for output levels (length T = 100)
T = 100
y = np.zeros(T)
y[0] = y_bar         # Initialize with steady‐state output

# Simulate y_t according to: y_t - y_bar = rho*(y_{t-1} - y_bar) + eps_t
for t in range(1, T):
    y[t] = y_bar + rho * (y[t-1] - y_bar) + np.random.normal(0, sigma_eps)

# 2. Discard the first 50 years of data.
y_trim = y[50:]

# 3. Compute growth rates of output and consumption, and the trade balance‐to‐output ratio.
# First, compute the steady‐state consumption:
d_minus1 = y_bar / 2          # Initial net foreign assets
steady_c = y_bar + r * d_minus1  # c = y + r*d

# Compute consumption series for trimmed sample
denominator = 1 + beta * rho
c_trim = steady_c + (y_trim - y_bar) / denominator

# Compute annual growth rates (in percentage terms) for y and c:
gy = 100.0 * (np.log(y_trim[1:]) - np.log(y_trim[:-1]))
gc = 100.0 * (np.log(c_trim[1:]) - np.log(c_trim[:-1]))

# Compute trade balance‐to‐output ratio:
tb_trim = y_trim - c_trim
tby = tb_trim / y_trim  # length = 50
tby_aligned = tby[1:]

# 4. Compute sample standard deviations and the correlation.
sigma_gy = np.std(gy, ddof=1)  # sample standard deviation
sigma_gc = np.std(gc, ddof=1)

# Compute correlation between output growth and trade balance‐to‐output ratio:
corr_gy_tby = np.corrcoef(gy, tby_aligned)[0, 1]

# Display results for this single replication:
print(f"std(gy) = {sigma_gy:.4f}")
print(f"std(gc) = {sigma_gc:.4f}")
print(f"std(gc)/std(gy) = {sigma_gc / sigma_gy:.4f}")
print(f"corr(gy, tby) = {corr_gy_tby:.4f}")

# 5. Replicate steps 1–4 10,000 times and keep record of σ_gy, σ_gc, and corr(gy, tby).
n_reps = 10_000

# Pre‐allocate arrays to store statistics from each replication
sigma_gy_arr = np.zeros(n_reps)
sigma_gc_arr = np.zeros(n_reps)
corr_arr = np.zeros(n_reps)

for rep in range(n_reps):
    # (a) Simulate a new 100‐year output path
    y_sim = np.zeros(T)
    y_sim[0] = y_bar
    for t in range(1, T):
        y_sim[t] = y_bar + rho * (y_sim[t-1] - y_bar) + np.random.normal(0, sigma_eps)
    
    # (b) Discard the first 50 observations
    y_sim_trim = y_sim[50:]
    
    # (c) Compute consumption series for trimmed sample
    c_sim_trim = steady_c + (y_sim_trim - y_bar) / denominator
    
    # (d) Compute growth rates for y and c
    gy_sim = 100.0 * (np.log(y_sim_trim[1:]) - np.log(y_sim_trim[:-1]))
    gc_sim = 100.0 * (np.log(c_sim_trim[1:]) - np.log(c_sim_trim[:-1]))
    
    # (e) Compute trade balance‐to‐output ratio
    tb_sim_trim = y_sim_trim - c_sim_trim
    tby_sim = tb_sim_trim / y_sim_trim
    tby_sim_aligned = tby_sim[1:]  # align with growth rate indices
    
    # (f) Compute sample statistics
    sigma_gy_arr[rep] = np.std(gy_sim, ddof=1)
    sigma_gc_arr[rep] = np.std(gc_sim, ddof=1)
    corr_arr[rep] = np.corrcoef(gy_sim, tby_sim_aligned)[0, 1]

# 6. Report average of std(gy), std(gc), std(gc)/std(gy), and corr(gy, tby) over all replications.
mean_sigma_gy = np.mean(sigma_gy_arr)
mean_sigma_gc = np.mean(sigma_gc_arr)
ratio_arr = sigma_gc_arr / sigma_gy_arr
mean_ratio = np.mean(ratio_arr)
mean_corr = np.mean(corr_arr)

print(f"Average std(gy) over {n_reps} reps = {mean_sigma_gy:.4f}")
print(f"Average std(gc) over {n_reps} reps = {mean_sigma_gc:.4f}")
print(f"Average std(gc)/std(gy) over {n_reps} reps = {mean_ratio:.4f}")
print(f"Average corr(gy, tby) over {n_reps} reps = {mean_corr:.4f}")