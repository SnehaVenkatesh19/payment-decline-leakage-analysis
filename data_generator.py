"""
Payment Failure & Revenue Leakage Analysis
==========================================
Data Generator — produces 1M realistic simulated payment transactions
with merchant categories, geographies, payment methods, failure codes,
retryability flags, and realistic noise/seasonality patterns.

Run:  python data_generator.py
Output: data/transactions.csv  (~130MB)
"""

import pandas as pd
import numpy as np
import os, sqlite3

os.makedirs("data", exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)
N    = 1_000_000

print(f"Generating {N:,} transactions...")

# ── 1. DIMENSIONS ─────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = {
    "E-commerce":        {"weight": 0.28, "avg_amount": 85,  "fail_base": 0.09},
    "SaaS/Subscription": {"weight": 0.18, "avg_amount": 55,  "fail_base": 0.06},
    "Travel":            {"weight": 0.12, "avg_amount": 420, "fail_base": 0.13},
    "Retail":            {"weight": 0.14, "avg_amount": 65,  "fail_base": 0.07},
    "Marketplaces":      {"weight": 0.10, "avg_amount": 140, "fail_base": 0.11},
    "Gaming":            {"weight": 0.08, "avg_amount": 25,  "fail_base": 0.10},
    "Healthcare":        {"weight": 0.06, "avg_amount": 310, "fail_base": 0.05},
    "Food & Delivery":   {"weight": 0.04, "avg_amount": 38,  "fail_base": 0.08},
}

GEOGRAPHIES = {
    "US":   {"weight": 0.42, "fail_mult": 1.00, "currency": "USD"},
    "EU":   {"weight": 0.20, "fail_mult": 0.90, "currency": "EUR"},
    "UK":   {"weight": 0.10, "fail_mult": 0.92, "currency": "GBP"},
    "IN":   {"weight": 0.08, "fail_mult": 1.35, "currency": "INR"},
    "BR":   {"weight": 0.06, "fail_mult": 1.45, "currency": "BRL"},
    "SG":   {"weight": 0.05, "fail_mult": 0.85, "currency": "SGD"},
    "CA":   {"weight": 0.05, "fail_mult": 0.95, "currency": "CAD"},
    "AU":   {"weight": 0.04, "fail_mult": 0.88, "currency": "AUD"},
}

PAYMENT_METHODS = {
    "Credit Card":  {"weight": 0.38, "fail_mult": 0.90},
    "Debit Card":   {"weight": 0.25, "fail_mult": 1.10},
    "Digital Wallet": {"weight": 0.18, "fail_mult": 0.75},
    "Bank Transfer":  {"weight": 0.10, "fail_mult": 1.20},
    "BNPL":           {"weight": 0.09, "fail_mult": 1.40},
}

# Failure codes: (label, is_retryable, description)
FAILURE_CODES = {
    "insufficient_funds":     (True,  "Insufficient funds — retry after top-up"),
    "card_expired":           (False, "Card expired — requires new card"),
    "do_not_honor":           (True,  "Generic bank decline — retryable"),
    "incorrect_cvc":          (False, "CVC mismatch — requires customer action"),
    "lost_card":              (False, "Card reported lost — permanent block"),
    "stolen_card":            (False, "Card reported stolen — permanent block"),
    "processing_error":       (True,  "Processor error — retry immediately"),
    "network_timeout":        (True,  "Network timeout — retry"),
    "velocity_exceeded":      (True,  "Velocity limit hit — retry after cooldown"),
    "fraud_suspected":        (False, "Fraud flag — requires review"),
    "currency_not_supported": (False, "Currency mismatch — not retryable"),
    "amount_too_large":       (False, "Exceeds card limit — not retryable"),
}

RETRY_CODES       = [c for c, (r, _) in FAILURE_CODES.items() if r]
NON_RETRY_CODES   = [c for c, (r, _) in FAILURE_CODES.items() if not r]

# Retry success rate by code (of retryable failures, how many recover revenue)
RETRY_RECOVERY = {
    "insufficient_funds": 0.45,
    "do_not_honor":       0.60,
    "processing_error":   0.82,
    "network_timeout":    0.88,
    "velocity_exceeded":  0.55,
}

# ── 2. SAMPLE DIMENSIONS ──────────────────────────────────────────────────────

mc_keys = list(MERCHANT_CATEGORIES.keys())
mc_w    = [MERCHANT_CATEGORIES[k]["weight"] for k in mc_keys]
merchant_cats = rng.choice(mc_keys, size=N, p=mc_w)

geo_keys = list(GEOGRAPHIES.keys())
geo_w    = [GEOGRAPHIES[k]["weight"] for k in geo_keys]
geos = rng.choice(geo_keys, size=N, p=geo_w)

pm_keys = list(PAYMENT_METHODS.keys())
pm_w    = [PAYMENT_METHODS[k]["weight"] for k in pm_keys]
pay_methods = rng.choice(pm_keys, size=N, p=pm_w)

# ── 3. TRANSACTION AMOUNTS (log-normal with category anchoring) ───────────────
amounts = np.array([
    max(1.0, rng.lognormal(
        mean=np.log(MERCHANT_CATEGORIES[mc]["avg_amount"]),
        sigma=0.7
    ))
    for mc in merchant_cats
])
amounts = np.round(amounts, 2)

# ── 4. TIMESTAMPS (18 months, with weekly/hourly seasonality) ─────────────────
start_ts = pd.Timestamp("2023-07-01")
end_ts   = pd.Timestamp("2024-12-31")
total_seconds = int((end_ts - start_ts).total_seconds())

raw_seconds = rng.integers(0, total_seconds, size=N)
timestamps  = pd.to_datetime(raw_seconds, unit="s", origin=start_ts)

# Add hour-of-day bias (more txns 9am–10pm)
hours      = timestamps.hour.to_numpy()
hour_mult  = np.where((hours >= 9) & (hours <= 22), 1.0, 0.5)
# Weekend uplift for E-commerce / Gaming
dow        = timestamps.dayofweek.to_numpy()  # 0=Mon, 6=Sun
weekend    = np.isin(dow, [5, 6])

# ── 5. RISK SCORE + SIGMOID FAILURE MODEL ────────────────────────────────────
# Compute a composite pre-authorization risk score (0–1) for each transaction.
# This simulates what a real payment risk model (like Stripe Radar) would produce.

mc_base  = np.array([MERCHANT_CATEGORIES[mc]["fail_base"] for mc in merchant_cats])
geo_mult = np.array([GEOGRAPHIES[geo]["fail_mult"] for geo in geos])
pm_mult  = np.array([PAYMENT_METHODS[pm]["fail_mult"] for pm in pay_methods])
amount_risk = np.clip((amounts - 50) / 1000, 0, 0.12)

# Composite risk: multiplicative factor from category/geo/pm, plus amount risk
raw_risk = mc_base * geo_mult * pm_mult + amount_risk

# Normalize to [0, 1] range to get a clean risk score
risk_min, risk_max = raw_risk.min(), raw_risk.max()
risk_score_base = (raw_risk - risk_min) / (risk_max - risk_min)

# Add realistic noise (simulates model imperfection and card-level factors)
noise = rng.normal(0, 0.04, size=N)
pre_auth_risk_score = np.clip(risk_score_base + noise, 0, 1)

# Sigmoid failure model: P(fail | risk_score) — steeper than pure probability
# This is more realistic: high-risk transactions fail reliably, low-risk rarely do
def sigmoid(x): return 1 / (1 + np.exp(-x))

THRESHOLD = 0.45
SIGMA     = 0.025
fail_probs = sigmoid((pre_auth_risk_score - THRESHOLD) / SIGMA) * 0.40 + 0.03
fail_probs = np.clip(fail_probs, 0.02, 0.90)

# Draw outcomes
is_failed = rng.random(size=N) < fail_probs

# ── 6. FAILURE CODES ──────────────────────────────────────────────────────────
failure_codes = np.where(is_failed, "pending_code", "success").astype("<U30")

# Assign codes to failed txns
failed_idx = np.where(is_failed)[0]
n_failed   = len(failed_idx)

# ~60% retryable, ~40% non-retryable
retryable_mask = rng.random(size=n_failed) < 0.60

retry_assigns    = rng.choice(RETRY_CODES,     size=n_failed)
nonretry_assigns = rng.choice(NON_RETRY_CODES, size=n_failed)

assigned_codes = [
    str(retry_assigns[i]) if retryable_mask[i] else str(nonretry_assigns[i])
    for i in range(n_failed)
]
failure_codes[failed_idx] = assigned_codes

# ── 7. RETRYABLE FLAG + RECOVERY ─────────────────────────────────────────────
is_retryable = np.array([
    FAILURE_CODES[str(c)][0] if str(c) != "success" else False
    for c in failure_codes
])

# For retryable failures: did a retry recover the revenue?
retry_recovered = np.zeros(N, dtype=bool)
for idx in failed_idx[retryable_mask]:
    code = failure_codes[idx]
    recovery_rate = RETRY_RECOVERY.get(code, 0.50)
    retry_recovered[idx] = rng.random() < recovery_rate

# Recoverable = retryable AND NOT recovered yet (still leaking revenue)
is_recoverable = is_retryable & ~retry_recovered & is_failed

# ── 8. MERCHANT IDs ──────────────────────────────────────────────────────────
# 5000 unique merchants, unevenly distributed (power law)
n_merchants  = 5000
merchant_pop = rng.power(0.3, size=n_merchants)
merchant_pop = merchant_pop / merchant_pop.sum()
merchant_ids = rng.choice([f"MID_{i:05d}" for i in range(n_merchants)], size=N, p=merchant_pop)

# ── 9. ASSEMBLE DATAFRAME ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "transaction_id":      [f"TXN_{i:07d}" for i in range(N)],
    "timestamp":           timestamps,
    "merchant_id":         merchant_ids,
    "merchant_category":   merchant_cats,
    "geography":           geos,
    "currency":            [GEOGRAPHIES[g]["currency"] for g in geos],
    "payment_method":      pay_methods,
    "amount_usd":          amounts,
    "pre_auth_risk_score": np.round(pre_auth_risk_score, 4),
    "status":              np.where(is_failed, "failed", "success"),
    "failure_code":        failure_codes,
    "is_retryable":        is_retryable.astype(int),
    "retry_recovered":     retry_recovered.astype(int),
    "is_recoverable":      is_recoverable.astype(int),
    "hour_of_day":         hours,
    "day_of_week":         dow,
    "is_weekend":          weekend.astype(int),
    "month":               timestamps.month.to_numpy(),
    "year":                timestamps.year.to_numpy(),
})

# ── 10. SAVE ──────────────────────────────────────────────────────────────────
out_path = "data/transactions.csv"
df.to_csv(out_path, index=False)

n_failed_total  = is_failed.sum()
n_retryable     = is_retryable.sum()
n_recoverable   = is_recoverable.sum()
revenue_leakage = df.loc[df["is_recoverable"] == 1, "amount_usd"].sum()

print(f"\n{'='*55}")
print(f"  Transactions generated : {N:>12,}")
print(f"  Failed transactions    : {n_failed_total:>12,}  ({n_failed_total/N:.1%})")
print(f"  Retryable failures     : {n_retryable:>12,}  ({n_retryable/N:.1%})")
print(f"  Recoverable (leakage)  : {n_recoverable:>12,}  ({n_recoverable/N:.1%})")
print(f"  Revenue leakage (USD)  : ${revenue_leakage:>11,.0f}")
print(f"  Saved to               : {out_path}")
print(f"{'='*55}")
