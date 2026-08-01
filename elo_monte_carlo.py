"""
Elo -- Motor Quantitativo do CFO (SEBRADS)
Simulacao de Monte Carlo de ARR/Valuation + Unit Economics + TAM/SAM/SOM
Todas as premissas sao explicitadas e claramente rotuladas como premissas de modelagem do CFO,
distintas dos dados de mercado reais e citados (ver relatorio para fontes).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json, os

OUT = "/sessions/cool-youthful-pasteur/mnt/outputs/elo_out"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

N_TRIALS = 20000
N_MONTHS = 36
months = np.arange(1, N_MONTHS + 1)

NAVY = "#1E2761"
GRAY = "#5B6472"
GOLD = "#C79A3E"
GREEN = "#2E7D32"
RED = "#B3261E"
LIGHT = "#E8EAF0"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
})

# ----------------------------------------------------------------------------
# 1) DEFINICAO DAS VERTICAIS (premissas de modelagem do CFO)
# ----------------------------------------------------------------------------
verticals = {
    "Varejo (CVM)": dict(
        launch_tri=(0, 0, 0),           # ja piloto ativo (Elo Varejo)
        sam=70,                          # contas-alvo estimadas (ver TAM/SAM/SOM)
        acv_base=45000,                  # ACV medio ponderado (R$/ano)
        ramp_start=0.5, ramp_end=4.0, ramp_q=8,
        cac_start=9000, cac_mature=28000,
    ),
    "Seguros (SUSEP)": dict(
        launch_tri=(6, 9, 14),
        sam=130,
        acv_base=85000,
        ramp_start=0.3, ramp_end=3.0, ramp_q=8,
        cac_start=14000, cac_mature=40000,
    ),
    "Bancos/Fintechs (BACEN)": dict(
        launch_tri=(12, 15, 20),
        sam=100,
        acv_base=70000,
        ramp_start=0.3, ramp_end=2.5, ramp_q=8,
        cac_start=13000, cac_mature=38000,
    ),
    "Utilities (ANEEL)": dict(
        launch_tri=(18, 22, 28),
        sam=40,
        acv_base=65000,
        ramp_start=0.2, ramp_end=1.8, ramp_q=8,
        cac_start=15000, cac_mature=35000,
    ),
}

price_lo, price_hi = 0.80, 1.05
margin_lo, margin_hi = 0.87, 0.93
churn_a, churn_b = 6, 44  # Beta -> media ~12% a.a.
mult_tri = (3, 6, 12)
strategic_prob = 0.05
strategic_range = (12, 23)  # cauda: comp Bureau van Dijk/Moody's (22.7x)

# ----------------------------------------------------------------------------
# 2) DRAWS POR TRIAL (vetorizado)
# ----------------------------------------------------------------------------
price_factor = rng.uniform(price_lo, price_hi, N_TRIALS)
gross_margin = rng.uniform(margin_lo, margin_hi, N_TRIALS)
churn_annual = rng.beta(churn_a, churn_b, N_TRIALS)
churn_monthly = 1 - (1 - churn_annual) ** (1 / 12)

is_strategic = rng.uniform(0, 1, N_TRIALS) < strategic_prob
mult_normal = rng.triangular(mult_tri[0], mult_tri[1], mult_tri[2], N_TRIALS)
mult_strategic = rng.uniform(strategic_range[0], strategic_range[1], N_TRIALS)
arr_multiple = np.where(is_strategic, mult_strategic, mult_normal)

arr_total = np.zeros((N_TRIALS, N_MONTHS))
logos_by_vertical = {}
arr_by_vertical = {}
cac_effective_by_vertical = {}

for vname, v in verticals.items():
    lo, mode, hi = v["launch_tri"]
    if lo == hi:
        launch = np.full(N_TRIALS, lo, dtype=float)
    else:
        launch = rng.triangular(lo, mode, hi, N_TRIALS)

    logos_active = np.zeros(N_TRIALS)
    arr_series = np.zeros((N_TRIALS, N_MONTHS))
    acv = v["acv_base"] * price_factor

    for m in range(N_MONTHS):
        month_num = m + 1
        active_mask = month_num >= launch
        months_since = np.clip(month_num - launch, 0, None)
        quarter_since = np.floor(months_since / 3.0)
        progress = np.clip(quarter_since / v["ramp_q"], 0, 1.0)
        mean_new_q = v["ramp_start"] + progress * (v["ramp_end"] - v["ramp_start"])
        mean_new_m = np.where(active_mask, mean_new_q / 3.0, 0.0)

        remaining = np.clip(v["sam"] - logos_active, 0, None)
        # draw new logos (poisson), cap at remaining SAM
        new_logos_raw = rng.poisson(np.clip(mean_new_m, 0, None))
        new_logos = np.minimum(new_logos_raw, remaining)
        new_logos = np.where(active_mask, new_logos, 0)

        churned = logos_active * churn_monthly
        logos_active = np.clip(logos_active + new_logos - churned, 0, None)

        arr_series[:, m] = logos_active * acv

    arr_total += arr_series
    logos_by_vertical[vname] = logos_active.copy()  # logos ativos ao final (mes 36)
    arr_by_vertical[vname] = arr_series
    # CAC efetivo no steady state (media start/mature ponderada pela penetracao do SAM)
    penet = np.clip(logos_active / v["sam"], 0, 1)
    cac_eff = v["cac_start"] + penet * (v["cac_mature"] - v["cac_start"])
    cac_effective_by_vertical[vname] = cac_eff

# ----------------------------------------------------------------------------
# 3) LTV / CAC / PAYBACK por trial (consistente com os mesmos draws)
# ----------------------------------------------------------------------------
blended_acv_end = arr_total[:, -1] / np.maximum(sum(logos_by_vertical[v] for v in verticals), 1e-9)
blended_cac = np.mean([cac_effective_by_vertical[v] for v in verticals], axis=0)
avg_lifetime_years = 1 / np.clip(churn_annual, 0.02, None)
ltv = blended_acv_end * gross_margin * avg_lifetime_years
ltv_cac = ltv / blended_cac
payback_months = blended_cac / (blended_acv_end * gross_margin / 12)

# Cenario de estresse: CAC "full-loaded" (time do founder deixa de ser gratis - custo de
# time comercial dedicado: AE + SDR + marketing amortizados por logo fechado)
loaded_factor = rng.uniform(2.5, 4.0, N_TRIALS)
blended_cac_loaded = blended_cac * loaded_factor
ltv_cac_loaded = ltv / blended_cac_loaded
payback_months_loaded = blended_cac_loaded / (blended_acv_end * gross_margin / 12)

# ----------------------------------------------------------------------------
# 4) VALUATION (mes 24 e mes 36)
# ----------------------------------------------------------------------------
arr_m24 = arr_total[:, 23]
arr_m36 = arr_total[:, 35]
arr_m12 = arr_total[:, 11]
valuation_m24 = arr_m24 * arr_multiple
valuation_m36 = arr_m36 * arr_multiple

# ----------------------------------------------------------------------------
# 5) SUMARIOS / PERCENTIS
# ----------------------------------------------------------------------------
def pct_table(arr2d_or_1d, months_list=None):
    if months_list is not None:
        rows = []
        for mm in months_list:
            col = arr2d_or_1d[:, mm - 1]
            rows.append({
                "mes": mm,
                "P10": np.percentile(col, 10),
                "P25": np.percentile(col, 25),
                "P50": np.percentile(col, 50),
                "media": np.mean(col),
                "P75": np.percentile(col, 75),
                "P90": np.percentile(col, 90),
            })
        return pd.DataFrame(rows)
    else:
        col = arr2d_or_1d
        return {
            "P10": np.percentile(col, 10), "P25": np.percentile(col, 25),
            "P50": np.percentile(col, 50), "media": np.mean(col),
            "P75": np.percentile(col, 75), "P90": np.percentile(col, 90),
        }

arr_percentiles = pct_table(arr_total, [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
arr_percentiles.to_csv(f"{OUT}/arr_percentiles.csv", index=False)

milestones = [250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000]
milestone_rows = []
for ms in milestones:
    row = {"marco_ARR": ms}
    for mm in [12, 18, 24, 30, 36]:
        prob = np.mean(arr_total[:, mm - 1] >= ms) * 100
        row[f"prob_mes_{mm}_%"] = round(prob, 1)
    milestone_rows.append(row)
milestone_df = pd.DataFrame(milestone_rows)
milestone_df.to_csv(f"{OUT}/milestones.csv", index=False)

val_summary = {
    "valuation_m24": pct_table(valuation_m24),
    "valuation_m36": pct_table(valuation_m36),
    "arr_m12": pct_table(arr_m12),
    "arr_m24": pct_table(arr_m24),
    "arr_m36": pct_table(arr_m36),
    "ltv_cac": pct_table(ltv_cac),
    "payback_months": pct_table(payback_months),
    "ltv": pct_table(ltv),
    "ltv_cac_loaded": pct_table(ltv_cac_loaded),
    "payback_months_loaded": pct_table(payback_months_loaded),
    "blended_cac": pct_table(blended_cac),
    "blended_cac_loaded": pct_table(blended_cac_loaded),
}
with open(f"{OUT}/valuation_summary.json", "w") as f:
    json.dump(val_summary, f, indent=2, default=float)

# vertical breakdown at month 36 (mean ARR)
vert_rows = []
for vname in verticals:
    col = arr_by_vertical[vname][:, 35]
    vert_rows.append({
        "vertical": vname,
        "ARR_m36_P10": np.percentile(col, 10),
        "ARR_m36_P50": np.percentile(col, 50),
        "ARR_m36_media": np.mean(col),
        "ARR_m36_P90": np.percentile(col, 90),
        "logos_ativos_m36_P50": np.percentile(logos_by_vertical[vname], 50),
    })
vert_df = pd.DataFrame(vert_rows)
vert_df.to_csv(f"{OUT}/vertical_breakdown_m36.csv", index=False)

print("=== ARR percentiles (meses-chave) ===")
print(arr_percentiles.to_string(index=False))
print("\n=== Probabilidade de marcos de ARR (%) ===")
print(milestone_df.to_string(index=False))
print("\n=== Breakdown por vertical (mes 36) ===")
print(vert_df.to_string(index=False))
print("\n=== Valuation (M24/M36) e Unit Economics -- resumo ===")
for k, v in val_summary.items():
    print(k, {kk: round(vv, 2) for kk, vv in v.items()})

# ----------------------------------------------------------------------------
# 6) SENSIBILIDADE (correlacao com ARR mes 36 e Valuation mes 36)
# ----------------------------------------------------------------------------
sens_rows = []
target = arr_m36
for name, arr_ in [
    ("Price realization", price_factor),
    ("Gross margin", gross_margin),
    ("Churn anual", churn_annual),
    ("Multiplo ARR (valuation)", arr_multiple),
]:
    corr = np.corrcoef(arr_, target)[0, 1] if name != "Multiplo ARR (valuation)" else np.corrcoef(arr_, valuation_m36)[0, 1]
    sens_rows.append({"driver": name, "correlacao_com_resultado": corr})
for vname in verticals:
    corr = np.corrcoef(arr_by_vertical[vname][:, 35], target)[0, 1]
    sens_rows.append({"driver": f"ARR vertical: {vname}", "correlacao_com_resultado": corr})
sens_df = pd.DataFrame(sens_rows).sort_values("correlacao_com_resultado", key=abs, ascending=False)
sens_df.to_csv(f"{OUT}/sensitivity.csv", index=False)
print("\n=== Sensibilidade (correlacao com ARR/Valuation mes 36) ===")
print(sens_df.to_string(index=False))

# save raw arrays needed for charts
np.save(f"{OUT}/arr_total.npy", arr_total)
np.save(f"{OUT}/valuation_m24.npy", valuation_m24)
np.save(f"{OUT}/valuation_m36.npy", valuation_m36)
np.save(f"{OUT}/ltv_cac.npy", ltv_cac)
np.save(f"{OUT}/payback_months.npy", payback_months)
def _safe(name):
    return name.split(" ")[0].replace("/", "-")

for vname in verticals:
    np.save(f"{OUT}/arr_vert_{_safe(vname)}.npy", arr_by_vertical[vname])

print("\nOK - modelo executado com sucesso.")
