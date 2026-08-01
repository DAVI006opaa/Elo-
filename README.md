# Elo

An engine for reading regulatory data that has already been audited.

Every regulated institution is legally required to publish detailed facts about
itself, signed by an auditor who carries liability for getting it wrong. That
corpus is public, mandated, and structurally unreadable — buried in footnotes,
named differently by every filer, dispersed across hundreds of thousands of
documents. Nobody joins it.

Elo normalizes that terminology across filers and joins it into a comparable
index per sector. One engine, remodeled per regulator.

**Pilot — Elo Varejo:** inventory shrink in Brazilian listed retail, read from
DFP/ITR filings submitted to the securities regulator (CVM).

## What's here

| File | What it is |
|---|---|
| `Elo_Memo_EN.pdf` | Two-page working memo — thesis, verified extraction, market data, modeled economics, honest ceiling |
| `elo_monte_carlo.py` | 20,000-scenario simulation across four regulated verticals. Reproducible (seed 42); every distribution assumption documented and separated from cited market data |
| `Elo_Varejo_Analise_Financeira.xlsx` | The comparative index itself. One filer fully reconciled against its audited source; remaining rows structured and explicitly marked as pending extraction |

## Status

Three weeks old. One filer reconciled (Magazine Luiza FY2023 — R$179.561M
provision on R$7.677B gross inventory, 2.34%, audited by EY; net inventory ties
exactly to the published figure). No customers, no revenue, not incorporated.

Four gates before quoting a price to anyone: five filers extracted, the
financial-distress case study closed quantitatively, pricing validated with two
operators, one commercial conversation converted to intent to pay.

## Running the simulation

```bash
pip install numpy pandas matplotlib
python elo_monte_carlo.py
```

Davi Lucas dos S. B. da Silva — August 2026
