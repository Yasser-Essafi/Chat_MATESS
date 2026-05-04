"""
rebuild_knowledge_base.py
==========================
Génère les documents markdown de la knowledge base depuis fact_statistiques_apf.xlsx
puis reconstruit le vectorstore ChromaDB.

Usage:
    py -3 rebuild_knowledge_base.py
"""

import os, sys, shutil
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
NEW_FILE = os.path.join(BASE, "data", "fact_statistiques_apf.xlsx")
OLD_FILE = os.path.join(BASE, "data", "apf_data.xlsx")
DOCS_DIR = os.path.join(BASE, "knowledge_base", "documents")

TODAY = datetime.now().strftime("%Y-%m-%d %H:%M")

MONTH_FR = {
    1:"Janvier", 2:"Février",  3:"Mars",    4:"Avril",
    5:"Mai",     6:"Juin",     7:"Juillet", 8:"Août",
    9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre",
}

# ── Load data ──────────────────────────────────────────────────────────────
print("Chargement de fact_statistiques_apf.xlsx ...")
df_raw = pd.read_excel(NEW_FILE)
df_raw['date_stat'] = pd.to_datetime(df_raw['date_stat'])
df_raw['mre'] = df_raw['mre'].fillna(0).astype(int)
df_raw['tes'] = df_raw['tes'].fillna(0).astype(int)
df_raw['total'] = df_raw['mre'] + df_raw['tes']
df_raw['year']  = df_raw['date_stat'].dt.year
df_raw['month'] = df_raw['date_stat'].dt.month
df_raw['month_name'] = df_raw['month'].map(MONTH_FR)
df_raw['year_month'] = df_raw['date_stat'].dt.to_period('M').astype(str)

# Filter relevant years (2019 historical + 2023-2026 current)
df = df_raw[df_raw['year'].isin([2019, 2023, 2024, 2025, 2026])].copy()

print(f"  Enregistrements: {len(df):,} | Periode: {df['date_stat'].min().date()} → {df['date_stat'].max().date()}")


# ── Helper ─────────────────────────────────────────────────────────────────
def n(v):
    """Format number with thousands separator."""
    try:
        return f"{int(v):,}"
    except:
        return str(v)

def pct(part, total):
    if total == 0: return "0.0%"
    return f"{100*part/total:.1f}%"

def write_doc(filename, content):
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✅ {filename}")


# ══════════════════════════════════════════════════════════════════════════
# 01 — Data Dictionary
# ══════════════════════════════════════════════════════════════════════════
def gen_01_data_dictionary():
    years = sorted(df['year'].unique().tolist())
    min_d = df['date_stat'].min().strftime('%Y-%m-%d')
    max_d = df['date_stat'].max().strftime('%Y-%m-%d')
    nats  = df['nationalite'].nunique()
    postes = df['poste_frontiere'].nunique()
    regs  = df['region'].nunique()
    conts = df['continent'].nunique()
    total_mre = df['mre'].sum()
    total_tes = df['tes'].sum()
    total     = df['total'].sum()

    content = f"""# 📖 Dictionnaire de Données — Statistiques APF

> Généré le {TODAY}

## Source
- **Fichier** : `fact_statistiques_apf.xlsx`
- **Organisme** : Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire (MTAESS)
- **Fournisseur** : Administration des Postes Frontières (APF) / DGSN
- **Période** : {min_d} → {max_d}
- **Volume** : {n(len(df))} enregistrements
- **Années disponibles** : {', '.join(str(y) for y in years)}

## Schéma

| # | Colonne | Type | Description |
|---|---------|------|-------------|
| 1 | statistiques_apf_id | UUID | Identifiant unique |
| 2 | nationalite | String | Pays de résidence du voyageur ({nats} valeurs) |
| 3 | poste_frontiere | String | Point d'entrée ({postes} postes) |
| 4 | region | String | Région administrative marocaine ({regs} régions) |
| 5 | continent | String | Continent d'origine ({conts} valeurs) |
| 6 | voie | String | Mode d'entrée (3 voies) |
| 7 | date_stat | Date | Date statistique (1er du mois) |
| 8 | mre | Integer | Marocains Résidant à l'Étranger |
| 9 | tes | Float | Touristes Étrangers Séjournistes |

## Totaux globaux (toutes années)

| Indicateur | Valeur |
|-----------|--------|
| Total arrivées | {n(total)} |
| Total MRE | {n(total_mre)} ({pct(total_mre, total)}) |
| Total TES | {n(total_tes)} ({pct(total_tes, total)}) |

## Définitions clés
- **MRE** — Marocains Résidant à l'Étranger : membres de la diaspora marocaine rentrant au pays (famille, vacances, affaires)
- **TES** — Touristes Étrangers Séjournistes : non-Marocains entrant sur le territoire pour au moins une nuitée
- **Voie Aérienne** — Arrivées via aéroports internationaux
- **Voie Maritime** — Arrivées via ports (ferries depuis Espagne, France, Italie)
- **Voie Terrestre** — Arrivées via postes frontières terrestres (Ceuta, Melilla)
- Chaque ligne = agrégat mensuel pour une combinaison (nationalité × poste × région × continent × voie)
- Granularité temporelle : **mensuelle**

## Préfixes postes frontières
- **A** = Aéroport  |  **Port de / P** = Port  |  **T** = Terrestre (Trab)

## Régions (12)
{', '.join(sorted(df['region'].dropna().unique().tolist()))}

## Voies (3)
V Aérienne, V Maritime, V Terrestre
"""
    write_doc("01_data_dictionary.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 02 — Vue d'ensemble statistique avec décomposition mensuelle
# ══════════════════════════════════════════════════════════════════════════
def gen_02_statistical_overview():
    # Annual summary
    annual = df.groupby('year').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum'),
        records=('mre','count')
    ).reset_index()

    annual_rows = "\n".join(
        f"| {r.year} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | "
        f"{pct(r.mre, r.total)} | {n(r.records)} |"
        for _, r in annual.iterrows()
    )

    # Monthly 2025 full year
    m2025 = df[df['year']==2025].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')

    rows_2025 = "\n".join(
        f"| {r.month_name} 2025 | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in m2025.iterrows()
    )
    total_2025 = df[df['year']==2025]['total'].sum()

    # Monthly 2026 (jan-feb)
    m2026 = df[df['year']==2026].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')

    rows_2026 = "\n".join(
        f"| {r.month_name} 2026 | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in m2026.iterrows()
    )

    # Jan/Feb comparison 2024 vs 2025 vs 2026
    comp_janfev = []
    for yr in [2024, 2025, 2026]:
        sub = df[(df['year']==yr) & (df['month'].isin([1,2]))]
        mre_v = sub['mre'].sum()
        tes_v = sub['tes'].sum()
        tot_v = sub['total'].sum()
        comp_janfev.append(f"| Jan-Fév {yr} | {n(mre_v)} | {n(tes_v)} | {n(tot_v)} |")
    rows_comp = "\n".join(comp_janfev)

    # Top 10 nationalities all time
    top_nat = df.groupby('nationalite')['total'].sum().nlargest(10).reset_index()
    rows_nat = "\n".join(
        f"| {r.nationalite} | {n(r.total)} | {pct(r.total, df['total'].sum())} |"
        for _, r in top_nat.iterrows()
    )

    content = f"""# 📊 Vue d'Ensemble Statistique — Arrivées Touristiques au Maroc

> Généré le {TODAY}

## Résumé global
- **Période couverte** : Janvier 2019 → Février 2026
- **Total arrivées (2023-2026)** : {n(df[df['year']>=2023]['total'].sum())}
- **Total MRE** : {n(df[df['year']>=2023]['mre'].sum())} ({pct(df[df['year']>=2023]['mre'].sum(), df[df['year']>=2023]['total'].sum())})
- **Total TES** : {n(df[df['year']>=2023]['tes'].sum())} ({pct(df[df['year']>=2023]['tes'].sum(), df[df['year']>=2023]['total'].sum())})
- **Dernière donnée disponible** : {df['date_stat'].max().strftime('%B %Y')} (Février 2026)

## Évolution annuelle

| Année | MRE | TES | Total | % MRE | Enregistrements |
|-------|-----|-----|-------|-------|----------------|
{annual_rows}

## Décomposition mensuelle 2025 (année complète)

| Mois | MRE | TES | Total |
|------|-----|-----|-------|
{rows_2025}
| **TOTAL 2025** | **{n(df[df['year']==2025]['mre'].sum())}** | **{n(df[df['year']==2025]['tes'].sum())}** | **{n(total_2025)}** |

## Décomposition mensuelle 2026 (données disponibles)

| Mois | MRE | TES | Total |
|------|-----|-----|-------|
{rows_2026}

> ⚠️ Les données 2026 disponibles s'arrêtent à **Février 2026**. Mars 2026 et au-delà ne sont pas encore dans le dataset.

## Comparaison Jan-Fév (2024 / 2025 / 2026)

| Période | MRE | TES | Total |
|---------|-----|-----|-------|
{rows_comp}

## Top 10 pays de résidence (toutes années)

| Pays de résidence | Total | % |
|-------------|-------|---|
{rows_nat}
"""
    write_doc("02_statistical_overview.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 03 — Données mensuelles détaillées 2024
# ══════════════════════════════════════════════════════════════════════════
def gen_03_monthly_2024():
    yr = 2024
    monthly = df[df['year']==yr].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')

    rows = "\n".join(
        f"| {r.month_name} {yr} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.mre, r.total)} |"
        for _, r in monthly.iterrows()
    )

    # Monthly top 3 nationalities for each month
    details = []
    for _, row in monthly.iterrows():
        m = int(row['month'])
        m_name = row['month_name']
        sub = df[(df['year']==yr) & (df['month']==m)]
        top3 = sub.groupby('nationalite')['total'].sum().nlargest(3)
        top3_str = " | ".join([f"{nat}: {n(v)}" for nat, v in top3.items()])
        details.append(f"| {m_name} {yr} | {n(row.total)} | {top3_str} |")

    rows_detail = "\n".join(details)

    content = f"""# 📅 Données Mensuelles Détaillées — 2024

> Généré le {TODAY}

## Vue mensuelle 2024

| Mois | MRE | TES | Total | % MRE |
|------|-----|-----|-------|-------|
{rows}
| **TOTAL {yr}** | **{n(df[df['year']==yr]['mre'].sum())}** | **{n(df[df['year']==yr]['tes'].sum())}** | **{n(df[df['year']==yr]['total'].sum())}** | |

## Top 3 pays de résidence par mois 2024

| Mois | Total | Top 3 pays de résidence |
|------|-------|---------------------|
{rows_detail}

## Observations clés 2024
- **Pic estival** : Juillet-Août 2024 concentrent le flux MRE (Opération Marhaba)
- **Forte saison** : Juillet 2024 — {n(df[(df['year']==2024)&(df['month']==7)]['total'].sum())} arrivées totales
- **Mois le plus faible** : {monthly.loc[monthly['total'].idxmin(), 'month_name']} 2024 — {n(monthly['total'].min())} arrivées
- **Croissance vs 2023** : +{pct(df[df['year']==2024]['total'].sum() - df[df['year']==2023]['total'].sum(), df[df['year']==2023]['total'].sum())} de croissance annuelle
"""
    write_doc("03_monthly_2024.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 04 — Données mensuelles détaillées 2025
# ══════════════════════════════════════════════════════════════════════════
def gen_04_monthly_2025():
    yr = 2025
    monthly = df[df['year']==yr].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')

    rows = "\n".join(
        f"| {r.month_name} {yr} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.mre, r.total)} |"
        for _, r in monthly.iterrows()
    )

    # Monthly top 5 countries of residence
    details = []
    for _, row in monthly.iterrows():
        m = int(row['month'])
        m_name = row['month_name']
        sub = df[(df['year']==yr) & (df['month']==m)]
        top5 = sub.groupby('nationalite')['total'].sum().nlargest(5)
        top5_str = ", ".join([f"{nat} ({n(v)})" for nat, v in top5.items()])
        details.append(f"\n### {m_name} 2025\n- **Total** : {n(row.total)} arrivées (MRE: {n(row.mre)} | TES: {n(row.tes)})\n- **Top 5** : {top5_str}")

    content = f"""# 📅 Données Mensuelles Détaillées — 2025

> Généré le {TODAY}

## Vue mensuelle 2025 (année complète)

| Mois | MRE | TES | Total | % MRE |
|------|-----|-----|-------|-------|
{rows}
| **TOTAL {yr}** | **{n(df[df['year']==yr]['mre'].sum())}** | **{n(df[df['year']==yr]['tes'].sum())}** | **{n(df[df['year']==yr]['total'].sum())}** | |

## Détail mensuel 2025 avec top pays de résidence

{''.join(details)}

## Observations clés 2025
- **Année complète** : {n(df[df['year']==2025]['total'].sum())} arrivées totales en 2025
- **Pic** : {monthly.loc[monthly['total'].idxmax(), 'month_name']} 2025 — {n(monthly['total'].max())} arrivées
- **Creux** : {monthly.loc[monthly['total'].idxmin(), 'month_name']} 2025 — {n(monthly['total'].min())} arrivées
- **Croissance vs 2024** : +{pct(df[df['year']==2025]['total'].sum() - df[df['year']==2024]['total'].sum(), df[df['year']==2024]['total'].sum())}
"""
    write_doc("04_monthly_2025.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 05 — Données mensuelles 2026 (jan-fév)
# ══════════════════════════════════════════════════════════════════════════
def gen_05_monthly_2026():
    yr = 2026
    monthly = df[df['year']==yr].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')

    rows = "\n".join(
        f"| {r.month_name} {yr} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.mre, r.total)} |"
        for _, r in monthly.iterrows()
    )

    # Comparison with same months 2025
    comp_rows = []
    for _, row in monthly.iterrows():
        m = int(row['month'])
        prev = df[(df['year']==2025) & (df['month']==m)]['total'].sum()
        curr = row['total']
        evol = f"+{pct(curr-prev,prev)}" if curr >= prev else pct(curr-prev,prev)
        comp_rows.append(f"| {row.month_name} 2025 | {n(prev)} | {row.month_name} 2026 | {n(curr)} | {evol} |")
    rows_comp = "\n".join(comp_rows)

    # Top 5 nat per month
    details = []
    for _, row in monthly.iterrows():
        m = int(row['month'])
        m_name = row['month_name']
        sub = df[(df['year']==yr) & (df['month']==m)]
        top5 = sub.groupby('nationalite')['total'].sum().nlargest(5)
        top5_str = "\n".join([f"  - {nat}: {n(v)} arrivées" for nat, v in top5.items()])

        # by voie
        voie_g = sub.groupby('voie')['total'].sum()
        voie_str = " | ".join([f"{v}: {n(c)}" for v, c in voie_g.items()])

        # by region top 3
        reg3 = sub.groupby('region')['total'].sum().nlargest(3)
        reg_str = " | ".join([f"{r}: {n(c)}" for r, c in reg3.items()])

        details.append(f"""
### {m_name} 2026
- **Total arrivées** : {n(row.total)}
- **MRE** : {n(row.mre)} ({pct(row.mre, row.total)}) | **TES** : {n(row.tes)} ({pct(row.tes, row.total)})
- **Top 5 pays de résidence** :
{top5_str}
- **Par voie** : {voie_str}
- **Top 3 régions** : {reg_str}
""")

    content = f"""# 📅 Données 2026 — Janvier & Février (Dernières données disponibles)

> Généré le {TODAY}

> ⚠️ **Les données disponibles pour 2026 s'arrêtent à Février 2026.**
> Mars 2026, Avril 2026 et les mois suivants ne sont PAS dans le dataset actuel.

## Vue mensuelle 2026

| Mois | MRE | TES | Total | % MRE |
|------|-----|-----|-------|-------|
{rows}
| **CUMUL Jan-Fév 2026** | **{n(df[df['year']==yr]['mre'].sum())}** | **{n(df[df['year']==yr]['tes'].sum())}** | **{n(df[df['year']==yr]['total'].sum())}** | |

## Comparaison avec les mêmes mois 2025

| Mois N-1 | Total 2025 | Mois 2026 | Total 2026 | Évolution |
|----------|-----------|-----------|-----------|-----------|
{rows_comp}

## Détail par mois
{''.join(details)}

## Conclusion Jan-Fév 2026
- Cumul Jan-Fév 2026 : **{n(df[df['year']==yr]['total'].sum())}** arrivées
- vs Jan-Fév 2025 : **{n(df[(df['year']==2025) & (df['month'].isin([1,2]))]['total'].sum())}** arrivées
- Évolution : {pct(df[df['year']==yr]['total'].sum() - df[(df['year']==2025)&(df['month'].isin([1,2]))]['total'].sum(), df[(df['year']==2025)&(df['month'].isin([1,2]))]['total'].sum())}
"""
    write_doc("05_monthly_2026.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 06 — Analyse par pays de résidence (mensuelle)
# ══════════════════════════════════════════════════════════════════════════
def gen_06_pays_residence_analysis():
    # Global top 20
    top20 = df.groupby('nationalite').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).nlargest(20, 'total').reset_index()

    grand_total = df['total'].sum()
    rows_top20 = "\n".join(
        f"| {r.nationalite} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.total, grand_total)} |"
        for _, r in top20.iterrows()
    )

    # 2025 monthly for top 5 countries of residence
    top5_names = df.groupby('nationalite')['total'].sum().nlargest(5).index.tolist()

    monthly_details = []
    for nat in top5_names:
        sub = df[(df['year']==2025) & (df['nationalite']==nat)]
        monthly = sub.groupby(['month','month_name'])['total'].sum().reset_index().sort_values('month')
        rows_nat = " | ".join([f"{r.month_name[:3]}: {n(r.total)}" for _, r in monthly.iterrows()])
        total_nat = sub['total'].sum()
        monthly_details.append(f"| {nat} | {n(total_nat)} | {rows_nat} |")

    # 2026 Jan-Fev by nationality top 20
    top20_2026 = df[df['year']==2026].groupby('nationalite').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).nlargest(20, 'total').reset_index()

    rows_2026 = "\n".join(
        f"| {r.nationalite} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in top20_2026.iterrows()
    )

    # 2025 by nationality
    top20_2025 = df[df['year']==2025].groupby('nationalite').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).nlargest(20, 'total').reset_index()
    rows_2025 = "\n".join(
        f"| {r.nationalite} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in top20_2025.iterrows()
    )

    content = f"""# 🌍 Analyse par Pays de résidence

> Généré le {TODAY}

## Top 20 pays de résidence — toutes années confondues

| Pays de résidence | MRE | TES | Total | % |
|-------------|-----|-----|-------|---|
{rows_top20}

## Top 20 pays de résidence — 2025 (année complète)

| Pays de résidence | MRE | TES | Total |
|-------------|-----|-----|-------|
{rows_2025}

## Top 20 pays de résidence — Janvier-Février 2026

| Pays de résidence | MRE | TES | Total |
|-------------|-----|-----|-------|
{rows_2026}

## Évolution mensuelle 2025 — Top 5 pays de résidence

| Pays de résidence | Total 2025 | Jan | Fév | Mar | Avr | Mai | Juin | Juil | Août | Sep | Oct | Nov | Déc |
|-------------|-----------|-----|-----|-----|-----|-----|------|------|------|-----|-----|-----|-----|
"""
    # Build month-by-month table for top 5
    for nat in top5_names:
        sub2025 = df[(df['year']==2025) & (df['nationalite']==nat)]
        total_2025 = sub2025['total'].sum()
        months_vals = []
        for m in range(1, 13):
            v = sub2025[sub2025['month']==m]['total'].sum()
            months_vals.append(n(v) if v > 0 else "-")
        content += f"| {nat} | {n(total_2025)} | " + " | ".join(months_vals) + " |\n"

    content += f"""
## Répartition MRE vs TES par pays de résidence (2025)

Les pays de résidence à forte proportion MRE (diaspora marocaine) :
"""
    mre_ratio = df[df['year']==2025].groupby('nationalite').agg(
        mre=('mre','sum'), total=('total','sum')
    )
    mre_ratio['ratio'] = mre_ratio['mre'] / mre_ratio['total'].replace(0, np.nan)
    top_mre = mre_ratio[mre_ratio['total'] > 1000].nlargest(10, 'ratio').reset_index()
    for _, r in top_mre.iterrows():
        content += f"- **{r.nationalite}** : {pct(r.mre, r.total)} MRE\n"

    write_doc("06_pays_residence_analysis.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 07 — Analyse par région (mensuelle)
# ══════════════════════════════════════════════════════════════════════════
def gen_07_regional_analysis():
    # Global by region
    reg_global = df.groupby('region').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    grand = df['total'].sum()
    rows_global = "\n".join(
        f"| {r.region} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.total, grand)} |"
        for _, r in reg_global.iterrows()
    )

    # 2025 monthly by region
    pivot_2025 = df[df['year']==2025].groupby(['region','month'])['total'].sum().unstack(fill_value=0)
    reg_details = []
    for reg in pivot_2025.index:
        row_vals = [n(pivot_2025.loc[reg, m]) if m in pivot_2025.columns else "-" for m in range(1, 13)]
        total_reg = pivot_2025.loc[reg].sum()
        reg_details.append(f"| {reg} | {n(total_reg)} | " + " | ".join(row_vals) + " |")
    rows_2025_monthly = "\n".join(reg_details)

    # 2026 Jan-Fev by region
    reg_2026 = df[df['year']==2026].groupby('region').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    rows_2026 = "\n".join(
        f"| {r.region} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in reg_2026.iterrows()
    )

    content = f"""# 🗺️ Analyse par Région

> Généré le {TODAY}

## Toutes années — Arrivées par région

| Région | MRE | TES | Total | % |
|--------|-----|-----|-------|---|
{rows_global}

## 2025 — Répartition mensuelle par région

| Région | Total | Jan | Fév | Mar | Avr | Mai | Jun | Jul | Aoû | Sep | Oct | Nov | Déc |
|--------|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
{rows_2025_monthly}

## Janvier-Février 2026 — Arrivées par région

| Région | MRE | TES | Total |
|--------|-----|-----|-------|
{rows_2026}

## Observations
- **Tanger-Tétouan-Al Hoceima** : porte d'entrée principale pour V Terrestre et V Maritime
- **Casablanca-Settat** : hub aérien principal (Mohammed V)
- **Marrakech-Safi** : première destination touristique étrangère (TES)
- **Souss-Massa** : Agadir, forte concentration TES
"""
    write_doc("07_regional_analysis.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 08 — Analyse par mode d'entrée (Voie)
# ══════════════════════════════════════════════════════════════════════════
def gen_08_entry_mode_analysis():
    voie_global = df.groupby('voie').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    grand = df['total'].sum()
    rows_global = "\n".join(
        f"| {r.voie} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.total, grand)} | {pct(r.mre, r.total)} |"
        for _, r in voie_global.iterrows()
    )

    # Annual by voie
    annual_voie = df.groupby(['year','voie'])['total'].sum().unstack(fill_value=0)
    rows_annual = "\n".join(
        f"| {yr} | " + " | ".join(n(annual_voie.loc[yr, v]) if v in annual_voie.columns else "-"
                                   for v in ['V Aérienne', 'V Maritime', 'V Terrestre']) + " |"
        for yr in annual_voie.index
    )

    # 2025 monthly by voie
    mv_2025 = df[df['year']==2025].groupby(['month','month_name','voie'])['total'].sum().unstack(fill_value=0).reset_index()
    rows_mv_2025 = "\n".join(
        f"| {r.month_name} 2025 | " + " | ".join(
            n(r[v]) if v in mv_2025.columns else "-"
            for v in ['V Aérienne', 'V Maritime', 'V Terrestre']
        ) + " |"
        for _, r in mv_2025.sort_values('month').iterrows()
    )

    # 2026 monthly by voie
    mv_2026 = df[df['year']==2026].groupby(['month','month_name','voie'])['total'].sum().unstack(fill_value=0).reset_index()
    rows_mv_2026 = "\n".join(
        f"| {r.month_name} 2026 | " + " | ".join(
            n(r[v]) if v in mv_2026.columns else "-"
            for v in ['V Aérienne', 'V Maritime', 'V Terrestre']
        ) + " |"
        for _, r in mv_2026.sort_values('month').iterrows()
    )

    content = f"""# ✈️ Analyse par Mode d'Entrée (Voie)

> Généré le {TODAY}

## Répartition globale

| Voie | MRE | TES | Total | % Total | % MRE |
|------|-----|-----|-------|---------|-------|
{rows_global}

## Évolution annuelle par voie

| Année | V Aérienne | V Maritime | V Terrestre |
|-------|-----------|-----------|------------|
{rows_annual}

## 2025 — Répartition mensuelle par voie

| Mois | V Aérienne | V Maritime | V Terrestre |
|------|-----------|-----------|------------|
{rows_mv_2025}

## 2026 — Répartition mensuelle par voie (Jan-Fév)

| Mois | V Aérienne | V Maritime | V Terrestre |
|------|-----------|-----------|------------|
{rows_mv_2026}

## Observations
- **Voie Aérienne** : domine pour les TES (~70% du total)
- **Voie Maritime** : forte proportion MRE (retours diaspora, ferries Espagne-Maroc)
- **Voie Terrestre** : surtout MRE et flux Ceuta/Melilla
- **Opération Marhaba** (été) : pic voie maritime en Juillet-Août
"""
    write_doc("08_entry_mode_analysis.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 09 — Top postes frontières (mensuel)
# ══════════════════════════════════════════════════════════════════════════
def gen_09_border_posts():
    # Top 15 postes all time
    top15 = df.groupby('poste_frontiere').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum'), voie=('voie','first')
    ).nlargest(15, 'total').reset_index()
    grand = df['total'].sum()
    rows_top15 = "\n".join(
        f"| {r.poste_frontiere} | {r.voie} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.total, grand)} |"
        for _, r in top15.iterrows()
    )

    # Top 10 postes 2025 monthly
    top10_names = df[df['year']==2025].groupby('poste_frontiere')['total'].sum().nlargest(10).index.tolist()
    pivot_2025 = df[(df['year']==2025) & (df['poste_frontiere'].isin(top10_names))].groupby(
        ['poste_frontiere', 'month'])['total'].sum().unstack(fill_value=0)

    rows_post_monthly = "\n".join(
        f"| {p} | {n(pivot_2025.loc[p].sum())} | " +
        " | ".join(n(pivot_2025.loc[p, m]) if m in pivot_2025.columns else "-" for m in range(1,13)) + " |"
        for p in top10_names if p in pivot_2025.index
    )

    # 2026 Jan-Fev by poste
    top10_2026 = df[df['year']==2026].groupby('poste_frontiere').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).nlargest(10, 'total').reset_index()
    rows_2026 = "\n".join(
        f"| {r.poste_frontiere} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in top10_2026.iterrows()
    )

    content = f"""# 🛂 Analyse par Postes Frontières

> Généré le {TODAY}

## Top 15 postes — toutes années

| Poste Frontière | Voie | MRE | TES | Total | % |
|-----------------|------|-----|-----|-------|---|
{rows_top15}

## Top 10 postes 2025 — Répartition mensuelle

| Poste | Total 2025 | Jan | Fév | Mar | Avr | Mai | Jun | Jul | Aoû | Sep | Oct | Nov | Déc |
|-------|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
{rows_post_monthly}

## Top 10 postes — Janvier-Février 2026

| Poste Frontière | MRE | TES | Total |
|-----------------|-----|-----|-------|
{rows_2026}

## Observations
- **A Marrakech-Ménara** : 1er aéroport touristique du Maroc
- **A Casablanca Mohammed V** : hub international principal, fort trafic MRE
- **T Bab Sebta** : principal poste terrestre, quasi exclusivement MRE et résidents
- **Port Tanger Med** : premier port roulier Méditerranée, MRE+TES
"""
    write_doc("09_border_posts_reference.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 10 — Analyse par continent
# ══════════════════════════════════════════════════════════════════════════
def gen_10_continent_analysis():
    cont_global = df.groupby('continent').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    grand = df['total'].sum()
    rows_cont = "\n".join(
        f"| {r.continent} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} | {pct(r.total, grand)} |"
        for _, r in cont_global.iterrows()
    )

    # 2025 by continent
    cont_2025 = df[df['year']==2025].groupby('continent').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    rows_2025 = "\n".join(
        f"| {r.continent} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in cont_2025.iterrows()
    )

    # 2026 Jan-Fev
    cont_2026 = df[df['year']==2026].groupby('continent').agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).sort_values('total', ascending=False).reset_index()
    rows_2026 = "\n".join(
        f"| {r.continent} | {n(r.mre)} | {n(r.tes)} | {n(r.total)} |"
        for _, r in cont_2026.iterrows()
    )

    content = f"""# 🌐 Analyse par Continent

> Généré le {TODAY}

## Répartition globale par continent

| Continent | MRE | TES | Total | % |
|-----------|-----|-----|-------|---|
{rows_cont}

## 2025 — Arrivées par continent

| Continent | MRE | TES | Total |
|-----------|-----|-----|-------|
{rows_2025}

## Janvier-Février 2026 — Arrivées par continent

| Continent | MRE | TES | Total |
|-----------|-----|-----|-------|
{rows_2026}

## Observations
- **Union Européenne** : principal bassin émetteur de TES
- **Autres Europe** : inclut Russie, pays de l'Est hors UE
- **Pays du Golfe** : flux croissant (Arabie Saoudite, EAU)
- **Afrique** : flux limité mais en croissance (MRE Afrique subsaharienne)
"""
    write_doc("10_continent_analysis.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 11 — Analyse MRE vs TES
# ══════════════════════════════════════════════════════════════════════════
def gen_11_mre_analysis():
    # Monthly MRE vs TES 2025
    mre_2025 = df[df['year']==2025].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')
    rows_2025 = "\n".join(
        f"| {r.month_name} 2025 | {n(r.mre)} | {pct(r.mre, r.total)} | {n(r.tes)} | {pct(r.tes, r.total)} |"
        for _, r in mre_2025.iterrows()
    )

    # 2026
    mre_2026 = df[df['year']==2026].groupby(['month','month_name']).agg(
        mre=('mre','sum'), tes=('tes','sum'), total=('total','sum')
    ).reset_index().sort_values('month')
    rows_2026 = "\n".join(
        f"| {r.month_name} 2026 | {n(r.mre)} | {pct(r.mre, r.total)} | {n(r.tes)} | {pct(r.tes, r.total)} |"
        for _, r in mre_2026.iterrows()
    )

    # Top MRE nationalities 2025
    top_mre_nat = df[df['year']==2025].groupby('nationalite')['mre'].sum().nlargest(10).reset_index()
    rows_mre_nat = "\n".join(
        f"| {r.nationalite} | {n(r.mre)} |"
        for _, r in top_mre_nat.iterrows()
    )

    content = f"""# 👥 Analyse MRE vs TES

> Généré le {TODAY}

## Définitions
- **MRE** (Marocains Résidant à l'Étranger) : Marocains vivant hors du Maroc qui rentrent au pays
- **TES** (Touristes Étrangers Séjournistes) : Étrangers non-Marocains visitant le Maroc

## 2025 — MRE vs TES par mois

| Mois | MRE | % MRE | TES | % TES |
|------|-----|-------|-----|-------|
{rows_2025}

## Janvier-Février 2026 — MRE vs TES

| Mois | MRE | % MRE | TES | % TES |
|------|-----|-------|-----|-------|
{rows_2026}

## Top 10 pays de résidence MRE (2025)

| Pays de résidence | MRE 2025 |
|-------------|----------|
{rows_mre_nat}

## Saisonnalité MRE
- **Pic MRE** : Juillet-Août (Opération Marhaba — retour diaspora)
- **Creux MRE** : Janvier-Mars (hors vacances scolaires)
- France et Espagne = 60%+ des MRE (forte diaspora dans ces pays)

## Opération Marhaba
Programme annuel du Groupe Al Barid Bank et MTAESS pour faciliter le retour des MRE :
- Dispositif actif juin-septembre
- Renforce les arrivées maritimes (ferries Tanger Med, Tanger Ville)
- Coordination avec les autorités douanières et policières
"""
    write_doc("11_mre_analysis.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 12 — Comparaison interannuelle
# ══════════════════════════════════════════════════════════════════════════
def gen_12_year_comparison():
    # Monthly comparison 2023-2026 by month (for common months)
    monthly_all = df.groupby(['year','month','month_name']).agg(
        total=('total','sum')
    ).reset_index()

    # Pivot by month for side-by-side comparison
    pivot = monthly_all.pivot_table(index=['month','month_name'], columns='year', values='total', fill_value=0)
    pivot = pivot.sort_index(level=0)

    years = sorted([y for y in [2023,2024,2025,2026] if y in pivot.columns])
    header = " | ".join(str(y) for y in years)
    rows_pivot = "\n".join(
        f"| {MONTH_FR.get(int(m), m)} | " + " | ".join(
            n(pivot.loc[(m, mn), y]) if y in pivot.columns else "-"
            for y in years
        ) + " |"
        for (m, mn) in pivot.index
    )

    # Growth rates
    growth_rows = []
    for yr in [2024, 2025]:
        prev = df[df['year']==yr-1]['total'].sum()
        curr = df[df['year']==yr]['total'].sum()
        g = (curr - prev) / prev * 100
        sign = "+" if g >= 0 else ""
        growth_rows.append(f"| {yr-1} → {yr} | {n(prev)} | {n(curr)} | {sign}{g:.1f}% |")
    # 2026 partial vs same period 2025
    t2026 = df[df['year']==2026]['total'].sum()
    months_2026 = sorted(df[df['year']==2026]['month'].unique())
    t2025_same = df[(df['year']==2025) & (df['month'].isin(months_2026))]['total'].sum()
    g2026 = (t2026 - t2025_same) / t2025_same * 100 if t2025_same > 0 else 0
    growth_rows.append(f"| 2025 (Jan-Fév) → 2026 (Jan-Fév) | {n(t2025_same)} | {n(t2026)} | +{g2026:.1f}% |")
    rows_growth = "\n".join(growth_rows)

    col_header = "| Mois | " + " | ".join(str(y) for y in years) + " |"
    col_sep    = "|------|" + "|".join(["------"]*len(years)) + "|"

    content = f"""# 📈 Comparaison Interannuelle 2023–2026

> Généré le {TODAY}

## Arrivées totales par année

| Année | Total arrivées | MRE | TES |
|-------|----------------|-----|-----|
"""
    for yr in [2023,2024,2025,2026]:
        sub = df[df['year']==yr]
        label = f"{yr} ({'Jan-Fév' if yr==2026 else 'complet'})"
        content += f"| {label} | {n(sub['total'].sum())} | {n(sub['mre'].sum())} | {n(sub['tes'].sum())} |\n"

    content += f"""
## Comparaison mois par mois

{col_header}
{col_sep}
{rows_pivot}

## Taux de croissance annuelle

| Période | Année N-1 | Année N | Croissance |
|---------|-----------|---------|-----------|
{rows_growth}

## Tendances clés
- Croissance soutenue depuis 2023 (+{pct(df[df['year']==2024]['total'].sum()-df[df['year']==2023]['total'].sum(), df[df['year']==2023]['total'].sum())} en 2024)
- 2025 confirme la dynamique post-Covid avec {n(df[df['year']==2025]['total'].sum())} arrivées totales
- Début 2026 : trajectoire positive vs 2025
"""
    write_doc("12_year_comparison.md", content)


# ══════════════════════════════════════════════════════════════════════════
# 13 — KPI & Définitions (conservé avec mise à jour)
# ══════════════════════════════════════════════════════════════════════════
def gen_13_kpi_definitions():
    content = f"""# 📐 Définitions KPI — Tourisme Maroc

> Généré le {TODAY}

## KPIs principaux STATOUR

| KPI | Définition | Source |
|-----|-----------|--------|
| Arrivées totales | MRE + TES entrant sur le territoire | APF / DGSN |
| Arrivées MRE | Marocains résidant à l'étranger | APF / DGSN |
| Arrivées TES | Touristes étrangers séjournistes | APF / DGSN |
| Taux MRE | MRE / Total arrivées | Calculé |
| Croissance YoY | (N - N-1) / N-1 × 100 | Calculé |
| Part de marché pays de résidence | Pays de résidence / Total × 100 | Calculé |
| Indice saisonnalité | Mois / (Total annuel / 12) | Calculé |

## Objectifs Vision 2030

| Cible | Valeur |
|-------|--------|
| Arrivées touristiques 2030 | 26 millions de TES |
| Capacité hôtelière | 260 000 lits |
| Emplois touristiques | 800 000 |
| Recettes touristiques | 120 Mds MAD |

## Saisons touristiques Maroc

| Saison | Mois | Caractéristiques |
|--------|------|-----------------|
| Haute | Mars-Mai, Sept-Oct | Températures idéales, Culturel |
| Estivale | Juin-Août | Pic MRE (Marhaba), Balnéaire |
| Basse | Nov-Fév | Faible affluence, Affaires |
| Hivernale Montagne | Déc-Fév | Ski Atlas, Désert |

## Postes frontières — Classification

| Préfixe | Type | Exemples |
|---------|------|---------|
| A ... | Aéroport | A Casablanca Mohammed V, A Marrakech-Ménara |
| Port de ... | Port maritime | Port de Tanger Med, Port de Tanger Ville |
| T ... | Terrestre | T Bab Sebta, T Béni Anzar |
"""
    write_doc("13_kpi_definitions.md", content)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — Generate + update data file + rebuild vectorstore
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STATOUR — Reconstruction de la Knowledge Base")
    print("="*60 + "\n")

    # Step 1 — Update apf_data.xlsx with new file
    print("Étape 1/3 — Mise à jour du fichier de données analytics...")
    shutil.copy2(NEW_FILE, OLD_FILE)
    print(f"  ✅ apf_data.xlsx mis à jour ({len(df):,} lignes, jusqu'à {df['date_stat'].max().strftime('%B %Y')})")

    # Step 2 — Generate all markdown documents
    print("\nÉtape 2/3 — Génération des documents knowledge base...")
    os.makedirs(DOCS_DIR, exist_ok=True)

    gen_01_data_dictionary()
    gen_02_statistical_overview()
    gen_03_monthly_2024()
    gen_04_monthly_2025()
    gen_05_monthly_2026()
    gen_06_pays_residence_analysis()
    gen_07_regional_analysis()
    gen_08_entry_mode_analysis()
    gen_09_border_posts()
    gen_10_continent_analysis()
    gen_11_mre_analysis()
    gen_12_year_comparison()
    gen_13_kpi_definitions()

    print(f"  13 documents générés dans {DOCS_DIR}")

    # Remove old docs that no longer exist
    old_docs = [
        "03_kpi_definitions.md", "04_pays_residence_analysis.md",
        "05_border_posts_reference.md", "06_regional_analysis.md",
        "07_temporal_trends.md", "08_continent_analysis.md",
        "09_entry_mode_analysis.md", "10_morocco_tourism_context.md",
        "11_statour_platform_guide.md", "12_mre_analysis.md",
    ]
    for old in old_docs:
        p = os.path.join(DOCS_DIR, old)
        if os.path.exists(p):
            os.remove(p)
            print(f"  🗑️  Supprimé: {old}")

    # Step 3 — Rebuild ChromaDB vectorstore
    print("\nÉtape 3/3 — Reconstruction du vectorstore ChromaDB...")
    from tools.rag_tools import RAGManager
    rag = RAGManager()
    result = rag.build_vectorstore(force_rebuild=True)
    print(f"  Status : {result['status']}")
    print(f"  {result['message']}")

    print("\n" + "="*60)
    print("  ✅ Knowledge base reconstruite avec succès !")
    print(f"  Documents : 13 fichiers markdown")
    print(f"  Chunks    : {result.get('total_chunks', '?')}")
    print(f"  Données   : Janvier 2019 → {df['date_stat'].max().strftime('%B %Y')}")
    print("="*60 + "\n")
