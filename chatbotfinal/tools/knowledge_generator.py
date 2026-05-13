"""
STATOUR Knowledge Base Generator
=================================
Generates 12 comprehensive markdown documents from APF tourism data
for the Researcher Agent's RAG system.

Usage:
    python tools/knowledge_generator.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import APF_DATA_PATH, DOCUMENTS_DIR


# ===========================================================================
# Helpers
# ===========================================================================
def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (int, np.integer)):
                cells.append(f"{val:,}")
            elif isinstance(val, float):
                cells.append(f"{val:,.1f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def fmt(n) -> str:
    if isinstance(n, (int, np.integer)):
        return f"{n:,}"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return str(n)


# ===========================================================================
# Generator Class
# ===========================================================================
class KnowledgeGenerator:

    def __init__(self):
        print("📊 Loading APF data...")
        self.df = pd.read_excel(APF_DATA_PATH)
        self.df["date_stat"] = pd.to_datetime(self.df["date_stat"])
        self.df["year"] = self.df["date_stat"].dt.year
        self.df["month"] = self.df["date_stat"].dt.month
        self.df["year_month"] = self.df["date_stat"].dt.strftime("%Y-%m")
        self.df["mre"] = self.df["mre"].fillna(0).astype(int)
        self.df["tes"] = self.df["tes"].fillna(0).astype(int)
        self.output_dir = DOCUMENTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_mre = int(self.df["mre"].sum())
        self.total_tes = int(self.df["tes"].sum())
        self.total_all = self.total_mre + self.total_tes
        self.date_min = self.df["date_stat"].min()
        self.date_max = self.df["date_stat"].max()
        self.years = sorted(self.df["year"].unique())
        print(f"   ✅ {len(self.df):,} records | {self.date_min.date()} → {self.date_max.date()}")

    def generate_all(self):
        print("\n" + "=" * 60)
        print("📚 STATOUR — Knowledge Base Generation")
        print("=" * 60 + "\n")
        docs = [
            ("01_data_dictionary.md",        self._gen_data_dictionary),
            ("02_statistical_overview.md",    self._gen_statistical_overview),
            ("03_kpi_definitions.md",         self._gen_kpi_definitions),
            ("04_pays_residence_analysis.md",    self._gen_pays_residence_analysis),
            ("05_border_posts_reference.md",  self._gen_border_posts),
            ("06_regional_analysis.md",       self._gen_regional_analysis),
            ("07_temporal_trends.md",         self._gen_temporal_trends),
            ("08_continent_analysis.md",      self._gen_continent_analysis),
            ("09_entry_mode_analysis.md",     self._gen_entry_mode),
            ("10_morocco_tourism_context.md", self._gen_tourism_context),
            ("11_statour_platform_guide.md",  self._gen_statour_guide),
            ("12_mre_analysis.md",            self._gen_mre_analysis),
        ]
        ok = 0
        for fname, func in docs:
            try:
                content = func()
                path = os.path.join(self.output_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✅ {fname}")
                ok += 1
            except Exception as exc:
                print(f"  ❌ {fname}: {exc}")
                import traceback; traceback.print_exc()
        print(f"\n{'=' * 60}")
        print(f"✅ Generated {ok}/{len(docs)} documents → {self.output_dir}")
        print("=" * 60)

    # ===================================================================
    #  01 — Data Dictionary
    # ===================================================================
    def _gen_data_dictionary(self) -> str:
        df = self.df
        nat_top = ", ".join(df["nationalite"].value_counts().head(10).index.tolist())
        pf_top = ", ".join(df["poste_frontiere"].value_counts().head(8).index.tolist())
        regions = ", ".join(sorted(df["region"].dropna().unique()))
        continents = ", ".join(sorted(df["continent"].dropna().unique()))
        voies = ", ".join(sorted(df["voie"].dropna().unique()))

        return f"""# 📖 Dictionnaire de Données — Statistiques APF

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Source
- **Fichier** : `apf_data.xlsx`
- **Organisme** : Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire
- **Fournisseur** : Administration des Postes Frontières (APF) / DGSN
- **Période** : {self.date_min.strftime("%Y-%m-%d")} → {self.date_max.strftime("%Y-%m-%d")}
- **Volume** : {len(df):,} enregistrements

## Schéma

| # | Colonne | Type | Description |
|---|---------|------|-------------|
| 1 | statistiques_apf_id | UUID | Identifiant unique |
| 2 | nationalite | String | Pays de résidence du voyageur ({df['nationalite'].nunique()} valeurs) |
| 3 | poste_frontiere | String | Point d'entrée ({df['poste_frontiere'].nunique()} postes) |
| 4 | region | String | Région administrative marocaine ({df['region'].nunique()} régions) |
| 5 | continent | String | Continent d'origine ({df['continent'].nunique()} valeurs) |
| 6 | voie | String | Mode d'entrée ({df['voie'].nunique()} voies) |
| 7 | date_stat | Date | Date statistique (1er du mois) |
| 8 | mre | Integer | Marocains Résidant à l'Étranger |
| 9 | tes | Integer | Touristes Étrangers Séjournistes |

## Valeurs de référence
- **Pays de résidences (top 10)** : {nat_top}
- **Postes (top 8)** : {pf_top}
- **Régions** : {regions}
- **Continents** : {continents}
- **Voies** : {voies}

## Conventions
- Préfixes postes : **A** = Aéroport, **P** = Port, **PF** = Poste Frontière terrestre
- **MRE** = Marocains Résidant à l'Étranger (~5 millions de diaspora)
- **TES** = Touristes Étrangers Séjournistes (non-marocains)
- Chaque ligne = agrégat mensuel pour une combinaison (pays de résidence, poste, région, continent, voie)
- Granularité temporelle : **mensuelle**
"""

    # ===================================================================
    #  02 — Statistical Overview
    # ===================================================================
    def _gen_statistical_overview(self) -> str:
        df = self.df

        yearly = df.groupby("year").agg(
            MRE=("mre", "sum"), TES=("tes", "sum"), Enregistrements=("mre", "count")
        ).reset_index()
        yearly["Total"] = yearly["MRE"] + yearly["TES"]
        yearly = yearly.rename(columns={"year": "Année"})
        yearly_t = df_to_md(yearly)

        nat = df.groupby("nationalite").agg(mre=("mre", "sum"), tes=("tes", "sum")).reset_index()
        nat["total"] = nat["mre"] + nat["tes"]
        nat = nat.sort_values("total", ascending=False).head(15)
        nat["pct"] = (nat["total"] / self.total_all * 100).round(1)
        nat_t = df_to_md(nat.rename(columns={
            "nationalite": "Pays de résidence", "mre": "MRE", "tes": "TES",
            "total": "Total", "pct": "%"
        }))

        bp = df.groupby("poste_frontiere").agg(mre=("mre", "sum"), tes=("tes", "sum")).reset_index()
        bp["total"] = bp["mre"] + bp["tes"]
        bp = bp.sort_values("total", ascending=False).head(10)
        bp["pct"] = (bp["total"] / self.total_all * 100).round(1)
        bp_t = df_to_md(bp.rename(columns={
            "poste_frontiere": "Poste Frontière", "mre": "MRE", "tes": "TES",
            "total": "Total", "pct": "%"
        }))

        return f"""# 📊 Vue d'Ensemble Statistique — Arrivées Touristiques au Maroc

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Résumé global
- **Total arrivées** : {fmt(self.total_all)}
- **Total MRE** : {fmt(self.total_mre)} ({self.total_mre/self.total_all*100:.1f}%)
- **Total TES** : {fmt(self.total_tes)} ({self.total_tes/self.total_all*100:.1f}%)
- **Période** : {self.date_min.strftime("%Y-%m")} → {self.date_max.strftime("%Y-%m")}
- **Nb enregistrements** : {len(df):,}

## Évolution annuelle

{yearly_t}

## Top 15 pays de résidence (toutes années confondues)

{nat_t}

## Top 10 postes frontière

{bp_t}
"""

    # ===================================================================
    #  03 — KPI Definitions
    # ===================================================================
    def _gen_kpi_definitions(self) -> str:
        return f"""# 📐 Définitions des KPIs — STATOUR

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## KPIs Principaux

### 1. Total Arrivées (TA)
- **Formule** : `TA = Σ(mre) + Σ(tes)`
- **Description** : Nombre total de voyageurs ayant franchi un poste frontière marocain
- **Unité** : Nombre de personnes
- **Fréquence** : Mensuelle, trimestrielle, annuelle

### 2. Arrivées MRE
- **Formule** : `Σ(mre)`
- **Description** : Marocains Résidant à l'Étranger revenant au Maroc
- **Importance** : Représente la diaspora (~5M personnes), fort impact économique (transferts, immobilier, consommation)

### 3. Arrivées TES
- **Formule** : `Σ(tes)`
- **Description** : Touristes Étrangers Séjournistes (non-marocains)
- **Importance** : Indicateur direct de l'attractivité touristique internationale

### 4. Part MRE (%)
- **Formule** : `(Σ mre / TA) × 100`
- **Description** : Proportion des MRE dans le total des arrivées

### 5. Part TES (%)
- **Formule** : `(Σ tes / TA) × 100`
- **Description** : Proportion des touristes étrangers

### 6. Taux de Croissance Annuel (TCA)
- **Formule** : `((TA_année_N - TA_année_N-1) / TA_année_N-1) × 100`
- **Description** : Variation annuelle en pourcentage

### 7. Indice de Saisonnalité (IS)
- **Formule** : `(Arrivées_mois / Moyenne_mensuelle_annuelle) × 100`
- **Description** : Mesure la concentration saisonnière (>100 = mois fort, <100 = mois faible)
- **Mois forts typiques** : Juillet, Août (MRE), Mars-Avril et Oct-Nov (TES)

### 8. Indice de Concentration par Pays de résidence (ICN)
- **Formule** : Herfindahl–Hirschman Index sur les parts par pays de résidence
- **Description** : Mesure la diversification des marchés sources

### 9. Taux de Pénétration par Voie
- **Formule** : `(Arrivées_voie / TA) × 100`
- **Description** : Répartition entre aérien, maritime, terrestre

### 10. Score Régional
- **Formule** : `(Arrivées_région / TA) × 100`
- **Description** : Poids relatif de chaque région dans les arrivées totales

## KPIs Avancés (Phase 2)

| KPI | Description |
|-----|-------------|
| Durée Moyenne de Séjour | Nécessite données hébergement (non disponible dans APF) |
| Recettes Touristiques | Source : Office des Changes |
| Taux d'Occupation Hôtelier | Source : Enquêtes hébergement |
| Dépense Moyenne par Touriste | Recettes / Nb TES |
| Nuitées Touristiques | Source : Observatoire du Tourisme |

## Sources de données complémentaires
- **APF (DGSN)** : Arrivées aux postes frontières ← *source actuelle*
- **Bank Al-Maghrib / Office des Changes** : Recettes en devises
- **Observatoire du Tourisme** : Nuitées, taux d'occupation
- **HCP** : Comptes satellites du tourisme
- **ONDA** : Trafic aéroportuaire
"""

    # ===================================================================
    #  04 — Pays de Residence Analysis
    # ===================================================================
    def _gen_pays_residence_analysis(self) -> str:
        df = self.df

        # Overall top 20
        nat = df.groupby("nationalite").agg(mre=("mre", "sum"), tes=("tes", "sum")).reset_index()
        nat["total"] = nat["mre"] + nat["tes"]
        nat = nat.sort_values("total", ascending=False)
        top20 = nat.head(20).copy()
        top20["pct"] = (top20["total"] / self.total_all * 100).round(1)
        top20["rank"] = range(1, 21)
        top20_t = df_to_md(top20[["rank", "nationalite", "mre", "tes", "total", "pct"]].rename(columns={
            "rank": "#", "nationalite": "Pays de résidence", "mre": "MRE",
            "tes": "TES", "total": "Total", "pct": "%"
        }))

        # Top TES only
        tes_top = nat.sort_values("tes", ascending=False).head(15).copy()
        tes_top["pct"] = (tes_top["tes"] / self.total_tes * 100).round(1)
        tes_t = df_to_md(tes_top[["nationalite", "tes", "pct"]].rename(columns={
            "nationalite": "Pays de résidence", "tes": "TES", "pct": "% des TES"
        }))

        # Top MRE
        mre_top = nat.sort_values("mre", ascending=False).head(10).copy()
        mre_top["pct"] = (mre_top["mre"] / self.total_mre * 100).round(1)
        mre_t = df_to_md(mre_top[["nationalite", "mre", "pct"]].rename(columns={
            "nationalite": "Pays de résidence", "mre": "MRE", "pct": "% des MRE"
        }))

        # Year over year for top 5
        top5_names = nat.head(5)["nationalite"].tolist()
        yoy_data = df[df["nationalite"].isin(top5_names)].groupby(
            ["year", "nationalite"]).agg(total=("mre", "sum")).reset_index()
        yoy_data["total"] = df[df["nationalite"].isin(top5_names)].groupby(
            ["year", "nationalite"]).apply(
            lambda x: x["mre"].sum() + x["tes"].sum()
        ).values
        pivot = df[df["nationalite"].isin(top5_names)].copy()
        pivot["total"] = pivot["mre"] + pivot["tes"]
        pivot_t = pivot.groupby(["year", "nationalite"])["total"].sum().unstack(fill_value=0)
        pivot_md = df_to_md(pivot_t.reset_index().rename(columns={"year": "Année"}))

        return f"""# 🌍 Analyse par Pays de résidence — Arrivées au Maroc

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Résumé
- **{df['nationalite'].nunique()} pays de résidence** représentées
- **Total** : {fmt(self.total_all)} arrivées

## Top 20 Pays de résidences (toutes catégories)

{top20_t}

## Top 15 — Touristes Étrangers (TES uniquement)

{tes_t}

## Top 10 — MRE par pays de résidence déclarée

{mre_t}

## Évolution annuelle — Top 5 pays de résidence

{pivot_md}

## Observations clés
- La France est historiquement le 1er marché source du Maroc (proximité, francophonie, diaspora)
- L'Espagne bénéficie de la proximité géographique (détroit de Gibraltar)
- Les marchés MRE sont dominés par les pays de résidence correspondant aux pays d'accueil de la diaspora (France, Espagne, Italie, Belgique, Pays-Bas, Allemagne)
- Les marchés long-courrier (USA, Chine, Japon) représentent un potentiel de croissance
"""

    # ===================================================================
    #  05 — Border Posts Reference
    # ===================================================================
    def _gen_border_posts(self) -> str:
        df = self.df

        bp = df.groupby("poste_frontiere").agg(
            mre=("mre", "sum"), tes=("tes", "sum"),
            region=("region", "first"), voie=("voie", "first")
        ).reset_index()
        bp["total"] = bp["mre"] + bp["tes"]
        bp = bp.sort_values("total", ascending=False)
        bp["pct"] = (bp["total"] / self.total_all * 100).round(1)
        bp["type"] = bp["poste_frontiere"].apply(
            lambda x: "Aéroport" if x.startswith("A ") else ("Port" if x.startswith("P ") else "Terrestre")
        )

        all_t = df_to_md(bp[["poste_frontiere", "type", "region", "voie", "mre", "tes", "total", "pct"]].rename(columns={
            "poste_frontiere": "Poste", "type": "Type", "region": "Région",
            "voie": "Voie", "mre": "MRE", "tes": "TES", "total": "Total", "pct": "%"
        }))

        by_type = bp.groupby("type").agg(
            nb_postes=("poste_frontiere", "count"), total=("total", "sum")
        ).reset_index()
        by_type["pct"] = (by_type["total"] / self.total_all * 100).round(1)
        type_t = df_to_md(by_type.rename(columns={
            "type": "Type", "nb_postes": "Nb Postes", "total": "Total Arrivées", "pct": "%"
        }))

        return f"""# 🛂 Référentiel des Postes Frontière — Maroc

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Résumé
- **{df['poste_frontiere'].nunique()} postes frontière** actifs
- Types : Aéroports (A), Ports (P), Postes terrestres (PF)

## Répartition par type

{type_t}

## Liste complète (classée par volume)

{all_t}

## Principaux aéroports
- **A Casablanca Mohammed V** : Hub principal, vols internationaux
- **A Marrakech Menara** : 2e aéroport, tourisme de loisirs
- **A Agadir Al Massira** : Tourisme balnéaire, charters européens
- **A Tanger Ibn Battouta** : Porte nord, connexion Europe

## Principaux ports
- **P Tanger Med** : Plus grand port d'Afrique, ferries depuis l'Espagne
- **P Tanger Ville** : Ferries historiques

## Principaux postes terrestres
- **PF Bab Sebta** : Frontière Ceuta (Espagne)
- **PF Bab Melilia** : Frontière Melilla (Espagne)
"""

    # ===================================================================
    #  06 — Regional Analysis
    # ===================================================================
    def _gen_regional_analysis(self) -> str:
        df = self.df

        reg = df.groupby("region").agg(
            mre=("mre", "sum"), tes=("tes", "sum"),
            nb_postes=("poste_frontiere", "nunique")
        ).reset_index()
        reg["total"] = reg["mre"] + reg["tes"]
        reg = reg.sort_values("total", ascending=False)
        reg["pct"] = (reg["total"] / self.total_all * 100).round(1)
        reg["ratio_tes"] = (reg["tes"] / reg["total"] * 100).round(1)
        reg_t = df_to_md(reg[["region", "nb_postes", "mre", "tes", "total", "pct", "ratio_tes"]].rename(columns={
            "region": "Région", "nb_postes": "Postes", "mre": "MRE",
            "tes": "TES", "total": "Total", "pct": "% National", "ratio_tes": "% TES"
        }))

        # Yearly by top 5 regions
        top5_reg = reg.head(5)["region"].tolist()
        reg_yr = df[df["region"].isin(top5_reg)].copy()
        reg_yr["total"] = reg_yr["mre"] + reg_yr["tes"]
        pivot = reg_yr.groupby(["year", "region"])["total"].sum().unstack(fill_value=0)
        pivot_t = df_to_md(pivot.reset_index().rename(columns={"year": "Année"}))

        return f"""# 🗺️ Analyse Régionale — Arrivées par Région Administrative

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Les 12 Régions du Maroc (découpage 2015)
Le Maroc est divisé en 12 régions administratives depuis 2015. Les postes frontière sont rattachés à la région où ils se trouvent géographiquement.

## Classement des régions

{reg_t}

## Évolution annuelle — Top 5 régions

{pivot_t}

## Contexte régional
- **Casablanca-Settat** : Hub économique, aéroport Mohammed V
- **Marrakech-Safi** : Capitale touristique, 1ère destination loisirs
- **Tanger-Tétouan-Al Hoceïma** : Porte nord, port Tanger Med, proximité Europe
- **Souss-Massa** : Agadir, tourisme balnéaire
- **Rabat-Salé-Kénitra** : Capitale administrative, aéroport Rabat-Salé
- **Fès-Meknès** : Tourisme culturel, médina de Fès
- **Oriental** : Frontière algérienne, aéroport Oujda, port Nador
- **L'Oriental et Drâa-Tafilalet** : Tourisme saharien, Merzouga
"""
        # ===================================================================
    #  07 — Temporal Trends
    # ===================================================================
    def _gen_temporal_trends(self) -> str:
        df = self.df

        # Monthly totals across all years
        monthly = df.groupby("year_month").agg(
            mre=("mre", "sum"), tes=("tes", "sum")
        ).reset_index()
        monthly["total"] = monthly["mre"] + monthly["tes"]
        monthly = monthly.sort_values("year_month")

        # Seasonality: average by month across years
        season = df.groupby("month").agg(
            mre=("mre", "sum"), tes=("tes", "sum")
        ).reset_index()
        season["total"] = season["mre"] + season["tes"]
        avg_monthly = season["total"].mean()
        season["indice_saisonnalite"] = (season["total"] / avg_monthly * 100).round(1)
        month_names = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }
        season["mois"] = season["month"].map(month_names)
        season_t = df_to_md(season[["mois", "mre", "tes", "total", "indice_saisonnalite"]].rename(columns={
            "mois": "Mois", "mre": "MRE", "tes": "TES",
            "total": "Total", "indice_saisonnalite": "Indice Saisonnalité"
        }))

        # Year over year growth
        yearly = df.groupby("year").agg(
            mre=("mre", "sum"), tes=("tes", "sum")
        ).reset_index()
        yearly["total"] = yearly["mre"] + yearly["tes"]
        yearly["croissance_pct"] = yearly["total"].pct_change().multiply(100).round(1)
        yearly["croissance_mre_pct"] = yearly["mre"].pct_change().multiply(100).round(1)
        yearly["croissance_tes_pct"] = yearly["tes"].pct_change().multiply(100).round(1)
        yearly_t = df_to_md(yearly.rename(columns={
            "year": "Année", "mre": "MRE", "tes": "TES", "total": "Total",
            "croissance_pct": "Croiss. %", "croissance_mre_pct": "Croiss. MRE %",
            "croissance_tes_pct": "Croiss. TES %"
        }))

        # Peak and low months
        peak_month = season.loc[season["total"].idxmax()]
        low_month = season.loc[season["total"].idxmin()]

        return f"""# 📈 Tendances Temporelles — Arrivées Touristiques au Maroc

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Croissance annuelle

{yearly_t}

## Saisonnalité mensuelle (toutes années confondues)

Un indice > 100 signifie un mois supérieur à la moyenne. Un indice < 100 signifie un mois inférieur.

{season_t}

## Points clés
- **Mois le plus fort** : {peak_month['mois']} (indice {peak_month['indice_saisonnalite']})
- **Mois le plus faible** : {low_month['mois']} (indice {low_month['indice_saisonnalite']})
- **Moyenne mensuelle** : {fmt(int(avg_monthly))} arrivées

## Patterns saisonniers typiques
- **Été (Juil-Août)** : Pic MRE — la diaspora rentre pour les vacances d'été
- **Printemps (Mars-Avril)** : Forte demande TES — climat idéal, Pâques
- **Automne (Oct-Nov)** : 2e pic TES — saison culturelle, températures agréables
- **Hiver (mois disponibles)** : Creux relatif, sauf stations de ski et Marrakech
- **Ramadan** : Impact variable selon sa position dans le calendrier (mobile)

## Facteurs d'influence
- Calendrier des vacances scolaires (France, Espagne, Allemagne)
- Événements : CAN, Coupe du Monde, festivals (Mawazine, FIFM, Gnaoua)
- Capacité aérienne (ouverture/fermeture de lignes low-cost)
- Conjoncture géopolitique régionale
- Taux de change EUR/MAD
"""

    # ===================================================================
    #  08 — Continent Analysis
    # ===================================================================
    def _gen_continent_analysis(self) -> str:
        df = self.df

        cont = df.groupby("continent").agg(
            mre=("mre", "sum"), tes=("tes", "sum"),
            nb_nat=("nationalite", "nunique")
        ).reset_index()
        cont["total"] = cont["mre"] + cont["tes"]
        cont = cont.sort_values("total", ascending=False)
        cont["pct"] = (cont["total"] / self.total_all * 100).round(1)
        cont_t = df_to_md(cont[["continent", "nb_nat", "mre", "tes", "total", "pct"]].rename(columns={
            "continent": "Continent", "nb_nat": "Pays de résidences",
            "mre": "MRE", "tes": "TES", "total": "Total", "pct": "%"
        }))

        # Top country of residence per continent
        top_per_cont = df.groupby(["continent", "nationalite"]).agg(
            total=("tes", "sum")
        ).reset_index()
        top_per_cont = top_per_cont.sort_values("total", ascending=False)
        top_per_cont = top_per_cont.groupby("continent").first().reset_index()
        top_per_cont = top_per_cont.sort_values("total", ascending=False)
        tpc_t = df_to_md(top_per_cont[["continent", "nationalite", "total"]].rename(columns={
            "continent": "Continent", "nationalite": "1ère Pays de résidence TES", "total": "TES"
        }))

        # Yearly by continent
        cont_yr = df.copy()
        cont_yr["total"] = cont_yr["mre"] + cont_yr["tes"]
        pivot = cont_yr.groupby(["year", "continent"])["total"].sum().unstack(fill_value=0)
        # Keep top 6 continents only for readability
        top6 = cont.head(6)["continent"].tolist()
        pivot = pivot[[c for c in top6 if c in pivot.columns]]
        pivot_t = df_to_md(pivot.reset_index().rename(columns={"year": "Année"}))

        return f"""# 🌐 Analyse par Continent d'Origine

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Répartition par continent

{cont_t}

## Première pays de résidence TES par continent

{tpc_t}

## Évolution annuelle par continent

{pivot_t}

## Observations
- L'Europe domine largement les arrivées (proximité, accords aériens, diaspora)
- Le continent africain représente un marché en croissance (coopération Sud-Sud)
- Les marchés des Amériques et d'Asie sont des cibles de diversification stratégique
- La catégorie "Autres" peut inclure des pays de résidence non classifiées ou rares
"""

    # ===================================================================
    #  09 — Entry Mode Analysis
    # ===================================================================
    def _gen_entry_mode(self) -> str:
        df = self.df

        voie = df.groupby("voie").agg(
            mre=("mre", "sum"), tes=("tes", "sum"),
            nb_postes=("poste_frontiere", "nunique")
        ).reset_index()
        voie["total"] = voie["mre"] + voie["tes"]
        voie = voie.sort_values("total", ascending=False)
        voie["pct"] = (voie["total"] / self.total_all * 100).round(1)
        voie["ratio_mre"] = (voie["mre"] / voie["total"] * 100).round(1)
        voie_t = df_to_md(voie[["voie", "nb_postes", "mre", "tes", "total", "pct", "ratio_mre"]].rename(columns={
            "voie": "Voie d'Entrée", "nb_postes": "Postes", "mre": "MRE",
            "tes": "TES", "total": "Total", "pct": "% Total", "ratio_mre": "% MRE"
        }))

        # Yearly by voie
        voie_yr = df.copy()
        voie_yr["total"] = voie_yr["mre"] + voie_yr["tes"]
        pivot = voie_yr.groupby(["year", "voie"])["total"].sum().unstack(fill_value=0)
        pivot_t = df_to_md(pivot.reset_index().rename(columns={"year": "Année"}))

        # Top post per voie
        bp_voie = df.groupby(["voie", "poste_frontiere"]).agg(total=("mre", "sum")).reset_index()
        bp_voie["total"] = df.groupby(["voie", "poste_frontiere"]).apply(
            lambda x: x["mre"].sum() + x["tes"].sum()
        ).values
        bp_voie = bp_voie.sort_values("total", ascending=False).groupby("voie").head(3)
        bp_voie = bp_voie.sort_values(["voie", "total"], ascending=[True, False])
        bp_t = df_to_md(bp_voie[["voie", "poste_frontiere", "total"]].rename(columns={
            "voie": "Voie", "poste_frontiere": "Poste Frontière", "total": "Total"
        }))

        return f"""# ✈️ Analyse par Mode d'Entrée (Voie)

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Trois modes d'entrée au Maroc
1. **V Aérienne** — Aéroports internationaux
2. **V Maritime** — Ports (ferries depuis l'Espagne, France, Italie)
3. **V Terrestre** — Postes frontières terrestres (Ceuta, Melilla)

## Répartition globale

{voie_t}

## Évolution annuelle par voie

{pivot_t}

## Top 3 postes par voie

{bp_t}

## Observations
- La **voie aérienne** domine pour les TES (touristes étrangers)
- La **voie maritime** a un fort ratio MRE (ferries Espagne-Maroc pour la diaspora)
- La **voie terrestre** concerne principalement les passages Ceuta/Melilla
- L'opération **Marhaba** (été) facilite le retour des MRE par voie maritime
- Le développement de Tanger Med a renforcé la voie maritime
"""

    # ===================================================================
    #  10 — Morocco Tourism Context
    # ===================================================================
    def _gen_tourism_context(self) -> str:
        return f"""# 🇲🇦 Contexte du Tourisme au Maroc

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Le Maroc en bref
- **Population** : ~37 millions d'habitants (2024)
- **Capitale** : Rabat
- **Langue officielle** : Arabe, Amazigh. Français largement utilisé
- **Monnaie** : Dirham marocain (MAD). 1 EUR ≈ 10.8 MAD
- **Superficie** : 710,850 km²
- **Diaspora (MRE)** : ~5 millions de Marocains à l'étranger

## Le tourisme dans l'économie marocaine
- **PIB touristique** : ~7% du PIB national
- **Emplois** : ~550,000 emplois directs, ~2 millions indirects
- **Recettes** : ~90 milliards MAD / an (en croissance)
- **Rang mondial** : Top 30 des destinations mondiales
- **Rang Afrique** : 1ère destination touristique du continent africain

## Stratégie nationale
### Vision 2020 (terminée)
- Objectif : 20 millions de touristes → partiellement atteint avant COVID
- 8 territoires touristiques identifiés
- Développement balnéaire (Plan Azur), culturel, rural

### Feuille de Route 2023-2026
- Objectif : **26 millions de touristes d'ici 2030**
- Augmentation capacité aérienne (Open Sky, nouvelles lignes low-cost)
- Diversification marchés (Asie, Amérique, Afrique subsaharienne)
- Tourisme durable et rural
- Digitalisation du secteur
- Formation professionnelle renforcée

## Principales destinations
| Destination | Spécialité |
|-------------|-----------|
| Marrakech | Culture, médina, riads, événementiel |
| Agadir | Balnéaire, golf, all-inclusive |
| Fès | Patrimoine UNESCO, artisanat, spiritualité |
| Casablanca | Business, hub aérien, architecture Art Déco |
| Tanger | Culture, proximité Europe, port Tanger Med |
| Essaouira | Vent, surf, musique gnaoua, charme |
| Ouarzazate | Cinéma, désert, kasbah |
| Chefchaouen | Tourisme de niche, ville bleue, photographie |
| Dakhla | Kitesurf, écotourisme, aventure |
| Merzouga | Désert, dunes, tourisme saharien |

## Typologie des touristes
1. **TES (Touristes Étrangers Séjournistes)** : Visiteurs internationaux
2. **MRE (Marocains Résidant à l'Étranger)** : Diaspora en visite
3. **Touristes internes** : Marocains résidents (non comptés dans APF)
4. **Croisiéristes** : Escales dans les ports marocains
5. **Excursionnistes** : Visiteurs d'un jour (Ceuta, Melilla)

## Principaux marchés sources (TES)
1. 🇫🇷 France — 1er marché historique
2. 🇪🇸 Espagne — Proximité, ferry, low-cost
3. 🇬🇧 Royaume-Uni — Marrakech, Agadir
4. 🇩🇪 Allemagne — Agadir, golf
5. 🇮🇹 Italie — Marrakech, Fès
6. 🇺🇸 États-Unis — Marché en forte croissance
7. 🇧🇪 Belgique — Diaspora + tourisme
8. 🇳🇱 Pays-Bas — Diaspora + tourisme

## Institutions clés
| Institution | Rôle |
|------------|------|
| Ministère du Tourisme (MTAESS) | Politique touristique, réglementation, statistiques |
| ONMT | Promotion du Maroc à l'étranger |
| SMIT | Investissements et aménagement touristique |
| Observatoire du Tourisme | Études, enquêtes, baromètres |
| ONDA | Gestion des aéroports |
| ANP | Gestion des ports |
| Bank Al-Maghrib | Statistiques recettes en devises |
| HCP | Comptes nationaux, enquêtes ménages |

## Événements majeurs impactant le tourisme
- **Opération Marhaba** : Chaque été, accueil des MRE aux ports/aéroports
- **CAN 2025** : Coupe d'Afrique des Nations au Maroc
- **Coupe du Monde 2030** : Co-organisation Maroc-Espagne-Portugal
- **Festival Mawazine** : Rabat, >2M de spectateurs
- **FIFM** : Festival International du Film de Marrakech
- **Festival Gnaoua** : Essaouira, musique du monde
"""    # ===================================================================
    #  11 — STATOUR Platform Guide
    # ===================================================================
    def _gen_statour_guide(self) -> str:
        return f"""# 💻 Guide de la Plateforme STATOUR

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Qu'est-ce que STATOUR ?
STATOUR est la plateforme centrale de gestion des statistiques touristiques du Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire (MTAESS) du Maroc.

## Mission
Collecter, valider, intégrer, analyser et diffuser les données statistiques touristiques pour :
- L'administration centrale (Rabat)
- Les 12 Directions Régionales du Tourisme (DRT)
- Les 15+ Délégations Provinciales

## Architecture actuelle (en modernisation)
- **Collecte** : Power Apps + formulaires manuels
- **Stockage** : Microsoft Dataverse
- **Analyse** : Power BI
- **Sécurité** : Row-Level Security (RLS)

## Architecture cible (modernisation PaaS)

| Couche | Actuel | Cible |
|--------|--------|-------|
| Collecte | Power Apps, saisie manuelle | Airbyte, APIs automatisées, OCR |
| Stockage | Dataverse | Lakehouse → Data Warehouse (PostgreSQL) |
| Analyse | Power BI basique | ML, Pandas, Scikit-learn, Plotly |
| Diffusion | Power BI Desktop | Power BI Service + Streamlit + Chatbot IA |
| Sécurité | RLS basique | Azure AD + RLS avancé + journalisation |

## Sources de données (15+)
| # | Source | Type | Contenu |
|---|--------|------|---------|
| 1 | APF (DGSN) | Excel/API | Arrivées aux postes frontière |
| 2 | Hébergement classé | Excel | Nuitées, TO, capacité |
| 3 | Office des Changes | PDF/Excel | Recettes touristiques |
| 4 | ONDA | API | Trafic aéroportuaire |
| 5 | ANP | Excel | Trafic portuaire |
| 6 | Enquêtes de conjoncture | SPSS | Satisfaction, dépenses |
| 7 | Réseaux sociaux | API | Sentiment, tendances |
| 8 | HCP | PDF | Comptes satellites |
| 9 | Bank Al-Maghrib | PDF/API | Indicateurs monétaires |
| 10 | Booking/TripAdvisor | Scraping | Avis, tarifs |
| 11 | Google Trends | API | Intérêt de recherche |
| 12 | IATA/ICAO | API | Données aviation |
| 13 | Météo Maroc | API | Données climatiques |
| 14 | Événementiel | Manuel | Calendrier événements |
| 15 | Investissements SMIT | SQL Server | Projets touristiques |

## Fonctionnalités du chatbot multi-agent
| Agent | Fonction |
|-------|----------|
| 🏛️ Assistant Général | Conversation, orientation, aide |
| 🔍 Chercheur Tourisme | Recherche web + base de connaissances |
| 📊 Analyste de Données | Statistiques, graphiques, tendances |
| 🗄️ Requêteur Base de Données | Questions → requêtes SQL |
| 🛂 Analyste Flux Frontières | Analyse spécialisée APF |

## Sécurité et accès
- Authentification Azure AD
- Row-Level Security (RLS) : chaque DRT ne voit que ses données régionales
- Rôles : Admin National, Admin Régional, Analyste, Consultation
- Chiffrement des données en transit et au repos
- Journalisation des accès et requêtes
"""

    # ===================================================================
    #  12 — MRE Analysis
    # ===================================================================
    def _gen_mre_analysis(self) -> str:
        df = self.df

        # MRE by year
        mre_yr = df.groupby("year")["mre"].sum().reset_index()
        mre_yr["croissance_pct"] = mre_yr["mre"].pct_change().multiply(100).round(1)
        mre_yr_t = df_to_md(mre_yr.rename(columns={
            "year": "Année", "mre": "Total MRE", "croissance_pct": "Croissance %"
        }))

        # MRE by month (seasonality)
        mre_month = df.groupby("month")["mre"].sum().reset_index()
        avg_m = mre_month["mre"].mean()
        mre_month["indice"] = (mre_month["mre"] / avg_m * 100).round(1)
        month_names = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }
        mre_month["mois"] = mre_month["month"].map(month_names)
        mre_month_t = df_to_md(mre_month[["mois", "mre", "indice"]].rename(columns={
            "mois": "Mois", "mre": "MRE", "indice": "Indice Saisonnalité"
        }))

        # MRE by voie
        mre_voie = df.groupby("voie")["mre"].sum().reset_index()
        mre_voie = mre_voie.sort_values("mre", ascending=False)
        mre_voie["pct"] = (mre_voie["mre"] / self.total_mre * 100).round(1)
        mre_voie_t = df_to_md(mre_voie.rename(columns={
            "voie": "Voie", "mre": "MRE", "pct": "%"
        }))

        # MRE by top border posts
        mre_bp = df.groupby("poste_frontiere")["mre"].sum().reset_index()
        mre_bp = mre_bp.sort_values("mre", ascending=False).head(10)
        mre_bp["pct"] = (mre_bp["mre"] / self.total_mre * 100).round(1)
        mre_bp_t = df_to_md(mre_bp.rename(columns={
            "poste_frontiere": "Poste Frontière", "mre": "MRE", "pct": "%"
        }))

        # MRE by country of residence
        mre_nat = df.groupby("nationalite")["mre"].sum().reset_index()
        mre_nat = mre_nat.sort_values("mre", ascending=False).head(15)
        mre_nat["pct"] = (mre_nat["mre"] / self.total_mre * 100).round(1)
        mre_nat_t = df_to_md(mre_nat.rename(columns={
            "nationalite": "Pays de résidence", "mre": "MRE", "pct": "%"
        }))

        peak = mre_month.loc[mre_month["mre"].idxmax()]

        return f"""# 🇲🇦 Analyse des MRE — Marocains Résidant à l'Étranger

> Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Contexte
Les **MRE (Marocains Résidant à l'Étranger)** constituent une composante majeure des arrivées au Maroc :
- **Diaspora estimée** : ~5 millions de personnes
- **Principaux pays d'accueil** : France, Espagne, Italie, Belgique, Pays-Bas, Allemagne, Canada, USA
- **Impact économique** : Transferts ~100 milliards MAD/an, investissements immobiliers, consommation locale

## Total MRE : {fmt(self.total_mre)}
- **Part dans le total des arrivées** : {self.total_mre/self.total_all*100:.1f}%

## Évolution annuelle

{mre_yr_t}

## Saisonnalité mensuelle

{mre_month_t}

- **Pic principal** : {peak['mois']} (indice {peak['indice']}) — vacances d'été, Opération Marhaba

## Répartition par voie d'entrée

{mre_voie_t}

## Top 10 postes frontière pour les MRE

{mre_bp_t}

## Top 15 pays de résidence déclarées par les MRE

{mre_nat_t}

## L'Opération Marhaba
Chaque année de juin à septembre, le Maroc lance l'**Opération Marhaba** pour accueillir les MRE :
- Accueil renforcé dans les ports (Tanger Med, Nador, Al Hoceïma)
- Permanences consulaires aux points de passage
- Assistance routière, sanitaire et sociale
- En 2023 : ~3 millions de MRE accueillis pendant l'opération
- Coordination : Fondation Mohammed V pour la Solidarité

## Enjeux stratégiques
1. **Fidélisation** : Maintenir le lien diaspora-Maroc
2. **Investissement** : Encourager les investissements des MRE
3. **Tourisme affinitaire** : Les MRE reviennent en famille (dépenses locales élevées)
4. **Saisonnalité** : Forte concentration en été → besoin de désaisonnaliser
5. **Nouvelles générations** : Les jeunes MRE nés à l'étranger — garder l'attachement
"""


# ===========================================================================
# Entry Point
# ===========================================================================
def main():
    generator = KnowledgeGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
