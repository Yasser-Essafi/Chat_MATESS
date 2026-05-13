"""
Fabric business catalog helpers for STATOUR.

This module keeps the table semantics that are not obvious from
INFORMATION_SCHEMA: metric definitions, output labels, joins and source
labels that are safe to show to users.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional


APF_TABLE = "fact_statistiques_apf"
HEBERGEMENT_TABLE = "fact_statistiqueshebergementnationaliteestimees"
CATEGORIES_TABLE = "gld_dim_categories_classements"
ETABLISSEMENTS_TABLE = "gld_dim_etablissements_hebergements"
DELEGATIONS_TABLE = "gld_dim_delegations"

APF_SCOPE_LABEL = "APF/DGSN, arrivees aux postes frontieres"
HEBERGEMENT_SCOPE_LABEL = "donnees d'hebergement estimees, etablissements classes"

ALLOWED_TABLES = {
    APF_TABLE,
    HEBERGEMENT_TABLE,
    CATEGORIES_TABLE,
    ETABLISSEMENTS_TABLE,
    DELEGATIONS_TABLE,
}


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def month_range_label(months: Iterable[int]) -> str:
    names = {
        1: "Janvier", 2: "Fevrier", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Aout",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Decembre",
    }
    clean = sorted({int(m) for m in months if m})
    if not clean:
        return "periode disponible"
    if clean == list(range(clean[0], clean[-1] + 1)) and clean[0] == 1:
        return names.get(clean[-1], str(clean[-1])) if clean[-1] == 1 else f"Janvier-{names.get(clean[-1], clean[-1])}"
    return ", ".join(names.get(m, str(m)) for m in clean)


def is_apf_context(text: str) -> bool:
    norm = norm_text(text)
    return bool(
        re.search(r"\b(apf|mre|tes|dgsn)\b", norm)
        or any(k in norm for k in ["frontiere", "poste", "voie", "aeroport", "maritime", "terrestre"])
    )


def is_hebergement_context(text: str) -> bool:
    norm = norm_text(text)
    return any(
        k in norm
        for k in [
            "hebergement", "hotel", "hoteliere", "hotelieres", "nuitee",
            "nuitees", "dms", "ehtc", "etablissement", "categorie",
            "maison d'hote", "riad", "camping", "delegation", "chambre",
        ]
    )


def business_catalog_text(datasets: Dict[str, Dict], schema: str = "dbo_GOLD") -> str:
    """Render concise business semantics for the currently accessible tables."""
    present = {name for name in datasets if name in ALLOWED_TABLES}
    lines: List[str] = [
        "CATALOGUE METIER FABRIC AUTORISE:",
        f"- Schema actif: [{schema}]",
        f"- Tables exposees: {', '.join(sorted(present)) if present else 'aucune'}",
        "",
    ]

    if APF_TABLE in present:
        lines.extend([
            f"- [{schema}].[{APF_TABLE}] = arrivees aux postes frontieres.",
            "  Colonnes cles: date_stat, nationalite, poste_frontiere, region, continent, voie, mre, tes.",
            "  Metriques: MRE = Marocains residant a l'etranger; TES = touristes etrangers sejournistes; arrivees APF = mre + tes.",
            "  Piege: nationalite = pays de residence, jamais nationalite ethnique.",
        ])
    if HEBERGEMENT_TABLE in present:
        lines.extend([
            f"- [{schema}].[{HEBERGEMENT_TABLE}] = donnees d'hebergement estimees.",
            "  Colonnes cles: eht_id, nationalite_name, categorie_name, province_name, region_name, date_stat, arrivees, nuitees.",
            "  Metriques: arrivees = check-ins hoteliers; nuitees = nuits passees; DMS = nuitees / arrivees.",
            "  Piege: arrivees hotelières != arrivees APF; nationalite_name = pays de residence.",
        ])
    if CATEGORIES_TABLE in present:
        lines.append(f"- [{schema}].[{CATEGORIES_TABLE}] jointure categorie_name -> type_eht_libelle/type_hebergement.")
    if ETABLISSEMENTS_TABLE in present:
        lines.append(f"- [{schema}].[{ETABLISSEMENTS_TABLE}] jointure CAST(etablissement_id_genere AS VARCHAR) = eht_id.")
    if DELEGATIONS_TABLE in present:
        lines.append(f"- [{schema}].[{DELEGATIONS_TABLE}] jointure delegation_bk = delegation_id -> delegation_name.")

    lines.extend([
        "",
        "REGLES D'INTERPRETATION:",
        "- Si l'utilisateur dit seulement arrivees, fournir APF et hebergement sauf contexte clair.",
        "- Toujours afficher nationalite/nationalite_name comme Pays de residence.",
        "- date_stat est une date mensuelle: filtrer avec YEAR(date_stat) et MONTH(date_stat), et ajouter date_stat IS NOT NULL pour l'hebergement.",
        "- Ne jamais inventer de tables non exposees dans FABRIC_TABLES.",
    ])
    return "\n".join(lines)
