# 📖 Dictionnaire de Données — Statistiques APF

> Généré le 2026-05-05 21:34

## Source
- **Fichier** : `fact_statistiques_apf.xlsx`
- **Organisme** : Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire (MTAESS)
- **Fournisseur** : Administration des Postes Frontières (APF) / DGSN
- **Période** : 2019-01-01 → 2026-02-05
- **Volume** : 101,519 enregistrements
- **Années disponibles** : 2019, 2023, 2024, 2025, 2026

## Schéma

| # | Colonne | Type | Description |
|---|---------|------|-------------|
| 1 | statistiques_apf_id | UUID | Identifiant unique |
| 2 | nationalite | String | Pays de résidence du voyageur (220 valeurs) |
| 3 | poste_frontiere | String | Point d'entrée (45 postes) |
| 4 | region | String | Région administrative marocaine (12 régions) |
| 5 | continent | String | Continent d'origine (14 valeurs) |
| 6 | voie | String | Mode d'entrée (3 voies) |
| 7 | date_stat | Date | Date statistique (1er du mois) |
| 8 | mre | Integer | Marocains Résidant à l'Étranger |
| 9 | tes | Float | Touristes Étrangers Séjournistes |

## Totaux globaux (toutes années)

| Indicateur | Valeur |
|-----------|--------|
| Total arrivées | 67,339,877 |
| Total MRE | 33,268,843 (49.4%) |
| Total TES | 34,071,034 (50.6%) |

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
Beni Mellal-Khénifra, Casablanca-Settat, Dakhla-Oued Ed Dahab, Drâa-Tafilalt, Fès-Meknès, Guelmim-Oued Noun, Laâyoune-Sakia El Hamra, Marrakech-Safi, Oriental, Rabat-Salé-Kénitra, Souss-Massa, Tanger-Tétouan-Al Hoceima

## Voies (3)
V Aérienne, V Maritime, V Terrestre
