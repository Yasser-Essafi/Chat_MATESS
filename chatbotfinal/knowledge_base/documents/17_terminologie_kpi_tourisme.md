# Terminologie et KPIs — Tourisme Maroc MTAESS

> Dictionnaire de référence pour l'interprétation des données STATOUR

## Abréviations et acronymes

| Sigle | Signification | Note |
|-------|--------------|------|
| APF | Arrivées aux Postes Frontières | Données DGSN mensuelles |
| MRE | Marocains Résidant à l'Étranger | ~5M personnes en diaspora |
| TES | Touristes Étrangers Séjournistes | Non-marocains entrant au Maroc |
| EHTC | Établissements Hébergement Touristique Classés | ~5000 établissements |
| TO | Taux d'Occupation | Métrique clé hébergement |
| DMS | Durée Moyenne de Séjour | nuitées/arrivées |
| DGSN | Direction Générale Sûreté Nationale | Fournisseur données APF |
| ONMT | Office National Marocain du Tourisme | Promotion touristique |
| MTAESS | Ministère Tourisme Artisanat Économie Sociale et Solidaire | Le client |
| ONDA | Office National Des Aéroports | Données trafic aérien |
| DRT | Direction Régionale du Tourisme | 12 régions |
| DPT | Délégation Provinciale du Tourisme | Saisie données STATOUR |

## Pays de résidence — clarification

Le champ `nationalite` dans les tables SQL MTAESS désigne le **pays de résidence** du
voyageur (où il vit), PAS sa nationalité ethnique ou son passeport.

Exemple : Un Marocain vivant en France est comptabilisé comme MRE "France".

**Top marchés émetteurs (pays de résidence)** :
1. France (30%)
2. Espagne (22%)
3. Belgique (6.5%)
4. Royaume-Uni (6.3%)
5. Italie (5.8%)
6. Hollande/Pays-Bas (5%)
7. Allemagne (4.8%)
8. États-Unis (2.7%)

## Voies d'entrée

| Voie | Signification | % des arrivées |
|------|--------------|---------------|
| V Aérienne | Via aéroports internationaux | ~69% |
| V Terrestre | Via postes terrestres (Ceuta, Melilla) | ~18% |
| V Maritime | Via ports (ferries Espagne-Maroc) | ~14% |

La voie maritime pic : juillet-août (Opération Marhaba MRE).

## Saisonnalité typique

- **Pic estival (juil-août)** : MRE retour diaspora (jusqu'à 67% MRE en juillet)
- **Printemps (mars-avril)** : Forte demande TES, climat idéal
- **Automne (oct-nov)** : 2e pic TES
- **Hiver (jan-mars)** : Creux relatif sauf désert et Marrakech

## Objectifs Vision 2030

| Indicateur | Objectif 2030 |
|-----------|--------------|
| Arrivées TES | 26 millions |
| Recettes touristiques | 120 Mrd MAD |
| Emplois directs | +600 000 |
| Part tourisme dans PIB | 12% |
