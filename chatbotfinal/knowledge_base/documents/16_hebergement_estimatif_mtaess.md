# Hébergement Touristique Estimé — MTAESS

> Base de connaissances interne STATOUR

## Périmètre disponible

Les analyses d'hébergement accessibles au chatbot doivent utiliser uniquement la table Fabric
`fact_statistiqueshebergementnationaliteestimees` et les dimensions Gold exposées dans `.env`.

La table décrit les établissements d'hébergement touristique classés avec une granularité mensuelle.
Les métriques principales sont :

| Métrique | Définition | Formule |
|----------|------------|---------|
| Nuitées | Nuits passées en hébergement | somme de `nuitees` |
| Arrivées hôtelières | Check-ins dans les établissements | somme de `arrivees` |
| DMS | Durée moyenne de séjour | `SUM(nuitees) / SUM(arrivees)` |

## Règles d'interprétation

- Les arrivées hôtelières ne sont pas les arrivées APF aux frontières.
- `nationalite_name` signifie pays de résidence du touriste.
- `province_name` représente la province ou destination utilisée pour filtrer Casablanca, Marrakech, Agadir, Tanger, etc.
- `region_name` représente la région administrative.
- `date_stat` est une période mensuelle: filtrer avec `YEAR(date_stat)` et `MONTH(date_stat)`.

## Jointures utiles

| Besoin | Jointure |
|--------|----------|
| Type d'hébergement | `gld_dim_categories_classements` sur `categorie_name`, puis `type_eht_libelle` |
| Nom et capacité d'établissement | `gld_dim_etablissements_hebergements` sur `CAST(etablissement_id_genere AS VARCHAR) = eht_id` |
| Délégation | `gld_dim_delegations` sur `delegation_bk = delegation_id` |

## Catégories d'hébergement

- Hôtels classés
- Maisons d'hôtes
- Campings
- Résidences de tourisme
- Auberges de jeunesse
- Autres catégories exposées par `gld_dim_categories_classements`
