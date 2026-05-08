# Hébergement Touristique et Estimatif — MTAESS

> Base de connaissances interne STATOUR

## Sources de données hébergement

Le Ministère du Tourisme collecte les données d'hébergement via 2 systèmes parallèles :

**STDN (Système de Télé-déclaration des Nuitées)**
- Source principale si taux de déclaration ≥ seuil fixé par la Direction Statistiques
- Déclarations quotidiennes électroniques par les établissements
- Couvre les EHTC (Établissements d'Hébergement Touristique Classés)

**STATOUR**
- Plateforme de saisie manuelle par les délégations régionales/provinciales
- Fréquence mensuelle
- Source complémentaire/backup quand STDN insuffisant
- Workflow : Agent saisie → Délégué → Validation Direction Statistiques

**EHTC** : ~5000 établissements classés (hôtels 1 à 5 étoiles, maisons d'hôtes, campings,
résidences, auberges)

## Métriques clés hébergement

| Métrique | Définition | Formule |
|----------|-----------|---------|
| Nuitées | Nuits passées en hébergement | 1 touriste × 1 nuit = 1 nuitée |
| Arrivées (EHTC) | Check-ins dans les établissements | ≠ Arrivées APF |
| TO (Taux d'Occupation) | % chambres occupées | chambres occupées / chambres disponibles × 100 |
| Indice de fréquentation | Personnes par chambre | occupants / clé (chambre/appartement) |
| DMS (Durée Moyenne de Séjour) | Nuits moyennes par arrivée | nuitées / arrivées |
| Capacité | Chambres disponibles | par établissement, historisé |

## Méthodologie de l'Estimatif

L'estimatif compense les non-déclarations (~60% des EHTC ne déclarent pas régulièrement).

**Étape 1 — Sélection de l'échantillon**
- Un établissement est retenu si son TO déclaré est dans l'intervalle valide de sa province
- Codification provinces : codes 1, 2, 3 = provinces standard | code 4 = stations balnéaires
- Intervalles TO définis différemment selon le code province

**Étape 2 — Calcul de la moyenne provinciale**
- Moyenne TO des établissements de l'échantillon par province

**Étape 3 — TO estimé par établissement**
- TO estimé = Moyenne provinciale × Chambres disponibles × Jours du mois

**Étape 4 — Nuitées estimées**
- Nuitées estimées = Chambres occupées estimées × Indice de fréquentation

**Étape 5 — Arrivées estimées**
- Arrivées estimées = Nuitées estimées / DMS

**Étape 6 — Répartition par nationalité**
- Pondération croisée : données APF (postes frontières) + ONDA (aéroports)
- Permet de calculer le poids de chaque pays de résidence dans les arrivées par établissement

*La Division Statistiques peut ajuster les paramètres de calcul via une interface dédiée.*

## Confrontation STATOUR / STDN

Le matching des établissements entre les deux référentiels atteint ~89% (3763/4209 EHTC).

**Pipeline de matching (4 niveaux)** :
1. **N1 — Exact** : correspondance parfaite nom + province
2. **N2 — Google Places API** : matching par `place_id` géographique
3. **N3 — GPT IA** : analyse sémantique de similarité
4. **N4 — Jaccard** : similarité de chaînes de caractères

**Résultats de la confrontation** :
- Établissements dans STATOUR mais pas STDN → potentiel non-déclarant STDN
- Établissements dans STDN mais pas STATOUR → non classé ou nouveau

La table Gold `dim_mapping_eht` stocke le résultat avec 38 colonnes incluant enrichissement
géographique, ratings Google Places, et indicateurs de tableau de bord.

## Exclusions

**Bivouacs** : exclus de l'estimatif (nature et capacité incompatibles avec le modèle).

## Catégories d'hébergement (type_eht_libelle)

- Hôtels classés (1 à 5 étoiles)
- Maisons d'hôtes
- Campings
- Résidences de tourisme
- Auberges de jeunesse
- Bivouacs (exclus de l'estimatif)
