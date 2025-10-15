# Paris Air Quality Forecast

Projet de prévision horaire de CO, NO2, O3, PM10, PM2.5 pour 3 semaines.

## Structure
- data/ : données brutes et externes (non versionnées)
- notebooks/ : EDA et prototypage
- src/ : code de préparation et modèles
- models/ : artefacts de modèles
- submissions/ : fichiers de soumission

## Installation rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluation
MAE moyenne sur 5 polluants.

## Données externes
- MeteoNet (Météo-France, 2016–2018) — licence: Etalab Open Licence 2.0.
  - Citer: Larvor G., Berthomier L., Chabot V., Le Pape B., Pradel B., Perez L. (2020). "MeteoNet, an open reference weather dataset by METEO FRANCE".
  - Repo: https://github.com/meteofrance/meteonet — Kaggle: https://www.kaggle.com/katerpillar/meteonet
  - Usage: pour créer des climatologies/profils horaires (pas de données futures sur la période test), respecter les règles du challenge.
