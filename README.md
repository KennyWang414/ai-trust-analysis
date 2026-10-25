# Exploring Trust in Artificial Intelligence

A multilevel analysis of what shapes people's trust in AI, and what happens to that trust when AI makes mistakes, using three datasets at the country, interaction and user level.

*AN6002 Analytics & Machine Learning in Business · MSc Business Analytics, Nanyang Technological University (2026)*

## Key results

- **Individual level (multinomial logit, pseudo R² 0.48, 1,000 AI interactions):** high-stakes decisions raised the odds of strong skepticism (vs. blind trust) **107×**, while cited sources lowered them **~36×**. AI familiarity, digital literacy and education all increased skepticism.
- **Country level (47 countries, 9 regression models compared):** a tuned KNN regressor (k = 3) was used to predict national AI trust from development indicators; AI trust was associated with age structure, mean years of schooling, government effectiveness and internet use.
- **After AI errors (301 AI-use sessions):** small trust drops tended to recover fully while large ones did not; K-means clustering identified three user types, with "Uncritical Enthusiasts" losing the most trust.

## Repository layout

| Folder | Contents |
|---|---|
| `docs/` | Final report, poster and presentation |
| `section2_country/` | Country-level ML models (Python, scikit-learn) and data |
| `section3_individual/` | Individual-level multinomial logit analysis (notebook) and data |
| `section4_trust_drop/` | Trust-drop analysis and clustering (Python) and data |

Each script reads its data file from its own folder, so run it from inside that folder.

**Not included:** `WB_WDI_WIDEF.csv` (World Bank World Development Indicators, 256 MB, above GitHub's file limit). Download it from the [World Bank DataBank](https://databank.worldbank.org/source/world-development-indicators) and place it in `section2_country/`. The Section 4 script expects the dataset to be named `dataset2.csv`; rename `section4_dataset.csv` before running.

## Team

| Section | Member |
|---|---|
| 1, 5, 6 — Introduction, solutions, conclusion | Wang Gaoyang (Kenny) |
| 2 — AI trust by country | Kavin Lertsattha |
| 3 — AI trust by individual | Triston Cheong |
| 4 — Trust after AI errors | Timothy Tan |

**Data sources:** [Global AI Trust and Attitudes by Country 2025](https://www.kaggle.com/datasets/kylefengkfeng209/global-ai-trust-and-attitudes-by-country-2025), [AI Trust Insights](https://www.kaggle.com/datasets/emanfatima2/ai-trust-insights), [Human Trust Decay in AI](https://www.kaggle.com/datasets/landgesakshi/human-trust-decay-in-ai), [UNDP Human Development Index](https://hdr.undp.org/data-center/human-development-index), [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators).
