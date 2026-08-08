DRIVE LINK - https://drive.google.com/drive/folders/1RY8CZtioErqBfjO0E2D3Bsib82kXF_FS



---Road Traffic Accident Severity Analysis

Exploratory data analysis and machine learning project on UK road traffic accident data, predicting **Accident Severity** (Slight / Serious / Fatal) from road, environmental, and vehicle-related factors.

Business Use Case

Road traffic accidents impose heavy costs on public health systems, insurers, and city infrastructure budgets. Being able to predict and understand what drives accident **severity** (not just occurrence) helps stakeholders act *before* incidents escalate, rather than just reporting on them after the fact.

Who this helps and how:

- Transport / Highway Authorities — Identify high-risk road types, junctions, and speed limits so infrastructure fixes (signage, lighting, road surface repair, junction redesign) can be prioritized where they reduce fatal/serious outcomes the most, instead of spreading budget evenly.
- Traffic Police & Emergency Services — Use conditions most linked to severe outcomes (e.g. speed limit, lighting, police response) to plan patrol allocation, enforcement zones, and ambulance/response positioning during high-risk time windows or weather conditions.
- Insurance Companies — Feed severity drivers into underwriting and risk-pricing models (e.g. adjusting premiums by region, road type, or typical driving conditions) and into fraud/claims triage by flagging claims inconsistent with a low-severity accident profile.
- Urban Planners / Smart City Programs — Use geographic hotspots (latitude/longitude importance) to justify infrastructure investment, speed-limit changes, or new traffic-calming measures in specific high-severity zones.
- Road Safety Policy Makers — Support evidence-based policy (e.g. stricter speed enforcement, mandatory lighting standards) using data-driven feature importance rather than anecdotal reasoning.

Why severity prediction specifically (not just accident count):
Not all accidents are equal — a Fatal accident and a Slight accident consume very different resources (hospital beds, legal processes, insurance payouts, media/political attention). A severity classifier lets stakeholders **triage and prioritize response and prevention efforts by expected impact**, which is more actionable than simply counting accidents.

Model choice implication: Random Forest's strong ROC-AUC (0.984) means it reliably ranks accidents by risk of severe outcomes — useful for building a **risk-scoring dashboard** that flags high-risk road segments or conditions in near real time, rather than only a static historical report.

Dataset

- Source: Road Accident data (`Accident_Information.csv`)
- Target variable: `Acident_Severity` — Slight (254,683), Serious (38,089), Fatal (4,228)
- Features: Road type, light conditions, weather, junction detail, speed limit, urban/rural area, number of vehicles/casualties, and more

Workflow

1. Data Loading & Exploration
   - Loaded and inspected shape, dtypes, and cardinality of categorical columns
   - Analyzed distribution of severity, road type, light and weather conditions
2. Data Cleaning
   - Handled missing values (mode imputation for categorical, mean for numerical)
   - Grouped rare weather categories into an "Other" bucket for cleaner visualization
3. Feature Analysis
   - Label-encoded categorical variables
   - Ranked feature relevance using correlation and **mutual information** scores
4. Handling Class Imbalance
   - Applied **SMOTE** to balance the three severity classes before training
5. Modeling
   - Trained and compared four classifiers:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
6. Evaluation
   - Accuracy, classification report, and confusion matrix per model
   - ROC-AUC (weighted, one-vs-rest) for multi-class comparison
   - Feature importance ranking from the best-performing model

Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.373 | 0.558 |
| Decision Tree | 0.856 | 0.892 |
| Random Forest | **0.933** | **0.984** |
| XG-Boost | ~0.90 | 0.931 |

Random Forest was the best-performing model on both accuracy and ROC-AUC.

Top predictive features included **speed limit**, **police attendance at scene**, **location coordinates (latitude/longitude)**, and **light conditions**.

Tech Stack

- **Python**, **Pandas**, **NumPy** — data wrangling
- **Matplotlib**, **Seaborn** — visualization
- **Scikit-learn** — preprocessing, modeling, evaluation
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **XG-Boost** — gradient boosting classifier

How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
3. Update the dataset path in the notebook to point to your local copy of `Accident_Information.csv`
4. Run `transport_safety_domain.ipynb` cell by cell (originally built in Google Colab)

Project Structure

```
├── transport_safety_domain.ipynb   
└── README.md
```
