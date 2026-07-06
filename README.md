# 🚢 Titanic Survival Prediction

Predicting Titanic passenger survival based on their characteristics (age, sex, class, etc.) using a binary classification model.

## 📋 Context

Machine learning learning project covering the full pipeline: data exploration, cleaning, feature engineering, model training and evaluation.

## 📊 Dataset

- **Source**: Titanic dataset (via `seaborn.load_dataset('titanic')`)
- **891 observations**, 15 raw columns
- **Target**: `survived` (0 = deceased, 1 = survived) — binary classification
- Target distribution: 62% deceased / 38% survived

## 🔍 Exploratory Data Analysis (EDA)

- Two variables strongly correlated with survival:
  - **Sex**: ~74% survival rate for women vs ~19% for men
  - **Class**: 1st class passengers survived at a much higher rate than 3rd class
- Missing values identified: `deck` (77%), `age` (20%), `embarked` (0.2%)

## 🧹 Data Cleaning & Feature Engineering

- Dropped unusable/redundant columns (`deck`, `embark_town`, `class`, `alive`, `who`, `adult_male`, `alone`)
- Imputed `age` (median) and `embarked` (mode)
- Encoding: `sex` converted to binary, `embarked` one-hot encoded
- Created a new feature, `family_size` (`sibsp` + `parch` + 1)

## 🤖 Modeling

- **Split**: 80% train / 20% test (`random_state=42`)
- **Model**: Logistic Regression (`scikit-learn`)

### Results

| Metric | Class 0 (deceased) | Class 1 (survived) |
|---|---|---|
| Precision | 0.83 | 0.79 |
| Recall | 0.86 | 0.74 |
| F1-score | 0.84 | 0.76 |

**Overall accuracy: 81%**

Confusion matrix:
[[90 15]
[19 55]]

## 🛠️ Tech Stack

- Python 3
- pandas, seaborn, matplotlib
- scikit-learn

## 🚀 How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn scikit-learn

python exploration.py    # EDA
python nettoyage.py       # cleaning + feature engineering
python modele.py          # training + evaluation
```

## 📁 Project Structure

itanic_project/
├── exploration.py       # Data loading and exploration
├── nettoyage.py          # Cleaning and feature engineering
├── modele.py             # Model training and evaluation
├── titanic.csv            # Raw data
├── titanic_clean.csv      # Cleaned data
└── README.md



## 🔮 Next Steps

- [ ] Compare with a Random Forest model
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning (GridSearch)
- [ ] Model deployment via an API (Django REST Framework)

## 👤 Author

Mohamed WARMA — [LinkedIn](https://www.linkedin.com/in/mohamed-warma) | mohamed.warma10@gmail.com


