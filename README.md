# 🔍 Kernel SVM — Model Evaluation & Hyperparameter Tuning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> Predict whether a social network user will purchase a product using a **Kernel SVM classifier**, with robust model evaluation via **k-Fold Cross Validation** and optimal hyperparameter selection via **Grid Search**.

---

## 📁 Project Structure

```
├── Social_Network_Ads.csv           # Dataset
├── k_fold_cross_validation.ipynb    # Notebook: Kernel SVM + k-Fold CV
├── grid_search.ipynb                # Notebook: Grid Search + best params
└── README.md
```

---

## 📊 Dataset

**File:** `Social_Network_Ads.csv`  
**Rows:** 400 | **Features:** 2 | **Target:** Binary (Purchased: 0 / 1)

| Feature | Type | Description |
|---|---|---|
| `Age` | int | Age of the user |
| `EstimatedSalary` | int | Estimated annual salary |
| `Purchased` | int (0/1) | Target — whether the user purchased |

---

## 🧠 Methodology

### 1. k-Fold Cross Validation (`k_fold_cross_validation.ipynb`)

Evaluates the Kernel SVM model's generalization performance using **10-fold cross-validation** on the training set to get a reliable accuracy estimate and standard deviation.

**Pipeline:**
- Load data → Train/Test split (75/25) → Feature Scaling (StandardScaler)
- Train `SVC(kernel='rbf')`
- Evaluate on test set → Confusion Matrix + Accuracy Score
- Apply `cross_val_score` with `cv=10` on training data
- Report mean accuracy and standard deviation

### 2. Grid Search (`grid_search.ipynb`)

Extends the above pipeline with **GridSearchCV** to find the optimal combination of `C`, `kernel`, and `gamma` hyperparameters.

**Search Space:**

| Param | Values Tried |
|---|---|
| `C` | 0.25, 0.5, 0.75, 1 |
| `kernel` | `linear`, `rbf` |
| `gamma` (rbf only) | 0.1 → 0.9 (step 0.1) |

- `cv=10`, `scoring='accuracy'`, `n_jobs=-1`
- Reports best accuracy and best parameter combination

---

## 🛠️ Tech Stack

- **Python 3.8+**
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn` — SVC, StandardScaler, train_test_split, cross_val_score, GridSearchCV

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/kernel-svm-gridsearch.git
cd kernel-svm-gridsearch

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Launch notebooks
jupyter notebook
```

Open `k_fold_cross_validation.ipynb` first, then `grid_search.ipynb`.

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Set Accuracy | ~90% |
| k-Fold CV Mean Accuracy | ~90.33% |
| k-Fold CV Std Deviation | ~6.57% |
| Grid Search Best Accuracy | ~90.33% |
| Best Parameters | `C=0.5`, `kernel='rbf'`, `gamma=0.6` *(example)* |

> ⚠️ *Exact results may vary slightly based on environment. Run the notebooks to reproduce.*

---

## 📌 Key Concepts

- **Kernel SVM (RBF):** Maps data into higher-dimensional space for non-linear classification
- **k-Fold Cross Validation:** Splits training data into k subsets to get a stable accuracy estimate and avoid overfitting to a single split
- **Grid Search:** Exhaustively searches over a defined hyperparameter grid to find the combination that maximizes CV accuracy

---

## 👤 Author

**Samsur**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)
