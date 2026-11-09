# Predictive Analytics

Two Python projects from my Bachelor of Business Analytics (course 3030AFE, Predictive Analytics), covering data wrangling, clustering, regression, tree-based models, classification and model interpretation. Each PDF is an export of my Jupyter notebook and shows the original questions followed by my code, outputs and written interpretation.

| Project | Techniques | File |
|---|---|---|
| 1. Firm risk profiles and global health data | Pandas wrangling, K-Means and hierarchical clustering, correlation analysis, data visualisation | [Assignment 1 - Questions and Answers.pdf](./Assignment%201%20-%20Questions%20and%20Answers.pdf) |
| 2. Retail customer preference modelling | Linear regression, Lasso, random forest, gradient boosting, logistic regression, decision trees, ROC curves, SHAP | [Assignment 2 - Questions and Answers.pdf](./Assignment%202%20-%20Questions%20and%20Answers.pdf) |

**Tools:** Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, statsmodels, SHAP, Jupyter Notebook.

---

## 1. Firm Risk Profiles and Global Health Data

### Data wrangling and exploration (121 Australian firms)
- Cleaned and renamed the columns, created dummy variables for state, and split a combined `Type` column into firm region and firm size.
- Profiled the firms by state: New South Wales has the most (36), followed by Victoria (32) and Queensland (30, or 24.8%), while Tasmania has only 2.
- Filtered firms by compliance score (61 firms score between 4 and 6), found the highest management-quality firm in each state, and grouped scores into Low, Mid and High categories.
- A boxplot of risk by firm size showed that small firms carry the highest median risk (about 2.3), medium firms about 2.1 and large firms the lowest (about 1.7), although large firms have several high-risk outliers.

### Clustering: segmenting firms by risk and growth
- Standardised the features (mean 0, standard deviation 1) so that no variable dominates the distance calculations.
- Used the elbow method to choose **k = 4** for K-Means, with a fixed `random_state` for reproducibility.
- Used a Ward-linkage dendrogram to choose 4 clusters for agglomerative clustering.
- Both methods found the same four-tier structure: high growth, moderate growth, low growth, and a small group of firms with severe negative growth and high risk. Agglomerative clustering produced cleaner horizontal cuts along growth thresholds, while K-Means clusters were shaped by distance to the centroid.

### Global health and wealth (country data, 1960 and 2025)
- Median life expectancy rose from 55.2 to 74.2 years, and the interquartile range narrowed from 19.8 to 10.3 years, which shows convergence between countries.
- Violin plots showed GDP growing from a compact distribution in 1960 to an extremely right-skewed one in 2025.
- Merged the GDP and life expectancy datasets and calculated correlations by region. The link strengthened in Asia (0.54 to 0.75), Africa (0.31 to 0.57) and Europe (0.55 to 0.67), and weakened in Oceania (0.91 to 0.77) and the Americas (0.64 to 0.53).
- The five countries with the largest GDP increase were Bermuda, Luxembourg, Ireland, Switzerland and Norway, and Europe had the largest median increase by region.
- Side-by-side scatter plots showed the Preston curve, in which gains in life expectancy flatten at higher income, plus clear regional convergence.

**Skills:** Data cleaning, grouping and aggregation, merging datasets, feature scaling, K-Means and hierarchical clustering, choosing the number of clusters, correlation analysis, visual storytelling.

---

## 2. Retail Customer Preference Modelling

**The data:** A personal-care retailer's customer survey (1,386 shoppers) with demographics, brand view, product-quality priority, income and item price. The goal was to understand and predict how much shoppers prefer premium products over value products.

### Regression: explaining preference
- Built a target variable, `Preference = log(PremiumShare / ValueShare)`, and checked linearity with scatterplots (linear against item price, not against income).
- A linear regression explained **66.9% of the variance** (R² = 0.669). Item price, income, brand view, quality priority and gender were statistically significant, while age, education and the strongest brand category were not.
- **Lasso (L1) regularisation** shrank five weaker predictors to exactly zero at a penalty of 0.1 and every coefficient to zero at a penalty of 1. Cross-validation (10-fold) chose a very small penalty (about 0.001), which kept all features.

### Tree-based models (85/15 train-test split)
- **Random forest:** The default model overfits (train MSE 0.057, test MSE 0.400). Limiting depth reduces the gap, and depth 4 gave the best test error (0.352).
- **Gradient boosting:** Tested five learning rates. A rate of 0.1 generalised best (test MSE 0.346). Higher rates drove training error down (0.051 at a rate of 1) while test error rose (0.625), which is a clear example of overfitting.
- **Feature importance:** Item price dominates (about 78% of importance), followed by income (about 9%), while brand perception contributes only about 2%.

### Classification: who is a premium buyer?
- Logistic regression classified shoppers as premium buyers or not, with 626 true positives, 552 true negatives, 100 false positives and 108 false negatives, and an **AUC of 0.94**. The write-up also explains how raising or lowering the cutoff probability trades true positives against false positives (the ROC curve).
- An unrestricted decision tree grew to depth 16 with 188 leaves, which overfits. Restricting it to depth 3 and 10 leaves gave a simple, interpretable tree with item price as the root split.
- **SHAP analysis** of a neural network classifier showed that item price and income have the widest spread of influence, and that higher values of both increase the likelihood of being a premium buyer.

**Skills:** Feature engineering, regression diagnostics and significance testing, regularisation, cross-validation, overfitting and bias-variance trade-offs, random forests, gradient boosting, logistic regression, ROC and AUC, decision trees, SHAP, communicating model results.

---
