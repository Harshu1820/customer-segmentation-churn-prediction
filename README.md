# customer-segmentation-churn-prediction
Analytical model for Customer Segmentation

An end-to-end, production-ready data science pipeline built to transform raw e-commerce transaction logs into actionable business intelligence. This framework implements a complete data lifecycle: parsing raw unstructured transactions, engineering dense behavioral features (RFM), isolating optimal customer segments using unsupervised algorithms, predicting churn using ensemble models, and charting strategic product revenue rankings.

## Key Pipeline Architecture & Engineering Phases

### 1. Production-Grade Data Cleansing & Data Engineering
* *Type Conversion & Handling:* Solved structural parsing issues by converting raw object IDs (CustomerID) into integers and handling mixed-format strings in date columns into standard, normalized datetimes.
* *Anomaly Filtering:* Engineered automated cleansing rules to scrub canceled transactions, negative quantity values, pricing errors, and missing database attributes across critical categorical identifiers.

### 2. Behavioral Feature Engineering & Exploratory Data Analysis (EDA)
* *RFM Calculation Matrix:* Aggregate continuous raw invoice records into a structured framework measuring customer behavioral metrics: *Recency* (days since last purchase), *Frequency* (total transaction volume), and *Monetary Value* (total capital spent).
* *Statistical Correlation Profiling:* Implemented Seaborn-mapped correlation heatmaps to assess linear dependencies, evaluate multi-collinearity, and understand feature weights prior to feeding downstream machine learning models.

### 3. Advanced Unsupervised Machine Learning (Market Segmentation)
* *K-Means Centroid Isolation:* Executed iterative distance-based optimizations verified via the Elbow Method (Within-Cluster Sum of Squares) to mathematically isolate distinct behavioral groups.
* *Hierarchical Tree-Linkage:* Implemented agglomerative tree-linkage architectures (scipy.cluster.hierarchy) using Ward's variance minimization technique to establish logical customer hierarchies.
* *DBSCAN (Density-Based Separation):* Deployed density-based scaling algorithms to group highly concentrated behavioral cohorts while filtering out noisy, fringe consumer profiles.
* *Clustering Validation Metrics:* Evaluated partition validity using dense mathematical parameters—including the *Silhouette Score* (achieved ~0.602 on Hierarchical tracks), *Davies-Bouldin Index, and **Calinski-Harabasz Metrics*—to prevent cluster overlap.

### 4. Supervised Predictive Churn Modeling
* *Ensemble Classification:* Built and fine-tuned robust predictive models using *Random Forest* and *XGBoost Classifiers* to actively anticipate customer attrition based on their RFM feature drift.
* *Rigorous Model Evaluation:* Tracked model stability and generalization power using robust metrics: Classification Accuracy, Precision-Recall vectors, F1-Scores, and the Area Under the ROC Curve (ROC-AUC).

### 5. Automated Inventory Analytics & Strategic Product Metrics
* *Revenue Aggregations:* Developed pandas-driven transaction groupings to monitor item-level demand trends and track operational revenue metrics.
* *Product Ranking Visualizations:* Designed automated Seaborn bar plots mapping top selling products (e.g., Paper Craft, Regency Cakestand, White Hanging Heart) to isolate key inventory drivers and mitigate stock outlays.

---

## Core Technical Stack
* *Language:* Python
* *Data Infrastructure:* Pandas, NumPy
* *Machine Learning & Math Frameworks:* Scikit-learn, XGBoost, SciPy (Cluster Hierarchy modules)
* *Data Visualization & Analytics Engines:* Matplotlib, Seaborn

---

## Developer Profile
* *Data Engineer:* Harshada Anil Hulawale
* *Academic Track:* Bachelor of Engineering in Computer Engineering
* *University Affiliate:* Savitribai Phule Pune University (SPPU), India
