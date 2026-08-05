# Fake-News-Detection
This repository presents an Explainable Fake News Detection system developed using classical machine learning techniques and evaluated on the LIAR benchmark dataset. The project focuses not only on improving classification performance but also on enhancing model interpretability using Explainable AI (XAI) methods.

🚀 Key Features

	•	🔍 Binary Fake News Classification (Fake vs Real)
	•	📊 Baseline Reproduction using:
	•	Logistic Regression (LR)
	•	Support Vector Machines (SVM)
	•	⚡ Advanced Models:
	•	Naive Bayes (NB)
	•	Random Forest (RF)
	•	XGBoost (Best performing)
	•	🧠 Feature Engineering:
	•	TF-IDF (Unigrams + Bigrams)
	•	Meta-data integration (speaker, party, context, etc.)
	•	🎯 Feature Selection Techniques:
	•	Chi-Square (χ²)
	•	Mutual Information (MI)
	•	📈 Hyperparameter Optimization using GridSearchCV
	•	🔎 Explainability Analysis:
	•	SHAP (global & local interpretation)
	
📊 Results Summary

	•	Best Model: XGBoost
	•	Achieved F1-score up to 0.87
	•	Feature selection significantly improved performance
	•	Explainability analysis shows models rely on meaningful linguistic patterns  ￼

🧪 Dataset

	•	LIAR Dataset (Wang, 2017)
	•	12,836 labeled political statements
	•	Converted from 6-class → binary classification
	•	Includes:
	•	Train / Validation / Test splits
	•	Text + meta-data features

⚙️ Methodology Overview

	1.	Data preprocessing (cleaning, tokenization, stemming)
	2.	TF-IDF feature extraction
	3.	Feature selection (χ², MI)
	4.	Model training & tuning
	5.	Evaluation using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	6.	Explainability using SHAP

🧠 Key Insights

	•	Feature selection is more important than model choice
	•	Ensemble models outperform linear models
	•	Mutual Information provides more stable results
	•	Explainability improves trust in predictions  ￼

🛠️ Tech Stack

	•	Python
	•	Scikit-learn
	•	XGBoost
	•	Pandas / NumPy
	•	SHAP (Explainable AI)

📌 Future Work

	•	Incorporate social/contextual features
	•	Apply deep learning (LSTM, BERT)
	•	Improve real-world deployment robustness

👨‍💻 Author

Ravindu Pathirana

University of Moratuwa

