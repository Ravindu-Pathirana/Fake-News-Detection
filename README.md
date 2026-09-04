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
	•	XGBoost
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

	•	Best model (label-corrected, macro-F1): Naive Bayes on text + metadata, test macro-F1 = 0.655
	•	Metric: macro-F1 is used throughout (not accuracy or positive-class F1), since the dataset is class-imbalanced
	•	Classical models are tuned by 5-fold cross-validation over the combined LIAR train+validation split (test is the only held-out data reported); the DistilBERT reference model instead follows the official train/validation/test split, so its training-data budget is not identical to the classical models'
	•	Correcting for multiple comparisons (Holm-Bonferroni, 15-comparison family): the previously reported Logistic Regression gain (0.630 vs 0.596 macro-F1) does *not* survive correction (p_holm = 0.38), but the actual best model's gain does -- Naive Bayes text+metadata vs text-only, 0.655 vs 0.604, p_holm = 0.045
	•	Generalization audit: despite the LIAR split being 85.2% speaker-overlapping, the metadata gain survives speaker- and subject-disjoint evaluation (only a 0.01-0.03 macro-F1 gap vs a matched random-split control) -- a much larger, but base-rate-confounded, gap appears under a party-shift protocol
	•	Explainability analysis (SHAP) shows the model relies heavily on topical/entity features, including party affiliation -- reported as evidence of dataset-level topical bias, not of learned deception cues; a speaker-disjoint SHAP check confirms this partisan signal is not simply speaker memorization

An earlier draft of this project reported an F1-score of 0.87, which was traced to a
label-mapping bug (see `Fake-News-Detection/CLAUDE.md`); the corrected, honest
performance range for LIAR-binary classification with classical models is macro-F1
~0.60-0.65, which this project treats as a normal, reportable result rather than a
shortfall. The full generalization-audit and significance-correction writeup is in
`claude-working-files/paper_icac2026.tex`.

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

🧠 Key Insights (verified on corrected labels)

	•	Metadata (subject/party/state/job/context) improves every model over its text-only counterpart; adding it produces the largest single gain in this study
	•	Naive Bayes on text+metadata is the strongest configuration overall, ahead of Random Forest and XGBoost -- ensemble/tree models do not outperform simpler models here
	•	Chi-square selects better features than Mutual Information for most models tested (3 of 5: Random Forest, SVM, XGBoost); Mutual Information is better for Logistic Regression (0.620 vs 0.615) and Naive Bayes (0.587 vs 0.582)
	•	Feature selection is fit inside every cross-validation fold (not once beforehand), so hyperparameter search never sees labels from its own held-out fold
	•	A further hyperparameter-tuning step, despite looking best under cross-validation, does not produce a statistically significant gain on held-out test data (McNemar p = 0.663, raw); after correcting for the full family of significance comparisons this project makes, the feature-selection-stage gain itself (p_holm = 0.38) is not distinguishable from noise either -- it's the actual best model's (Naive Bayes) metadata gain that survives correction
	•	SHAP shows the model relies substantially on topical/entity/partisan features -- useful for auditing what the model actually keys on, not evidence the model detects deception

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

