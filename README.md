# PhishShield
PhishShield is an advanced phishing URL detection framework designed to safeguard users and organizations from deceptive web links used in phishing attacks. It employs a robust approach to identify visually misleading URLs that exploit techniques such as homographs, zero-width characters, punycode, bit-squatting, and typo-squatting to trick users into visiting fraudulent websites.

By analyzing both URL structures and webpage content, PhishShield enhances detection accuracy and mitigates evolving phishing threats. Unlike traditional detection methods that rely on static rule-based approaches, this framework continuously updates its detection capabilities, allowing it to adapt to new attack patterns without requiring complete retraining. Additionally, PhishShield offers customizable security profiles, catering to different security needs across various user levels and organizational settings.

Built on a well-structured dataset of legitimate and phishing URLs, it ensures a more reliable and scalable defense mechanism against cyber threats. Its efficient detection methodology makes it a powerful tool in combating phishing attacks, ensuring a safer browsing experience in today’s rapidly evolving digital landscape.

To run the whole project in a single run, run the python file **main.py**

This will run the scripts in the following sequence:

├── data_preprocessing.py
├── data_visualization.py
├── handle_outliers.py
├── train_logistic_regression.py
├── Train_Ridge_Lasso.py
├── train_decision_tree.py
├── train_knn.py
├── train_random_forest.py
├── train_svc.py
├── Model_Evaluation.py
├── random_forest_top20_vs_all.py
├── model_testing.py
├── app.py

Once the execution is complete, visit the following URL to check whether a given URL is phishing or legitimate:
**http://127.0.0.1:5000/**
