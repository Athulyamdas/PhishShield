import subprocess
import sys

scripts = [
    "data_preprocessing.py",
    "data_visualization.py",
    "handle_outliers.py",
    "train_logistic_regression.py",
    "train_Ridge_Lasso.py",
    "train_decision_tree.py",
    "train_knn.py",
    "train_random_forest.py",
    "train_svc.py",
    "model_Evaluation.py",
    "random_forest_top20_vs_all.py",
    "model_testing.py",
    "app.py"
]

for script in scripts:
    print(f"\nRunning {script}...")
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"{script} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script}: {e}")
        break
