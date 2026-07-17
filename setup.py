from setuptools import setup, find_packages

setup(
    name="Explainable-Credit-Card-Fraud-Detection",
    version="1.0.0",
    author="Anadhyanth",
    description="Explainable Credit Card Fraud Detection using XGBoost and SHAP",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "shap",
        "joblib",
    ],
    python_requires=">=3.10",
)