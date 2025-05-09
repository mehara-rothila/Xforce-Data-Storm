# 🚀 Insurance Agent NILL Prediction - Data Storm v6.0 Championship Solution | [LIVE DASHBOARD](https://dashboard.mehara.io/)

## First Place Solution for ABC Insurance Company's Agent Performance Challenge

## 📋 Project Overview

This repository contains our award-winning solution for predicting insurance agents who are at risk of not selling policies in the following month (One Month NILL). Our approach leverages advanced ensemble modeling with SMOTE data augmentation and Optuna hyperparameter optimization to achieve state-of-the-art prediction accuracy.

## 🌟 Key Features

- Advanced ensemble modeling with weighted predictions
- Time-series based cross-validation with stratification
- Feature importance stability analysis
- Dynamic thresholding for agent-specific predictions
- Custom intervention recommendations based on risk profile

## 📊 Live Dashboard

### ✨ **[INTERACTIVE DASHBOARD: https://dashboard.mehara.io/](https://dashboard.mehara.io/)** ✨

We've deployed a comprehensive interactive dashboard to visualize predictions and insights. This dashboard is a key component of our solution and provides:

- Real-time model performance metrics
- Agent risk classification with filtering options
- Interactive feature importance analysis
- Time series forecasting with trend visualization
- Personalized action recommendations based on agent profiles

The dashboard is optimized for both desktop and mobile devices, allowing managers to monitor agent performance on the go.

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Create a virtual environment (optional but recommended)
python -m venv nill_env
source nill_env/bin/activate  # On Windows: nill_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna joblib matplotlib seaborn tqdm
```

### Directory Structure

```
project_root/
├── dataset/                           # Data files
│   ├── train_storming_round.csv
│   ├── test_storming_round.csv
│   └── sample_submission_storming_round.csv
├── 094400-public.py                   # Championship Model (0.944 on public leaderboard)
├── 094508-public.py                   # Ceiling-Breaking Model (0.945 on public leaderboard)
├── 094640-public.py                   # Ultra-Optimized Champion Model (0.946 on public leaderboard)
├── EDA_and_Feature_Analysis.ipynb     # Exploratory Data Analysis
├── README.md                          # This file
└── outputs/                           # Will be created automatically
```

> **Note:** The Python filenames (094400, 094508, 094640) directly correspond to our public leaderboard scores, showcasing our model's evolution and improvement.

## 🚀 Running the Models

### Championship Model (Recommended for Submission)

```bash
python 094400-public.py
```

This model uses an ensemble approach with SMOTE data augmentation and Optuna hyperparameter optimization, achieving 91.8% accuracy on validation data and 0.944 score on the public leaderboard.

### Ceiling-Breaking Model

```bash
python 094508-public.py
```

This model focuses on breaking accuracy ceiling with specialized threshold calibration for different agent patterns, reaching 0.945 on the public leaderboard.

### Ultra-Optimized Champion Model

```bash
python 094640-public.py
```

This model implements segment-specific thresholds and weighted ensemble predictions, achieving our highest score of 0.946 on the public leaderboard.

## 📓 Running the Jupyter Notebook

Our EDA and feature analysis notebook can be run with:

```bash
# Navigate to the notebooks directory
cd notebooks

# Start Jupyter
jupyter notebook EDA_and_Feature_Analysis.ipynb

# Alternatively, use JupyterLab
jupyter lab EDA_and_Feature_Analysis.ipynb
```

The notebook contains:
- Comprehensive exploratory data analysis
- Feature engineering explanations
- Performance visualization
- Agent segmentation analysis

## 📈 Model Performance

| Model                    | Accuracy | Precision | Recall  | F1 Score |
|--------------------------|----------|-----------|---------|----------|
| Championship Model       | 91.8%    | 88.2%     | 86.5%   | 87.3%    |
| Ceiling-Breaking Model   | 94.2%    | 86.7%     | 89.3%   | 88.0%    |
| Ultra-Optimized Model    | 90.2%    | 89.4%     | 83.1%   | 86.1%    |

## 🔍 Key Insights

Our analysis revealed several important factors affecting agent performance:
- Activity consistency is more predictive than total activity volume
- Conversion rates from quotations to policies strongly indicate future performance
- Agents with 3+ consecutive NILL months have 78% chance of another NILL month
- First 90 days of agent activity establish long-term performance patterns

## 📝 Personalized Action Plans

The models generate personalized interventions based on agent profiles:
1. High-risk agents receive targeted 1:1 coaching and daily activity monitoring
2. Medium-risk agents get specialized training on conversion optimization
3. Low-risk agents receive performance maintenance plans and mentorship assignments

## 👨‍💻 Team

- Data Storm v6.0 Championship Team
- First Place Solution

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.