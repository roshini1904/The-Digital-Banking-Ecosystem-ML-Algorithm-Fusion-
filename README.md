# 🏦 The Digital Banking Ecosystem — ML Algorithm Fusion
## 🌐 Live Demo **[Click Here to Open Project]https://roshini1904.pythonanywhere.com/dashboard

> MCA Final Year Capstone Project | AI-Powered Banking System

---

## 📌 Project Overview

Digital Banking  is a full-stack intelligent banking web application
that implements three Machine Learning algorithms completely from
scratch in pure Python — without using scikit-learn or NumPy.
The system demonstrates real banking workflows including loan
approval, fraud detection, and credit limit management.

---

## 🤖 ML Algorithms Implemented

| Algorithm | Purpose | Accuracy |
|---|---|---|
| K-Nearest Neighbors (KNN) | Loan Approval | 92.31% |
| Random Forest (CART Trees) | Fraud Detection | 92.31% |
| Q-Learning (RL) | Credit Adjustment | — |

### Key Points
- All algorithms implemented in **pure Python**
- No scikit-learn, no NumPy
- Evaluated on **25% held-out test split**
- Confusion Matrix, Precision, Recall, F1 Score computed

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + Flask |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Auth | SHA-256 + Flask Sessions |
| ML | Pure Python (from scratch) |

---

## ✨ Features

### Customer Side
- ✅ Register and Login securely
- ✅ Apply for Loan (KNN prediction)
- ✅ Check Transaction Fraud (Random Forest)
- ✅ Get Credit Adjustment (Q-Learning)
- ✅ Submit Credit Limit Increase Request
- ✅ View request status and history

### Admin Side
- ✅ Real-time notification with sound alert
- ✅ Credit request approval workflow
- ✅ AI evaluation using Q-Learning agent
- ✅ Live charts from customer data
- ✅ Model evaluation metrics dashboard
- ✅ Q-Table with live retraining panel
- ✅ User management (Add/Edit/Delete)
- ✅ Decision audit logs

---

## 📊 Model Evaluation Results

### KNN — Loan Approval
| Metric | Score |
|---|---|
| Accuracy | 92.31% |
| Precision | 88.89% |
| Recall | 100.00% |
| F1 Score | 94.12% |

### Random Forest — Fraud Detection
| Metric | Score |
|---|---|
| Accuracy | 92.31% |
| Precision | 100.00% |
| Recall | 83.33% |
| F1 Score | 90.91% |

---

## 🚀 How to Run Locally

```bash
# Step 1 - Clone the repository
git clone https://github.com/roshini1904/The-Digital-Banking-Ecosystem-ML-Algorithm-Fusion-.git

# Step 2 - Go into the folder
cd The-Digital-Banking-Ecosystem-ML-Algorithm-Fusion-

# Step 3 - Install dependencies
pip install flask

# Step 4 - Run the application
python app.py

# Step 5 - Open in browser
http://127.0.0.1:5000
```

---

## 🔑 Login Credentials

| Role | Username | Password |
|---|---|---|
| Admin | admin | Admin@123 |
| Customer | customer | Customer@123 |

> New customers can register at /register

---

## 📁 Project Structure
Digital Banking/
├── app.py ← Flask app + ML models
├── requirements.txt ← Dependencies
├── README.md ← This file
└── templates/
├── login.html ← Login page
├── register.html ← Registration page
├── customer_dashboard.html ← Customer portal
└── admin_dashboard.html ← Admin console
---

## 🧠 Algorithm Details

### KNN Loan Approval
- Distance Metric: Euclidean Distance
- Normalization: Min-Max Scaling
- K Value: 3 neighbors
- Features: Income, Credit Score, Loan Amount

### Random Forest Fraud Detection
- Trees: 15 CART Decision Trees
- Split Criterion: Gini Impurity
- Voting: Weighted Majority
- Features: Amount, Location Mismatch, Late Night

### Q-Learning Credit Adjustment
- States: 8 discrete customer risk levels
- Actions: Decrease / Hold / Increase limit
- Gamma: 0.9 (discount factor)
- Alpha: 0.1 (learning rate)
- Episodes: 500 training rounds
- Update Rule: Bellman Equation

---

## 🔐 Security Features

- SHA-256 password hashing
- Session-based authentication
- Role-based access control (Admin/Customer)
- Protected API endpoints
- Decision audit logging

---

## 👩‍💻 Developed By

**Roshini N**
MCA Final Year Student
Capstone Project — 2026

---

## 📄 Note

This is a demo deployment.
Database resets on server restart.
For permanent storage, PostgreSQL can be integrated.
