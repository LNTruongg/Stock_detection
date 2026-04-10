# 🚀 Hybrid Anomaly Detection System for Stock Market Manipulation

> A production-oriented hybrid system combining financial domain knowledge and machine learning to detect stock manipulation in real-time.

---

## 📌 Overview

This project detects anomalous trading behavior using:
- Rule-based financial heuristics
- Isolation Forest (unsupervised ML)
- Hybrid scoring mechanism

---

## 🎯 Objectives
- Detect unusual price & volume movements
- Provide interpretable signals
- Balance accuracy, speed, explainability

---

## 🧠 Methodology

### Rule-based
- RSI, Bollinger Bands, Volume spikes
- Human-readable logic

### Machine Learning
- Isolation Forest
- Detects hidden patterns

### Hybrid Score
Score = 0.6 * Rule + 0.4 * ML

---

## 🏗️ Pipeline
1. Data Collection
2. Preprocessing
3. Feature Engineering
4. Detection
5. Scoring
6. Alert

---

## 📊 Performance
- F1-score: 0.748
- ROC-AUC: 0.983
- Speed: ~32.7 µs/sample

---

## 📈 Visualizations

![Anomaly Plot](notebooks/images/anomaly_plot.png)
![Feature Distribution](notebooks/images/feature_distribution.png)

---

## ⚡ Features
- Explainable AI
- Real-time processing
- CPU-friendly
- Cold-start ready

---

## 🌍 Use Cases
- Detect manipulation
- Risk warning
- Market analysis

---

## ⚠️ Limitations
- Fixed contamination rate
- No market normalization

---

## 🔮 Future Work
- NLP sentiment analysis
- Autoencoder
- More backtesting

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy

---

## 🚀 Run

```bash
git clone https://github.com/your-repo/project.git
pip install -r requirements.txt
python main.py
```

---

## 👨‍💻 Authors
FPT University Team
