# 🚀 Hybrid Anomaly Detection System for Stock Market Manipulation

> 🔍 A Data Science project combining **Financial Domain Knowledge** and
> **Machine Learning** to detect suspicious stock behaviors in
> real-time.

------------------------------------------------------------------------

## 🧑‍💻 Author

**Le Nhat Truong**\
Data Science Enthusiast \| Machine Learning \| Finance Analytics

------------------------------------------------------------------------

## 📌 Project Highlights

-   🔥 Hybrid model: Rule-based + Isolation Forest
-   ⚡ Real-time anomaly detection (\~32.7 µs/sample)
-   📈 High performance: F1-score **0.748**, ROC-AUC **0.983**
-   🧠 Explainable AI (XAI-friendly)
-   💻 Runs on standard hardware (no GPU required)

------------------------------------------------------------------------

## 🎯 Problem Statement

Stock market manipulation (pump & dump, abnormal trading) is difficult
to detect early.\
This project builds a system to:

-   Detect unusual trading patterns
-   Alert investors in real-time
-   Improve market transparency

------------------------------------------------------------------------

## 🧠 Solution Architecture

### 🔹 Hybrid Model

We combine:

1.  **Rule-based System (60%)**
    -   Financial indicators:
        -   RSI
        -   Bollinger Bands
        -   Volume spikes
        -   Daily returns
    -   Provides interpretability
2.  **Machine Learning (40%)**
    -   Model: Isolation Forest
    -   Detects non-linear anomalies
    -   Unsupervised learning (no labels required)

------------------------------------------------------------------------

## ⚙️ Pipeline

``` mermaid
graph TD
A[Data Collection] --> B[Preprocessing]
B --> C[Feature Engineering]
C --> D[Rule-based Detection]
C --> E[Isolation Forest]
D --> F[Hybrid Scoring]
E --> F
F --> G[Alert System]
```

------------------------------------------------------------------------

## 📊 Model Performance

  Model              Precision   Recall      F1-score    ROC-AUC
  ------------------ ----------- ----------- ----------- -----------
  Isolation Forest   0.587       0.607       0.597       0.955
  Z-score            0.562       0.885       0.688       0.974
  ⭐ Hybrid Model    **0.700**   **0.803**   **0.748**   **0.983**

------------------------------------------------------------------------

## 📦 Features Used

-   Log-return
-   Volatility
-   RSI
-   MACD
-   ATR
-   Volume ratio

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Clone repo

``` bash
git clone https://github.com/yourusername/hybrid-anomaly-stock.git
cd hybrid-anomaly-stock
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run notebook

``` bash
jupyter notebook
```

------------------------------------------------------------------------

## 📈 Use Cases

-   📊 Stock screening tools
-   💼 Retail investor decision support
-   🏦 Financial institutions
-   🪙 Crypto anomaly detection

------------------------------------------------------------------------

## ⚠️ Limitations

-   Fixed contamination rate (0.01)
-   No market index normalization
-   Limited dataset scope

------------------------------------------------------------------------

## 🔮 Future Improvements

-   NLP sentiment analysis (news + social media)
-   Deep learning (Autoencoders)
-   Cross-market generalization
-   Backtesting engine

------------------------------------------------------------------------

## 🏁 Conclusion

This project demonstrates how combining **domain knowledge + ML** can
create: - Efficient - Interpretable - Scalable anomaly detection systems

------------------------------------------------------------------------

## ⭐ If you like this project

Give it a ⭐ on GitHub and connect with me!

------------------------------------------------------------------------

## 📅 Last Updated

2026-04-10
