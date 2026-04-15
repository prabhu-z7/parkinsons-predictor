# 🧠 Parkinson's Disease Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

**92.3% Accurate SVM model** that predicts Parkinson's Disease from 22 voice biomarkers. Auto-loads UCI ML Dataset #174.

## ✨ **Live Demo**
[Try the app](https://yourusername-parkinsons-predictor.hf.space)

## 📊 **Performance**
| Metric | Score |
|--------|-------|
| Test Accuracy | **92.3%** |
| ROC-AUC | **98.6%** |
| PD Recall | **93%** |

**Model:** RBF SVM (C=10, γ=0.1)

## 🗄️ **Dataset**
- **Source:** [UCI ML Repository #174](https://archive.ics.uci.edu/dataset/174/parkinsons)
- **Samples:** 195 voice recordings
- **Features:** 22 biomedical voice measures
- **Citation:** `DOI: 10.24432/C59C74`

## 🚀 **Quick Start**

```bash
# Clone repo
git clone https://github.com/YOURUSERNAME/parkinsons-predictor
cd parkinsons-predictor

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 🛠️ **Files**
