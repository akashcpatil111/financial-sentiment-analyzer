# Financial News Sentiment Analyzer 📊

A real-time financial news sentiment analysis dashboard powered by **FinBERT** (Hugging Face) and **Gemini API**. Enter any company name or stock ticker to instantly analyze recent news sentiment with AI-generated market summaries.

🔗 **[Live Demo](https://financial-sentiment-analyzer.onrender.com)** *(update with your Render URL)*

---

## ✨ Features

- **Real-time news fetching** — pulls latest financial headlines via NewsAPI
- **FinBERT sentiment classification** — industry-standard NLP model fine-tuned on financial text (~97% accuracy)
- **Confidence scores** — each headline shows prediction confidence
- **Interactive charts** — sentiment distribution donut chart + per-headline bar chart
- **AI market summary** — Gemini API generates a concise analyst-style summary
- **Overall verdict** — Bullish / Bearish / Neutral market signal
- **Dark mode dashboard** — clean, professional UI

---

## 🧠 How It Works

```
User inputs company name
         │
         ▼
   NewsAPI (last 7 days)     ← Fetches up to 15 recent headlines
         │
         ▼
   ProsusAI/FinBERT           ← Classifies each headline:
   (Hugging Face)                Positive / Negative / Neutral
         │                       + confidence score per label
         ▼
   Aggregate Statistics       ← Counts, percentages, overall verdict
         │
         ▼
   Gemini 2.0 Flash           ← Generates market summary from headlines
         │
         ▼
   Flask Dashboard            ← Renders charts, headlines, summary
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| NLP Model | ProsusAI/FinBERT (Hugging Face Transformers) |
| LLM | Google Gemini 2.0 Flash |
| News Source | NewsAPI |
| Frontend | HTML5, CSS3, Chart.js |
| Deployment | Render |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/akashcpatil111/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Get your free API keys:
- **NewsAPI:** https://newsapi.org/register (free tier: 100 requests/day)
- **Gemini API:** https://aistudio.google.com/ (free tier available)

### 4. Run the app
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🌐 Deploy on Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set environment variables: `NEWS_API_KEY`, `GEMINI_API_KEY`
5. Deploy — Render auto-detects `render.yaml`

> **Note:** First request after deployment may take 30–60 seconds (model loading on cold start). Subsequent requests are fast.

---

## 📁 Project Structure

```
financial-sentiment-analyzer/
├── app.py                  ← Flask backend + FinBERT + Gemini logic
├── templates/
│   └── index.html          ← Frontend dashboard (Chart.js)
├── requirements.txt
├── render.yaml             ← Render deployment config
├── .env.example            ← Environment variables template
├── .gitignore
└── README.md
```

---

## 🔮 Potential Extensions

- Add stock price correlation (Yahoo Finance API)
- Sentiment trend over time (last 30 days)
- Email alerts when sentiment shifts significantly
- Support for multiple tickers simultaneously
- Export sentiment report as PDF

---

## 📄 License

MIT License
