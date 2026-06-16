import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from google import genai

load_dotenv()

app = Flask(__name__)

def is_placeholder(key):
    return not key or "here" in key.lower() or "your_" in key.lower()

# ── Load sentiment model ─────────────────────────────────────────────────────
# Using mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
# Lighter model (~80MB) vs FinBERT (~265MB) — fits in 512MB free tier RAM
# Still fine-tuned specifically on financial news text
print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    device=-1,
    top_k=None
)
print("Sentiment model loaded successfully.")

# ── Configure Gemini API ─────────────────────────────────────────────────────
gemini_key = os.getenv("GEMINI_API_KEY")
if not is_placeholder(gemini_key):
    gemini_client = genai.Client(api_key=gemini_key)
else:
    gemini_client = None

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


# ── Helper: Fetch headlines ──────────────────────────────────────────────────
def fetch_headlines(query, max_articles=15):
    if is_placeholder(NEWS_API_KEY):
        import random
        random.seed(sum(ord(c) for c in query))
        mock_templates = [
            ("{query} shares surge as quarterly earnings beat Wall Street estimates by 15%", "positive"),
            ("{query} faces regulatory scrutiny and potential fines over compliance issues", "negative"),
            ("{query} launches new AI-powered platform to drive future enterprise revenue", "positive"),
            ("{query} announces strategic partnership with leading cloud provider", "positive"),
            ("{query} CEO announces transition plan and changes in executive leadership", "neutral"),
            ("Analysts debate {query}'s market valuation and long-term growth prospects", "neutral"),
            ("Supply chain constraints could impact {query}'s shipments next quarter", "negative"),
            ("{query} increases dividend payment by 8% following record fiscal profits", "positive"),
            ("Competitor pressure intensifies for {query} in key international markets", "negative"),
            ("{query} schedules annual general meeting and shareholder vote", "neutral"),
            ("Brokerage upgrades {query} to Outperform with a revised target price", "positive"),
            ("Investors react to macroeconomic trends affecting {query}", "neutral"),
            ("Patent infringement lawsuit filed against {query} over software algorithms", "negative"),
            ("{query} invests $500M in green energy and sustainability projects", "positive"),
            ("Short interest in {query} rises ahead of key product announcements", "negative"),
        ]
        selected = random.sample(mock_templates, min(max_articles, len(mock_templates)))
        sources  = ["Bloomberg", "Reuters", "MarketWatch", "CNBC", "Financial Times"]
        headlines = []
        for i, (template, _) in enumerate(selected):
            headlines.append({
                "title":       template.replace("{query}", query),
                "source":      random.choice(sources),
                "publishedAt": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "url":         "https://www.reuters.com"
            })
        return headlines, None

    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q":        f"{query} stock OR earnings OR revenue OR profit OR market",
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": max_articles,
        "from":     from_date,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code != 200:
            return None, f"NewsAPI error {response.status_code}"
        articles = response.json().get("articles", [])
        if not articles:
            return None, "No recent news found for this company."
        headlines = []
        for a in articles:
            title = a.get("title", "").strip()
            if title and title != "[Removed]":
                headlines.append({
                    "title":       title,
                    "source":      a.get("source", {}).get("name", "Unknown"),
                    "publishedAt": a.get("publishedAt", "")[:10],
                    "url":         a.get("url", "#")
                })
        return headlines, None
    except Exception as e:
        return None, str(e)


# ── Helper: Run sentiment analysis ──────────────────────────────────────────
def analyze_sentiment(headlines):
    texts   = [h["title"] for h in headlines]
    results = sentiment_pipeline(texts, truncation=True, max_length=512)

    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    color_map = {"Positive": "#2ecc71",  "Negative": "#e74c3c",  "Neutral": "#95a5a6"}

    enriched = []
    for headline, result in zip(headlines, results):
        scores = {r["label"]: round(r["score"] * 100, 1) for r in result}
        top    = max(result, key=lambda x: x["score"])
        label  = label_map.get(top["label"].lower(), "Neutral")
        enriched.append({
            **headline,
            "sentiment":  label,
            "confidence": round(top["score"] * 100, 1),
            "color":      color_map[label],
            "scores": {
                "Positive": scores.get("positive", 0),
                "Negative": scores.get("negative", 0),
                "Neutral":  scores.get("neutral",  0),
            }
        })
    return enriched


# ── Helper: Compute stats ────────────────────────────────────────────────────
def compute_stats(enriched):
    counts     = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_conf = 0
    for item in enriched:
        counts[item["sentiment"]] += 1
        total_conf += item["confidence"]

    total       = len(enriched)
    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
    avg_conf    = round(total_conf / total, 1)

    if counts["Positive"] > counts["Negative"] and counts["Positive"] > counts["Neutral"]:
        verdict, verdict_color = "Bullish 📈", "#2ecc71"
    elif counts["Negative"] > counts["Positive"] and counts["Negative"] > counts["Neutral"]:
        verdict, verdict_color = "Bearish 📉", "#e74c3c"
    else:
        verdict, verdict_color = "Neutral ➡️", "#f39c12"

    return {
        "counts": counts, "percentages": percentages,
        "avg_confidence": avg_conf, "total": total,
        "verdict": verdict, "verdict_color": verdict_color,
    }


# ── Helper: Gemini summary ───────────────────────────────────────────────────
def generate_summary(company, enriched, stats):
    if gemini_client is None:
        pos = stats['percentages']['Positive']
        neg = stats['percentages']['Negative']
        neu = stats['percentages']['Neutral']
        return (
            f"[Demo Mode - Add GEMINI_API_KEY for live AI summaries] "
            f"Sentiment analysis of recent {company} headlines shows a {stats['verdict']} market signal. "
            f"{pos}% positive, {neg}% negative, {neu}% neutral coverage."
        )

    headlines_text = "\n".join([
        f"[{h['sentiment']} - {h['confidence']}%] {h['title']}"
        for h in enriched[:10]
    ])
    prompt = (
        f"You are a financial analyst. Based on these recent headlines about {company}, "
        f"write a concise 3-4 sentence market summary covering the sentiment trend, "
        f"key themes, and notable events. Be factual and professional.\n\n"
        f"Headlines:\n{headlines_text}\n\n"
        f"Overall: {stats['percentages']['Positive']}% Positive, "
        f"{stats['percentages']['Negative']}% Negative, "
        f"{stats['percentages']['Neutral']}% Neutral.\n\nMarket Summary:"
    )
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data    = request.get_json()
    company = data.get("company", "").strip()

    if not company:
        return jsonify({"error": "Please enter a company name or ticker."}), 400
    if len(company) > 50:
        return jsonify({"error": "Input too long."}), 400

    headlines, error = fetch_headlines(company)
    if error:
        return jsonify({"error": error}), 400

    enriched = analyze_sentiment(headlines)
    stats    = compute_stats(enriched)
    summary  = generate_summary(company, enriched, stats)

    return jsonify({
        "company":     company,
        "headlines":   enriched,
        "stats":       stats,
        "summary":     summary,
        "analyzed_at": datetime.now().strftime("%B %d, %Y at %I:%M %p")
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "distilroberta-financial-sentiment"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
