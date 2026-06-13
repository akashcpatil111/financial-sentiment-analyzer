import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)

def is_placeholder(key):
    return not key or "here" in key.lower() or "your_" in key.lower()

# ── Load FinBERT model (loaded once at startup) ──────────────────────────────
# ProsusAI/finbert is a BERT model fine-tuned specifically on financial text
# It achieves ~97% accuracy on financial sentiment classification
print("Loading FinBERT model... (first load may take ~1 minute)")
sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=-1,          # -1 = CPU
    top_k=None          # return scores for all 3 labels
)
print("FinBERT loaded successfully.")

# ── Configure Gemini API ─────────────────────────────────────────────────────
gemini_key = os.getenv("GEMINI_API_KEY")
if not is_placeholder(gemini_key):
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
else:
    gemini_model = None

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


# ── Helper: Fetch headlines from NewsAPI ─────────────────────────────────────
def fetch_headlines(query, max_articles=15):
    """Fetch recent financial headlines for a given company/ticker."""
    if is_placeholder(NEWS_API_KEY):
        import random
        # Seed the random number generator using the query so that the same search yields consistent headlines
        random.seed(sum(ord(c) for c in query))
        
        mock_templates = [
            ("{query} shares surge as quarterly earnings beat Wall Street estimates by 15%", "Positive"),
            ("{query} faces regulatory scrutiny and potential fines over compliance issues", "Negative"),
            ("{query} launches new AI-powered platform to drive future enterprise revenue", "Positive"),
            ("{query} announces strategic partnership with leading cloud provider to expand operations", "Positive"),
            ("{query} CEO announces transition plan and changes in executive leadership team", "Neutral"),
            ("Analysts debate {query}'s market valuation and long-term growth prospects", "Neutral"),
            ("Supply chain constraints could impact {query}'s shipments next quarter, warns CFO", "Negative"),
            ("{query} increases dividend payment by 8% following record fiscal profits", "Positive"),
            ("Competitor pressure intensifies for {query} in key international markets", "Negative"),
            ("{query} schedules annual general meeting and shareholder vote for late next month", "Neutral"),
            ("Brokerage upgrades rating on {query} to 'Outperform' with a revised target price", "Positive"),
            ("Investors react to latest industry-wide macroeconomic trends affecting {query}", "Neutral"),
            ("Patent infringement lawsuit filed against {query} over software algorithms", "Negative"),
            ("{query} invests $500M in green energy initiatives and sustainability projects", "Positive"),
            ("Short interest positions in {query} rise ahead of key product announcements", "Negative")
        ]
        
        selected = random.sample(mock_templates, min(max_articles, len(mock_templates)))
        
        headlines = []
        for i, (template, _) in enumerate(selected):
            title = template.replace("{query}", query)
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            sources = ["Bloomberg", "Reuters", "MarketWatch", "CNBC", "Financial Times", "Wall Street Journal"]
            headlines.append({
                "title": title,
                "source": random.choice(sources),
                "publishedAt": date,
                "url": "https://www.reuters.com"
            })
        return headlines, None

    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": f"{query} stock OR earnings OR revenue OR profit OR market",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "from": from_date,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=10)
    if response.status_code != 200:
        return None, f"NewsAPI error: {response.status_code} (Add your NEWS_API_KEY to the .env file to run live)"

    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        return None, "No recent news found for this company."

    headlines = []
    for a in articles:
        title = a.get("title", "").strip()
        if title and title != "[Removed]":
            headlines.append({
                "title": title,
                "source": a.get("source", {}).get("name", "Unknown"),
                "publishedAt": a.get("publishedAt", "")[:10],
                "url": a.get("url", "#")
            })

    return headlines, None


# ── Helper: Run FinBERT sentiment on headlines ───────────────────────────────
def analyze_sentiment(headlines):
    """Run FinBERT on each headline and return enriched results."""
    texts = [h["title"] for h in headlines]
    results = sentiment_pipeline(texts, truncation=True, max_length=512)

    enriched = []
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}

    for headline, result in zip(headlines, results):
        # result is a list of {label, score} for all 3 classes
        scores = {r["label"]: round(r["score"] * 100, 1) for r in result}
        top = max(result, key=lambda x: x["score"])
        label = label_map.get(top["label"], top["label"])

        enriched.append({
            **headline,
            "sentiment": label,
            "confidence": round(top["score"] * 100, 1),
            "color": color_map[label],
            "scores": {
                "Positive": scores.get("positive", 0),
                "Negative": scores.get("negative", 0),
                "Neutral":  scores.get("neutral", 0),
            }
        })

    return enriched


# ── Helper: Aggregate sentiment stats ────────────────────────────────────────
def compute_stats(enriched):
    """Compute overall sentiment distribution and average confidence."""
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_conf = 0

    for item in enriched:
        counts[item["sentiment"]] += 1
        total_conf += item["confidence"]

    total = len(enriched)
    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
    avg_conf = round(total_conf / total, 1)

    # Overall verdict
    dominant = max(counts, key=counts.get)
    if counts["Positive"] > counts["Negative"] and counts["Positive"] > counts["Neutral"]:
        verdict = "Bullish 📈"
        verdict_color = "#2ecc71"
    elif counts["Negative"] > counts["Positive"] and counts["Negative"] > counts["Neutral"]:
        verdict = "Bearish 📉"
        verdict_color = "#e74c3c"
    else:
        verdict = "Neutral ➡️"
        verdict_color = "#f39c12"

    return {
        "counts": counts,
        "percentages": percentages,
        "avg_confidence": avg_conf,
        "total": total,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }


# ── Helper: Generate Gemini market summary ───────────────────────────────────
def generate_summary(company, enriched, stats):
    """Use Gemini to generate a concise market summary from the headlines."""
    if gemini_model is None:
        pos = stats['percentages']['Positive']
        neg = stats['percentages']['Negative']
        neu = stats['percentages']['Neutral']
        verdict = stats['verdict']
        
        summary = (
            f"[MOCK SUMMARY - Add GEMINI_API_KEY to run live] "
            f"Based on our sentiment analysis of recent headlines for {company}, the market signal is leaning {verdict}. "
            f"Approximately {pos}% of coverage is positive, driven by optimistic outlooks and strong performance indicators, "
            f"while {neg}% is negative, reflecting concerns over industry headwinds, regulatory risks, or competition. "
            f"Neutral coverage sits at {neu}%. Overall, {company} exhibits stable market interest, with analysts focusing closely on "
            f"upcoming product rollouts and macroeconomic factors."
        )
        return summary

    headlines_text = "\n".join([
        f"[{h['sentiment']} - {h['confidence']}%] {h['title']}"
        for h in enriched[:10]
    ])

    prompt = f"""You are a financial analyst. Based on the following recent news headlines about {company}, 
write a concise 3-4 sentence market summary. Mention the overall sentiment trend, 
key themes you observe, and any notable events. Be factual and professional.

Headlines:
{headlines_text}

Overall sentiment: {stats['percentages']['Positive']}% Positive, 
{stats['percentages']['Negative']}% Negative, {stats['percentages']['Neutral']}% Neutral.

Market Summary:"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        pos = stats['percentages']['Positive']
        neg = stats['percentages']['Negative']
        neu = stats['percentages']['Neutral']
        verdict = stats['verdict']
        
        summary = (
            f"[LIVE AI SUMMARY UNAVAILABLE - API Error: {str(e)}] "
            f"Based on our sentiment analysis of recent headlines for {company}, the market signal is leaning {verdict}. "
            f"Approximately {pos}% of coverage is positive, driven by optimistic outlooks and strong performance indicators, "
            f"while {neg}% is negative, reflecting concerns over industry headwinds, regulatory risks, or competition. "
            f"Neutral coverage sits at {neu}%. Overall, {company} exhibits stable market interest, with analysts focusing closely on "
            f"upcoming product rollouts and macroeconomic factors."
        )
        return summary


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    company = data.get("company", "").strip()

    if not company:
        return jsonify({"error": "Please enter a company name or ticker."}), 400

    if len(company) > 50:
        return jsonify({"error": "Input too long. Please enter a company name or ticker."}), 400

    # 1. Fetch headlines
    headlines, error = fetch_headlines(company)
    if error:
        return jsonify({"error": error}), 400

    # 2. Run FinBERT sentiment analysis
    enriched = analyze_sentiment(headlines)

    # 3. Compute aggregate statistics
    stats = compute_stats(enriched)

    # 4. Generate Gemini market summary
    summary = generate_summary(company, enriched, stats)

    return jsonify({
        "company": company,
        "headlines": enriched,
        "stats": stats,
        "summary": summary,
        "analyzed_at": datetime.now().strftime("%B %d, %Y at %I:%M %p")
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "ProsusAI/finbert"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
