# 🧸 Mattel Singapore — Telegram Customer Support Bot

A LangGraph multi-agent Telegram bot for Mattel Singapore customer support.

**Agents:**
- 🛍️ Product Advisor — toy recommendations, age suitability, gift ideas
- 🎁 Promotions & Deals — sales events, vouchers, bundles
- 🔄 After-Sales & Returns — warranty, returns, defective products

**Stack:** Python · python-telegram-bot · LangGraph · Groq (LLaMA 3.3) · DuckDuckGo

---

## 🚀 Deploy to Railway (Step by Step)

### Step 1 — Create your Telegram Bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name: e.g. `Mattel Singapore Support`
4. Choose a username: e.g. `MattelSGSupportBot`
5. Copy the **bot token** (looks like `123456:ABC-DEF...`)

### Step 2 — Get your Groq API key
1. Go to https://console.groq.com/keys
2. Sign up / log in
3. Click **Create API Key**
4. Copy the key

### Step 3 — Push code to GitHub
1. Create a new GitHub repository (can be private)
2. Upload all files in this folder to the repo:
   - `bot.py`
   - `knowledge.py`
   - `requirements.txt`
   - `Procfile`

### Step 4 — Deploy on Railway
1. Go to https://railway.app and sign up (free)
2. Click **New Project → Deploy from GitHub repo**
3. Select your repository
4. Go to **Variables** tab and add:
   ```
   TELEGRAM_BOT_TOKEN = your_telegram_bot_token
   GROQ_API_KEY       = your_groq_api_key
   ```
5. Railway will automatically detect the Procfile and deploy
6. Wait 2-3 minutes for build to complete

### Step 5 — Test your bot
1. Open Telegram
2. Search for your bot username
3. Send `/start`
4. Ask a question!

---

## 💬 Example Questions

| Type | Example |
|---|---|
| Product | What Barbie should I get for a 5-year-old? |
| Product | Best Hot Wheels gift under S$50? |
| Promotions | When is the 11.11 sale for Mattel toys? |
| Promotions | How do I get the best deal on Barbie Dreamhouse? |
| After-Sales | My Fisher-Price toy arrived broken. What do I do? |
| After-Sales | How do I return a toy bought on Lazada? |

---

## 📁 File Structure

```
mattel-bot/
├── bot.py          ← Main bot + LangGraph agents
├── knowledge.py    ← Embedded knowledge bases (no DB needed)
├── requirements.txt
├── Procfile        ← Railway start command
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather on Telegram |
| `GROQ_API_KEY` | From console.groq.com/keys |
| `GROQ_MODEL` | Optional — defaults to `llama-3.3-70b-versatile` |

---

## 🔄 How It Works

```
User message → Telegram
    → Router (1 LLM call — classifies intent)
    → Agent node:
        1. RAG search (keyword match on embedded KB)
        2. DuckDuckGo (only for live/current queries)
        3. Single LLM call (generates response)
    → Reply sent to Telegram
```

Total: 2 LLM calls per message. Fast and quota-efficient.
