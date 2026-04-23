"""
Mattel Singapore — Telegram Customer Support Bot
Powered by LangGraph + Groq + DuckDuckGo
Hosted on Railway
"""

import os
import base64
import logging
from typing import TypedDict

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

from knowledge import PRODUCT_KB, PROMOTIONS_KB, AFTERSALES_KB

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Environment variables ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
GROQ_MODEL         = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Initialise LLM + DuckDuckGo ───────────────────────────────────────────────
llm        = ChatGroq(model=GROQ_MODEL,   api_key=GROQ_API_KEY, temperature=0.2)
vision_llm = ChatGroq(model=VISION_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
ddg = DuckDuckGoSearchRun()

# ── Keywords that trigger a live web search ───────────────────────────────────
LIVE_KEYWORDS = [
    "latest", "new", "current", "today", "now", "2024", "2025", "2026",
    "just released", "recently", "this week", "promo code", "voucher code",
    "compare", "vs", "versus", "review", "best price", "cheapest"
]

def needs_web_search(query: str) -> bool:
    return any(kw in query.lower() for kw in LIVE_KEYWORDS)

def web_search(query: str) -> str:
    try:
        result = ddg.run(f"Mattel Singapore {query}")
        return result if result else ""
    except Exception:
        return ""

# ── Simple keyword-based RAG over embedded knowledge base ────────────────────
def rag_search(kb_text: str, query: str, top_chars: int = 3000) -> str:
    """
    Split the knowledge base into chunks and return the most relevant ones
    based on keyword overlap with the query. No vector DB needed.
    """
    query_words = set(query.lower().split())
    # Remove common stop words
    stop = {"i", "a", "the", "is", "it", "my", "for", "to", "do", "how",
            "what", "can", "me", "in", "of", "and", "on", "at", "with"}
    query_words -= stop

    # Split KB into paragraphs
    paragraphs = [p.strip() for p in kb_text.split('\n\n') if p.strip()]

    # Score each paragraph by keyword overlap
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for w in query_words if w in para_lower)
        if score > 0:
            scored.append((score, para))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top matching paragraphs up to top_chars
    result = []
    total = 0
    for _, para in scored:
        if total + len(para) > top_chars:
            break
        result.append(para)
        total += len(para)

    return "\n\n".join(result) if result else ""

# ── Agent runner ──────────────────────────────────────────────────────────────
def run_agent(system_prompt: str, kb_text: str, query: str) -> str:
    """RAG + optional DuckDuckGo + single LLM call."""
    rag_context  = rag_search(kb_text, query)
    web_context  = web_search(query) if needs_web_search(query) else ""

    context_parts = []
    if rag_context:
        context_parts.append(f"--- INTERNAL KNOWLEDGE BASE ---\n{rag_context}")
    if web_context:
        context_parts.append(f"--- LIVE WEB RESULTS ---\n{web_context}")

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    full_prompt = (
        f"{system_prompt}\n\n"
        f"{context}\n\n"
        f"--- END CONTEXT ---\n\n"
        f"Customer query: {query}\n\n"
        f"Answer using the context above. Be concise — this is a Telegram message, "
        f"so keep your response under 300 words. Use simple formatting (no markdown tables). "
        f"If context does not cover the query, say so politely."
    )

    response = llm.invoke([HumanMessage(content=full_prompt)])
    return response.content

# ── Agent system prompts ──────────────────────────────────────────────────────
PRODUCT_PROMPT = (
    "You are the Mattel Singapore Product Advisor. "
    "Help customers find the right Mattel toy by age, interests, budget, and occasion. "
    "Mattel brands: Barbie, Hot Wheels, Fisher-Price, UNO. "
    "Always include a price range when recommending. Be warm and concise. "
    "If asked about promotions say: Our Promotions Agent can help with that! "
    "If asked about returns say: Our After-Sales Agent is the right contact for that!"
)

PROMOTIONS_PROMPT = (
    "You are the Mattel Singapore Promotions and Deals Agent. "
    "Help customers get the best value when buying Mattel products in Singapore. "
    "Cover: sale events (9.9, 11.11, Children Day, GSS, Christmas), "
    "retailer promotions (Toys R Us, Shopee, Lazada, Amazon), loyalty programmes, bundles. "
    "Be enthusiastic and mention multiple platforms where relevant. "
    "If asked about product specs say: Our Product Advisor can help with that! "
    "If asked about returns say: Our After-Sales Agent is the right contact for that!"
)

AFTERSALES_PROMPT = (
    "You are the Mattel Singapore After-Sales and Returns Agent. "
    "Help customers resolve post-purchase issues: returns, refunds, warranty, "
    "missing parts, defective products, and product safety recalls. "
    "Return windows: Toys R Us 14 days, Shopee 7 days, Lazada 7-15 days, Amazon 30 days. "
    "Mattel warranty is 90 days. Claims at https://service.mattel.com. "
    "Be empathetic. Always give clear step-by-step instructions. "
    "If asked about product recommendations say: Our Product Advisor can help with that! "
    "If asked about promotions say: Our Promotions Agent can help with that!"
)

# ── LangGraph setup ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    response: str
    next_node: str
    debug_log: str

VALID_ROUTES = ["product_agent", "promotions_agent", "aftersales_agent"]

def router_node(state):
    query = state["query"]
    raw = ""
    routing_prompt = (
        f"You are a routing classifier. Reply with EXACTLY one label, nothing else.\n\n"
        f"Labels:\n"
        f"product_agent — toy recommendations, product specs, Barbie, Hot Wheels, "
        f"Fisher-Price, UNO, age suitability, gift ideas\n"
        f"promotions_agent — sales, discounts, promotions, vouchers, bundle deals, "
        f"best time to buy, 11.11, GSS, Children Day, loyalty programmes\n"
        f"aftersales_agent — returns, refunds, warranty, defective products, "
        f"missing parts, recalls, broken toy, exchange\n\n"
        f"Customer query: {query}\n\n"
        f"Reply with only one of these three words:\n"
        f"product_agent\npromotions_agent\naftersales_agent"
    )
    try:
        response = llm.invoke([HumanMessage(content=routing_prompt)])
        raw = response.content.strip().lower()
        decision = "product_agent"
        for route in VALID_ROUTES:
            if route in raw:
                decision = route
                break
    except Exception as e:
        logger.error(f"Router error: {e}")
        decision = "product_agent"

    return {"next_node": decision, "debug_log": f"Router → {decision}"}

def product_agent_node(state):
    response = run_agent(PRODUCT_PROMPT, PRODUCT_KB, state["query"])
    return {"response": response, "next_node": "product_agent",
            "debug_log": state.get("debug_log","") + " | 🛍️ Product Advisor"}

def promotions_agent_node(state):
    response = run_agent(PROMOTIONS_PROMPT, PROMOTIONS_KB, state["query"])
    return {"response": response, "next_node": "promotions_agent",
            "debug_log": state.get("debug_log","") + " | 🎁 Promotions & Deals"}

def aftersales_agent_node(state):
    response = run_agent(AFTERSALES_PROMPT, AFTERSALES_KB, state["query"])
    return {"response": response, "next_node": "aftersales_agent",
            "debug_log": state.get("debug_log","") + " | 🔄 After-Sales"}

workflow = StateGraph(AgentState)
workflow.add_node("router",           router_node)
workflow.add_node("product_agent",    product_agent_node)
workflow.add_node("promotions_agent", promotions_agent_node)
workflow.add_node("aftersales_agent", aftersales_agent_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda state: state["next_node"])
workflow.add_edge("product_agent",    END)
workflow.add_edge("promotions_agent", END)
workflow.add_edge("aftersales_agent", END)
app = workflow.compile()

logger.info("✅ LangGraph app compiled")

# ── Agent badge labels ────────────────────────────────────────────────────────
BADGE = {
    "product_agent":    "🛍️ Product Advisor",
    "promotions_agent": "🎁 Promotions & Deals",
    "aftersales_agent": "🔄 After-Sales & Returns",
}

# ── Telegram handlers ─────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *Mattel Singapore Customer Support*!\n\n"
        "I can help you with:\n"
        "🛍️ *Product advice* — find the right toy by age & budget\n"
        "🎁 *Promotions & deals* — sales, vouchers, bundles\n"
        "🔄 *After-sales support* — returns, warranty, defects\n"
        "📸 *Toy recognition* — send a photo and I'll identify it!\n\n"
        "Just type your question or send a photo to get started!",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Example questions you can ask:*\n\n"
        "🛍️ What Barbie should I get for a 5-year-old?\n"
        "🛍️ Best Hot Wheels gift under S$50?\n"
        "🎁 When is the 11.11 sale for Mattel toys?\n"
        "🎁 How do I get the best deal on Barbie Dreamhouse?\n"
        "🔄 My Fisher-Price toy arrived broken. What do I do?\n"
        "🔄 How do I return a toy bought on Lazada?\n"
        "📸 *Send a photo* of any Mattel toy and I'll identify it!\n\n"
        "Type any question or send a photo to get started!",
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text.strip()
    chat_id    = update.message.chat_id

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        result = app.invoke({
            "query":     user_query,
            "response":  "",
            "next_node": "",
            "debug_log": ""
        })

        response  = result.get("response", "Sorry, I could not generate a response. Please try again.")
        node      = result.get("next_node", "")
        badge     = BADGE.get(node, "🤖 Mattel Support")

        # Send response with agent badge footer
        full_reply = f"{response}\n\n_{badge}_"
        await update.message.reply_text(full_reply, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text(
            "⚠️ Sorry, something went wrong. Please try again in a moment."
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        # Download highest-resolution photo
        photo_file = await (await context.bot.get_file(update.message.photo[-1].file_id)).download_as_bytearray()
        image_b64  = base64.b64encode(bytes(photo_file)).decode("utf-8")

        # Ask vision model to identify the toy
        vision_response = vision_llm.invoke([
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        "You are a Mattel toy identification expert. "
                        "Examine this image and identify:\n"
                        "1. Toy brand (Barbie, Hot Wheels, Fisher-Price, UNO, Masters of the Universe, etc.)\n"
                        "2. Specific product name or line if visible\n"
                        "3. Approximate target age range\n"
                        "4. Any visible condition issues (for after-sales purposes)\n\n"
                        "Be specific but concise. If it is not a Mattel toy, say so clearly."
                    )
                }
            ])
        ])
        identification = vision_response.content.strip()

        # Include any caption the user sent with the photo
        caption    = (update.message.caption or "").strip()
        full_query = (
            f"Photo analysis: {identification}. Customer note: {caption}"
            if caption else
            f"Photo analysis: {identification}"
        )

        # Route through the normal multi-agent graph
        result = app.invoke({
            "query":     full_query,
            "response":  "",
            "next_node": "",
            "debug_log": ""
        })

        response = result.get("response", "Sorry, I could not generate a response.")
        node     = result.get("next_node", "")
        badge    = BADGE.get(node, "🤖 Mattel Support")

        full_reply = (
            f"📸 *Toy Identified:*\n{identification}\n\n"
            f"─────────────────\n\n"
            f"{response}\n\n"
            f"_{badge}_"
        )
        await update.message.reply_text(full_reply, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't process that image. "
            "Please try again or describe the toy in text."
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("help",  help_command))
    app_bot.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_bot.add_error_handler(error_handler)

    logger.info("🤖 Mattel Singapore Bot is running...")
    app_bot.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
