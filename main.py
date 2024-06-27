import os
import logging

from telegram import Update
from telegram.ext import (
    CommandHandler,
    CallbackContext,
    ApplicationBuilder,
    MessageHandler,
    filters,
)
from load_dotenv import load_dotenv
from utils.rag_utils import RAGPipeline

# Load the .env file
load_dotenv(".env")

# Define the bot token
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Create an instance of the RAGPipeline
rag_pipeline = RAGPipeline()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Hello! I am an AI bot. Ask me anything about the legal information."
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Type your question and I will try to answer based on the data I have."
    )


async def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text

    response = rag_pipeline.run(user_message)

    await update.message.reply_text(response)


def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
