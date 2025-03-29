import requests 
import json
import time
import hashlib
import hmac
import base64
import threading
import os
import pandas as pd
import numpy as np
import websockets
import asyncio
import tweepy
import csv
from dotenv import load_dotenv
from flask import Flask
import discord
from discord.ext import commands, tasks
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pygame
from urllib.parse import urlencode
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from textblob import TextBlob

# ✅ Load Environment Variables Correctly
load_dotenv()

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# ✅ Flask Setup
app = Flask(__name__)

@app.route("/")
def home():
    return "Flask server is running!"

# ✅ Initialize pygame mixer for sound
pygame.mixer.init()

# ✅ Kraken API Configuration
KRAKEN_TRADE_URL = "https://api.kraken.com/0/private/AddOrder"
KRAKEN_BALANCE_URL = "https://api.kraken.com/0/private/Balance"
KRAKEN_COPY_TRADERS_URL = "https://api.kraken.com/0/public/Trades"

# ✅ Define Discord Channel IDs
CRYPTO_PRICES_CHANNEL_ID = 123456789012345678
TRADE_ALERTS_CHANNEL_ID = 123456789012345678
TWITTER_UPDATES_CHANNEL_ID = 123456789012345678

# ✅ Kraken Signature Function
def get_kraken_signature(url_path, data, secret):
    post_data = urlencode(data)
    encoded = (data['nonce'] + post_data).encode()
    message = url_path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

# ✅ Discord Bot Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ✅ Placeholder: Replace with your own implementation
def get_available_balance():
    return {"BTC": 0.1, "XRP": 500, "DOGE": 1000}

def place_trade(pair, side, volume):
    print(f"Simulated {side.upper()} trade for {volume} of {pair}")

async def log_to_discord(channel_id, message):
    channel = bot.get_channel(channel_id)
    if channel:
        await channel.send(message)

price_data = {"BTCGBP": 30000, "XRPGBP": 0.5, "DOGEUSD": 0.1}  # Example

class AITrading:
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.historical_prices = {"BTCGBP": [], "XRPGBP": [], "DOGEUSD": []}
        self.sentiment_scores = []
        self.positions = {}
        self.trading_active = False
        self.challenge_mode = False
        self.challenge_start_balance = 100
        self.challenge_balance = 100
        self.challenge_history = []

    def build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(50, 1)),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def analyze_sentiment(self, tweets):
        sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        self.sentiment_scores.append(avg_sentiment)

    async def send_twitter_updates(self):
        while True:
            headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
            url = f"https://api.twitter.com/2/tweets/search/recent?query=crypto&max_results=5"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                tweets = [tweet['text'] for tweet in response.json().get("data", [])]
                twitter_channel = bot.get_channel(TWITTER_UPDATES_CHANNEL_ID)
                if twitter_channel:
                    for tweet in tweets:
                        await twitter_channel.send(f"🐦 **Crypto Twitter Update:** {tweet}")

            await asyncio.sleep(180)

    def update_challenge_balance(self, pair, price, side, size):
        if self.challenge_mode:
            if side == "buy":
                self.challenge_balance -= size * price
            else:
                self.challenge_balance += size * price
            self.challenge_history.append(f"{side.upper()} {pair} at {price:.2f} | Sim Bal: £{self.challenge_balance:.2f}")

    async def trading_logic(self):
        while True:
            if not self.trading_active:
                await asyncio.sleep(5)
                continue

            for pair in ["BTCGBP", "XRPGBP", "DOGEUSD"]:
                if pair not in price_data:
                    continue

                price = price_data[pair]
                self.historical_prices[pair].append(price)

                if len(self.historical_prices[pair]) > 60:
                    prices_array = np.array(self.historical_prices[pair][-60:]).reshape(-1, 1)
                    scaled_prices = self.scaler.fit_transform(prices_array)
                    X = np.array([scaled_prices[i:i+50] for i in range(10)])
                    y = np.array([scaled_prices[i+50] for i in range(10)])

                    self.model.fit(X, y, epochs=3, batch_size=1, verbose=0)
                    prediction = self.model.predict(X[-1].reshape(1, 50, 1))[0][0]
                    predicted_price = self.scaler.inverse_transform([[prediction]])[0][0]

                    if self.sentiment_scores and self.sentiment_scores[-1] > 0:
                        predicted_price *= 1.01

                    if pair in self.positions:
                        buy_price = self.positions[pair]["buy_price"]
                        if price < buy_price * 0.98:
                            await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"❗ Stop-loss triggered on {pair}")
                            place_trade(pair, "sell", self.positions[pair]["size"])
                            self.update_challenge_balance(pair, price, "sell", self.positions[pair]["size"])
                            del self.positions[pair]
                            continue
                        elif price > buy_price * 1.05:
                            await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"💰 Profit target hit on {pair}")
                            place_trade(pair, "sell", self.positions[pair]["size"])
                            self.update_challenge_balance(pair, price, "sell", self.positions[pair]["size"])
                            del self.positions[pair]
                            continue

                    balance = get_available_balance()
                    base_asset = pair.replace("GBP", "").replace("USD", "")
                    trade_size = float(balance.get(base_asset, 0)) * 0.2

                    if predicted_price > price and pair not in self.positions:
                        place_trade(pair, "buy", trade_size)
                        self.positions[pair] = {"buy_price": price, "size": trade_size}
                        self.update_challenge_balance(pair, price, "buy", trade_size)
                        await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"📈 Bought {trade_size} of {pair} at {price:.2f}")
                    elif predicted_price < price and pair in self.positions:
                        place_trade(pair, "sell", self.positions[pair]["size"])
                        self.update_challenge_balance(pair, price, "sell", self.positions[pair]["size"])
                        await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"📉 Sold {self.positions[pair]['size']} of {pair} at {price:.2f}")
                        del self.positions[pair]

            await asyncio.sleep(60)

ai_trader = AITrading()

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    asyncio.create_task(ai_trader.trading_logic())
    asyncio.create_task(ai_trader.send_twitter_updates())

@bot.command()
async def start_trading(ctx):
    ai_trader.trading_active = True
    await ctx.send("🟢 Trading started.")

@bot.command()
async def stop_trading(ctx):
    ai_trader.trading_active = False
    await ctx.send("🔴 Trading stopped.")

@bot.command()
async def trading_stats(ctx):
    stats = "\n".join([f"{pair}: {price_data.get(pair, 'N/A')}" for pair in ["BTCGBP", "XRPGBP", "DOGEUSD"]])
    sentiment = ai_trader.sentiment_scores[-1] if ai_trader.sentiment_scores else 'N/A'
    await ctx.send(f"📊 **Current Prices:**\n{stats}\n\n📈 **Sentiment Score:** {sentiment}")

@bot.command()
async def start_challenge(ctx):
    ai_trader.challenge_mode = True
    ai_trader.challenge_balance = ai_trader.challenge_start_balance
    ai_trader.challenge_history.clear()
    await ctx.send("🏁 Challenge Mode activated! Bot will now try to grow a simulated £1000.")

@bot.command()
async def stop_challenge(ctx):
    ai_trader.challenge_mode = False
    await ctx.send("🛑 Challenge Mode stopped.")

@bot.command()
async def challenge_status(ctx):
    if not ai_trader.challenge_mode:
        await ctx.send("⚠️ Challenge Mode is not active.")
        return
    log = "\n".join(ai_trader.challenge_history[-5:]) or "No trades yet."
    await ctx.send(f"📈 **Challenge Balance:** £{ai_trader.challenge_balance:.2f}\n📘 Last Trades:\n{log}")

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"debug": True, "use_reloader": False}).start()
    bot.run(DISCORD_BOT_TOKEN)
