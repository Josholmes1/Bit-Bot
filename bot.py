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
from datetime import datetime

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
# Replace these with actual channel IDs from your Discord server
CRYPTO_PRICES_CHANNEL_ID = 123456789012345678  # e.g., #crypto-prices
TRADE_ALERTS_CHANNEL_ID = 123456789012345678   # e.g., #trade-alerts
TWITTER_UPDATES_CHANNEL_ID = 123456789012345678  # e.g., #twitter-updates

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

price_data = {}

async def update_live_prices():
    while True:
        for pair in ["BTCGBP", "XRPGBP", "DOGEUSD"]:
            try:
                url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
                response = requests.get(url)
                result = response.json().get("result", {})
                key = list(result.keys())[0] if result else None
                if key:
                    price_data[pair] = float(result[key]['c'][0])
            except Exception as e:
                print(f"Error fetching price for {pair}: {e}")
        await asyncio.sleep(30)  # Update every 30 seconds

class AITrading:
    def __init__(self):
        self.feature_count = 6
        self.model = self.build_model()  # Price, Sentiment, RSI, EMA, Volume, MACD
        self.scaler = MinMaxScaler()
        self.historical_prices = {"BTCGBP": [], "XRPGBP": [], "DOGEUSD": []}
        self.sentiment_scores = []
        self.positions = {}
        self.trading_active = False
        self.challenge_mode = False
        self.challenge_start_balance = 100
        self.challenge_balance = 100
        self.challenge_history = []
        self.last_trade_time = {}
        self.cooldown_seconds = 180
        self.csv_file = "trade_history.csv"
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Pair", "Side", "Price", "Size", "Simulated Balance"])

    def build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(50, self.feature_count)),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    async def send_twitter_updates(self):
        while True:
            headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
            url = "https://api.twitter.com/2/tweets/search/recent?query=crypto&max_results=5"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                tweets = [tweet['text'] for tweet in response.json().get("data", [])]
                twitter_channel = bot.get_channel(TWITTER_UPDATES_CHANNEL_ID)
                if twitter_channel:
                    for tweet in tweets:
                        await twitter_channel.send(f"🐦 **Crypto Twitter Update:** {tweet}")

            await asyncio.sleep(180)

    def analyze_sentiment(self, tweets):
        sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        self.sentiment_scores.append(avg_sentiment)

    def can_trade(self, pair):
        now = time.time()
        return pair not in self.last_trade_time or now - self.last_trade_time[pair] > self.cooldown_seconds

    def update_challenge_balance(self, pair, price, side, size):
        if self.challenge_mode:
            if side == "buy":
                self.challenge_balance -= size * price
            else:
                self.challenge_balance += size * price
            self.challenge_history.append(f"{side.upper()} {pair} at {price:.2f} | Sim Bal: £{self.challenge_balance:.2f}")
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pair, side.upper(), f"{price:.2f}", size, f"{self.challenge_balance:.2f}"])

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
                    sentiment = self.sentiment_scores[-1] if self.sentiment_scores else 0
                    price_list = self.historical_prices[pair][-60:]
                    sentiment_list = [sentiment] * len(price_list)
                    rsi_series = pd.Series(price_list).diff()
                    gain = rsi_series.where(rsi_series > 0, 0.0)
                    loss = -rsi_series.where(rsi_series < 0, 0.0)
                    avg_gain = gain.rolling(window=14, min_periods=14).mean()
                    avg_loss = loss.rolling(window=14, min_periods=14).mean()
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_filled = rsi.fillna(50).tolist()

                    # EMA (Exponential Moving Average)
                    ema = pd.Series(price_list).ewm(span=10, adjust=False).mean().fillna(method='bfill').tolist()

                    # MACD (Moving Average Convergence Divergence)
                    ema_12 = pd.Series(price_list).ewm(span=12, adjust=False).mean()
                    ema_26 = pd.Series(price_list).ewm(span=26, adjust=False).mean()
                    macd_line = ema_12 - ema_26
                    macd = macd_line.fillna(0).tolist()

                    # Real volume from Kraken (placeholder logic)
                    volume = []
                    try:
                        for symbol in [pair]:
                            url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval=1"
                            response = requests.get(url)
                            ohlc = response.json().get("result")
                            if ohlc:
                                key = next(iter(ohlc))
                                volume_data = [float(candle[6]) for candle in ohlc[key][-60:]]
                                volume = volume_data if len(volume_data) >= 60 else [np.mean(volume_data)] * 60
                            else:
                                volume = [1000] * 60
                    except Exception as e:
                        print(f"Volume fetch error: {e}")
                        volume = [1000] * 60
    
                    combined = np.array([[p, sentiment_list[i], rsi_filled[i], ema[i], volume[i], macd[i]] for i, p in enumerate(price_list)])
                    scaled = self.scaler.fit_transform(combined)
                    X = np.array([scaled[i:i+50] for i in range(10)])
                    y = np.array([scaled[i+50][0] for i in range(10)])

                    self.model.fit(X, y, epochs=3, batch_size=1, verbose=0)
                    prediction = self.model.predict(X[-1].reshape(1, 50, self.feature_count))[0][0]
                    predicted_price = self.scaler.inverse_transform([[prediction, sentiment, rsi_filled[-1], ema[-1], volume[-1], macd[-1]]])[0][0]

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

                    if not self.can_trade(pair):
                        continue

                    balance = get_available_balance()
                    base_asset = pair.replace("GBP", "").replace("USD", "")
                    trade_size = float(balance.get(base_asset, 0)) * 0.2

                    if predicted_price > price and pair not in self.positions:
                        place_trade(pair, "buy", trade_size)
                        self.positions[pair] = {"buy_price": price, "size": trade_size}
                        self.last_trade_time[pair] = time.time()
                        self.update_challenge_balance(pair, price, "buy", trade_size)
                        await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"📈 Bought {trade_size} of {pair} at {price:.2f}")
                    elif predicted_price < price and pair in self.positions:
                        place_trade(pair, "sell", self.positions[pair]["size"])
                        self.update_challenge_balance(pair, price, "sell", self.positions[pair]["size"])
                        self.last_trade_time[pair] = time.time()
                        await log_to_discord(TRADE_ALERTS_CHANNEL_ID, f"📉 Sold {self.positions[pair]['size']} of {pair} at {price:.2f}")
                        del self.positions[pair]

            await asyncio.sleep(60)

ai_trader = AITrading()

@bot.event
async def on_ready():
    asyncio.create_task(update_live_prices())
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
    await ctx.send("🏁 Challenge Mode activated! Bot will now try to grow a simulated £100.")

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
