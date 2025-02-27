import requests
import json
import time
import hashlib
import hmac
import base64
from flask import Flask, render_template, request, jsonify
import threading

app = Flask(__name__)

# ✅ Betfair API Configuration
BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketCatalogue/"
BETFAIR_BET_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/placeOrders/"
BETFAIR_API_KEY = "your_betfair_api_key"
SESSION_TOKEN = "your_betfair_session_token"

# ✅ Kraken API Configuration
KRAKEN_API_KEY = "your_kraken_api_key"
KRAKEN_API_SECRET = "your_kraken_api_secret"
KRAKEN_TRADE_URL = "https://api.kraken.com/0/private/AddOrder"

# ✅ Fetch Horse Racing Data from Betfair
def fetch_horse_racing_data():
    headers = {
        "X-Application": BETFAIR_API_KEY,
        "X-Authentication": SESSION_TOKEN,
        "Content-Type": "application/json"
    }

    params = {
        "filter": {
            "eventTypeIds": ["7"],  # Horse Racing
            "marketCountries": ["GB"],  # UK Races
            "marketTypeCodes": ["WIN"],  # Win Markets
            "maxResults": "5"
        }
    }

    response = requests.post(BETFAIR_API_URL, headers=headers, json=params)

    if response.status_code == 200:
        try:
            data = response.json()
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    
    return []

# ✅ Place a Bet on Betfair
def place_bet(market_id, selection_id, stake, price):
    headers = {
        "X-Application": BETFAIR_API_KEY,
        "X-Authentication": SESSION_TOKEN,
        "Content-Type": "application/json"
    }

    payload = {
        "marketId": market_id,
        "instructions": [
            {
                "selectionId": selection_id,
                "side": "BACK",
                "orderType": "LIMIT",
                "limitOrder": {
                    "size": stake,
                    "price": price,
                    "persistenceType": "LAPSE"
                }
            }
        ]
    }

    response = requests.post(BETFAIR_BET_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to place bet"}

# ✅ Fetch Crypto Price from Kraken
def fetch_crypto_price():
    response = requests.get("https://api.kraken.com/0/public/Ticker?pair=XBTGBP")
    if response.status_code == 200:
        try:
            data = response.json()
            btc_price = data["result"]["XXBTZGBP"]["c"][0]  # Last trade price
            return {"BTC_GBP": btc_price}
        except KeyError:
            return {"error": "Invalid response from Kraken API"}
    return {"error": "Failed to fetch crypto data"}

# ✅ Place a Trade on Kraken
def place_trade(pair, trade_type, price, volume):
    nonce = str(int(time.time() * 1000))
    data = {
        "nonce": nonce,
        "ordertype": "limit",
        "type": trade_type,  # "buy" or "sell"
        "price": str(price),
        "volume": str(volume),
        "pair": pair
    }

    uri_path = "/0/private/AddOrder"
    post_data = "&".join([f"{key}={value}" for key, value in data.items()])
    message = uri_path.encode() + hashlib.sha256(nonce.encode() + post_data.encode()).digest()
    signature = hmac.new(base64.b64decode(KRAKEN_API_SECRET), message, hashlib.sha512).digest()
    api_signature = base64.b64encode(signature).decode()

    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": api_signature
    }

    response = requests.post(KRAKEN_TRADE_URL, headers=headers, data=data)
    return response.json()

# ✅ Flask Routes
@app.route('/')
def dashboard():
    horse_racing_data = fetch_horse_racing_data()
    crypto_data = fetch_crypto_price()
    return render_template('dashboard.html', horse_racing=horse_racing_data, crypto=crypto_data)

@app.route('/place_bet', methods=['POST'])
def betting():
    data = request.json
    market_id = data['market_id']
    selection_id = data['selection_id']
    stake = data['stake']
    price = data['price']
    result = place_bet(market_id, selection_id, stake, price)
    return jsonify(result)

@app.route('/place_trade', methods=['POST'])
def trading():
    data = request.json
    trade_type = data['trade_type']
    price = data['price']
    volume = data['volume']
    result = place_trade("XBTGBP", trade_type, price, volume)
    return jsonify(result)

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
