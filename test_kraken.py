import requests

url = "https://api.kraken.com/0/public/AssetPairs"
response = requests.get(url).json()

for pair in response["result"]:
    if "XRP" in pair and "GBP" in pair:
        print(f"Kraken Pair Found: {pair}")
