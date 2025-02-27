import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API Keys
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")

print("Kraken API Key:", KRAKEN_API_KEY)
print("Kraken API Secret:", KRAKEN_API_SECRET)
