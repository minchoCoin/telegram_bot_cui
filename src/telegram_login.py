from telethon import TelegramClient
from dotenv import load_dotenv
import os
load_dotenv()

api_id = int(os.getenv("api_id"))
api_hash = os.getenv("api_hash")
bot_username = os.getenv("bot_username")

client = TelegramClient("pi_session", api_id, api_hash)

async def main():
    await client.start() 
    print("login successful")

with client:
    client.loop.run_until_complete(main())