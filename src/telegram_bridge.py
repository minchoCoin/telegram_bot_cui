import asyncio
from telethon import TelegramClient, events

from config import (
    API_ID,
    API_HASH,
    BOT_USERNAME,
    TELEGRAM_IDLE_SECONDS,
    TELEGRAM_TIMEOUT,
)


class TelegramReplyCollector:
    def __init__(self, client: TelegramClient, target: str):
        self.client = client
        self.target = target
        self.done_event = asyncio.Event()
        self.silence_task = None
        self.messages = []

    async def reset_silence_timer(self):
        if self.silence_task and not self.silence_task.done():
            self.silence_task.cancel()

        async def waiter():
            try:
                await asyncio.sleep(TELEGRAM_IDLE_SECONDS)
                self.done_event.set()
            except asyncio.CancelledError:
                pass

        self.silence_task = asyncio.create_task(waiter())

    async def start_and_send(self, text: str) -> str:
        @self.client.on(events.NewMessage(from_users=self.target))
        async def on_new(event):
            self.messages.append(event.raw_text)
            print("[NEW]", event.raw_text)
            await self.reset_silence_timer()

        @self.client.on(events.MessageEdited(from_users=self.target))
        async def on_edit(event):
            self.messages.append(event.raw_text)
            print("[EDIT]", event.raw_text)
            await self.reset_silence_timer()

        await self.client.start()
        entity = await self.client.get_entity(self.target)

        await self.client.send_message(entity, text)
        print("Message sent.")

        await asyncio.wait_for(self.done_event.wait(), timeout=TELEGRAM_TIMEOUT)

        if not self.messages:
            raise RuntimeError("Failed to receive Telegram reply.")

        return self.messages[-1]


async def send_to_telegram_and_get_reply(text: str) -> str:
    if not API_ID or not API_HASH or not BOT_USERNAME:
        raise RuntimeError(".env Telegram settings (api_id, api_hash, bot_username) are not configured.")

    client = TelegramClient("pi_session", API_ID, API_HASH)
    collector = TelegramReplyCollector(client, BOT_USERNAME)

    async with client:
        reply = await collector.start_and_send(text)
        return reply