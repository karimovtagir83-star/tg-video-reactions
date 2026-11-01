from pathlib import Path
from telethon import TelegramClient, events
from telethon.tl.types import Channel, User

ROOT=Path(__file__).resolve().parent.parent

def client(config): return TelegramClient(str(ROOT/"data"/"user"),config.api_id,config.api_hash,catch_up=False)

def group_dialog(d):
    e = d.entity
    return bool(getattr(d, "is_group", False) and not getattr(e, "broadcast", False))

async def list_groups(c):
    return [(d.name,d.id) async for d in c.iter_dialogs() if group_dialog(d)]

def event_urls(event):
    return event.message.message or "", getattr(event.message,"entities",[]) or []
