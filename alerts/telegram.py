"""
Telegram alert sender — mirrors the Hawala v2 pattern.

Setup:
  1. Open Telegram → @BotFather → /newbot → copy the token
  2. Message your bot once, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Copy the "id" value from the "chat" block → TELEGRAM_CHAT_IDS in .env
"""

import requests


def send(token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[telegram] send failed for {chat_id}: {e}")
        return False


def send_document(token: str, chat_id: str, file_path: str, caption: str = '') -> bool:
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": f},
                timeout=30,
            )
        return resp.status_code == 200
    except Exception as e:
        print(f"[telegram] send_document failed for {chat_id}: {e}")
        return False


def broadcast(token: str, chat_ids: list[str], text: str) -> None:
    for cid in chat_ids:
        send(token, cid, text)
