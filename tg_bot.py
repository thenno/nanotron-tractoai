import requests
import yt.wrapper as yt
import time

import os


tg_sec = os.environ["YT_SECURE_VAULT_TG_SEC"]

PERSISTANT_CHAT_IDS = [
    103324766,
    183144474,
]
OPERATION_INFO_DOCUMENT = "//home/chiffa/training_operation_info"
BASE_URL = f"https://api.telegram.org/bot{tg_sec}"
ITER_TIMEOUT = 30


def get_chat_ids() -> set:
    response = requests.get(
        f"{BASE_URL}/getUpdates",
    )
    response.raise_for_status()
    chat_ids = set()
    for message in response.json()["result"]:
        chat_id = message["message"]["chat"]["id"]
        chat_ids.add(chat_id)
    return chat_ids


def get_operation_id(path: str) -> str:
    content = yt.get(path)
    return str(content["operation_id"])


def is_op_valid(op_id):
    op_state = yt.get_operation_state(op_id)
    return not op_state.is_unsuccessfully_finished()


def send_uvaga_message(chat_ids: set, message: str):
    url = f"{BASE_URL}/sendMessage"
    for chat_id in chat_ids:
        data = {
            "chat_id": chat_id,
            "text": message,
        }
        response = requests.post(url, data=data)
        response.raise_for_status()


def make_message(operation_id: str):
    return f"UVAGA! Training https://charlie.yt.nebius.yt/charlie/operations/{operation_id}/details has been failed"


def save_chat_ids(chat_ids: set):
    with open("chat_ids", "w") as f:
        f.write(",".join(map(str, chat_ids)))


def save_error(e):
    with open("errors", "w") as f:
        f.write(f"{time.time()} {e}")


def main():
    chat_ids = set(PERSISTANT_CHAT_IDS)
    while True:
        try:
            new_chat_ids = get_chat_ids()
            chat_ids.update(new_chat_ids)
            save_chat_ids(chat_ids)
            operation_id = get_operation_id(OPERATION_INFO_DOCUMENT)
            is_operation_valid = is_op_valid(operation_id)
            if not is_operation_valid:
                message = make_message(operation_id)
                send_uvaga_message(chat_ids, message)
        except Exception as e:
            print("there is an error", e)
            save_error(e)
        finally:
            time.sleep(ITER_TIMEOUT)
