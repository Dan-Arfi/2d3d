from __future__ import annotations

import json
from typing import Any


def make_message(msg_type: str, **payload: Any) -> str:
    body = {"type": msg_type, **payload}
    return json.dumps(body)



def parse_message(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid message format: expected object")
    if "type" not in data:
        raise ValueError("Invalid message format: missing type")
    return data

