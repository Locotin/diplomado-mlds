from uuid import uuid4


def generate_source_id() -> str:
    return f"src-{uuid4().hex}"
