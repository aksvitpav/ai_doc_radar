from pydantic import BaseModel

class Status(BaseModel):
    ok: bool = True
