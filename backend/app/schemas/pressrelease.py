from pydantic import BaseModel

# BaseModelを継承したプレスリリース用スキーマ
class PressReleaseItem(BaseModel):
    date: str