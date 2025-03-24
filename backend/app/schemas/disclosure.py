from pydantic import BaseModel

# BaseModelを継承した開示学習用スキーマ
class LearningItem(BaseModel):
    date: str