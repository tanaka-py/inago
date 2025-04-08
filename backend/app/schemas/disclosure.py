from pydantic import BaseModel

# BaseModelを継承した開示学習用スキーマ
class LearningItem(BaseModel):
    date: str
    work_load: bool = False # デフォルトは読み込まない
    