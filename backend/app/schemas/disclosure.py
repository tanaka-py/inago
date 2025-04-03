from pydantic import BaseModel

# BaseModelを継承した開示学習用スキーマ
class LearningItem(BaseModel):
    date: str
    mode: int = 0 # 0:全件取得がデフォルト 1:決算のみ 2:決算以外
    work_load: bool = False # デフォルトは読み込まない
    