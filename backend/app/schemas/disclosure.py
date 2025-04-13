from pydantic import BaseModel

# BaseModelを継承した開示学習用スキーマ
class LearningItem(BaseModel):
    date: str
    work_load: bool = False # デフォルトは読み込まない
    
    
class StatItem(BaseModel):
    
    # 学習中日
    target_date: str
    
    # 学習中作業データの有無
    is_work_data: bool
    
    # 学習中モデルの有無
    is_model_data: bool
    
    # 予測値元データあり
    is_eval_data: bool
    