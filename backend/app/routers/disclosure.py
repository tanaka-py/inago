# 開示情報用APIを定義
import os
import pandas as pd
from ..services import disclosure, googleapi, learning
from ..schemas.disclosure import LearningItem
from fastapi import APIRouter

router = APIRouter()

# 環境変数から会社リストの保存GCSパスを取得
gcs_list_csv_path = os.getenv('GCS_LIST_CSV_PATH', '')
# 収集データを保存して次のページへ行くかどうか
not_next = os.getenv('NOT_NEXT', 'False').lower() == 'true'


# tdnetの開示を取得(＊料金が高すぎるのでTdnetサイトのスクレイピングで)
@router.get('/tdnetlist/{select_date}')
async def get_tdnetlist(select_date):
    
    # 処理対象ページを取得
    scraping_max_page = int(os.getenv('SCRAPING_MAX_PAGE', 1))
    
    # 開示一覧取得
    df = await disclosure.get_list(
        select_date,
        scraping_max_page
        )
    
    if df is None or df.empty:
        return {
            'datalist': []
        }
        
    if not not_next:
        # consleに開示をアップロードする
        await learning.upload_disclosure_from_list(
            df[['Time','Date','Code', 'Name', 'Title', 'Link']],
            is_today=True
            )
    
    return {
        'datalist': df.to_dict(orient='records')
    }

# 開示情報を収集する(CSVに落とす)
@router.post('/upload')
async def upload_disclosure(item: LearningItem):
    
    # moreで変換していく値 スタートはnext_link.txtで管理
    target_page = ''
    nextlink_path = os.path.join(os.path.dirname(__file__), '../data/next_link.txt')
    if os.path.exists(nextlink_path):
        with open(nextlink_path, "r", encoding='utf-8') as f:
            target_page = f.read().strip()
    
     # 処理対象ページを取得
    scraping_max_page = int(os.getenv('SCRAPING_MAX_PAGE', 1))
    
    # 開示一覧取得
    next_link, df = await disclosure.get_irbank_list(
        scraping_max_page,
        target_page
        )
    
    if df is None or df.empty:
        return {
            'message': '開示が取得できなかったため終了'
            }
    
    if not not_next:
        # consleに開示をアップロードする
        await learning.upload_disclosure_from_list(df)
            
        # 全てが完了したら次のリンクをファイルに保存
        with open(nextlink_path, 'w', encoding='utf-8') as f:
            f.write(next_link)
    
    return {
        'message': item.date
        }

# 開示から学習を行う
@router.post('/learning')
async def learning_disclosure(item: LearningItem):
    
    # 現在学習中の日付を取得
    target_date = ''
    target_file_path = os.path.join(os.path.dirname(__file__), '../data/next_learndate.txt')
    if os.path.exists(target_file_path):
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_date = f.read().strip()
            
    
    # 日付ファイル毎に
    learning.learning_from_save_data(target_date)
    
    return {
        'message': item.date
        }

# 開示とその要約したリストを取得（決算or決算以外）
# LearningItem.is_finance_onlyで管理
@router.post('/summarizelist')
async def confirm_summarize_financial(item: LearningItem):
    return await learning.get_summarize_list(item.date, is_financial_only=item.is_finance_only)
    
    
@router.get('/dummylist/{select_date}')
async def get_dummylist():
    print('get_dummylist')
    return {
        'datalist':[
            {
                'Time': '22:00',
                'Code': 'tete',
                'Name': 'dummy',
                'Title': 'なんとか株式会社',
                'Link': 'リンク',
                'Place': '東証やで'
            }
        ]
    }