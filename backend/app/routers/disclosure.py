# 開示情報用APIを定義
import os
import pandas as pd
from ..services import disclosure, googleapi, learning, mlp
from ..schemas.disclosure import LearningItem, StatItem
from fastapi import APIRouter
from fastapi import BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import json
import re

router = APIRouter()

# 環境変数から会社リストの保存GCSパスを取得
gcs_list_csv_path = os.getenv('GCS_LIST_CSV_PATH', '')
gcs_work_csv_path = os.getenv('GCS_WORK_CSV_PATH', '')
# 収集データを保存して次のページへ行くかどうか
is_debug = os.getenv('is_debug', 'False').lower() == 'true'


# tdnetの開示を取得(＊料金が高すぎるのでTdnetサイトのスクレイピングで)
@router.get('/tdnetlist/{select_date}')
async def get_tdnetlist(select_date):
    
    # 撮り忘れ
    #select_date = '2025-04-30'
    
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
        
    if not is_debug:
        # consleに開示をアップロードする
        await learning.upload_disclosure_from_list(
            df[['Time','Date','Code', 'Name', 'Title', 'Link', 'Url']],
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
    
    rtnMes = 'エラーなし'
    if not is_debug:
        
        # error行ないかチェック
        error_df = df[
            df['Link'].str.contains(r'^(Error[:：]|\bError\b)', na=False, flags=re.IGNORECASE) & 
            ~df['Link'].str.contains('URL', na=False)
            ]
        
        # 次のリンクがちゃんとあるかも
        error_link = [None, '', 'td?y=&f=']
        
        # いったんエラーが5つ以上の場合にエラー
        if len(error_df) < 5 and next_link not in error_link:
            # consleに開示をアップロードする
            await learning.upload_disclosure_from_list(df,is_new_list=True)
                
            # 全てが完了したら次のリンクをファイルに保存
            with open(nextlink_path, 'w', encoding='utf-8') as f:
                f.write(next_link)
                
            # 一応一個前をバックアップ
            nextlinkbackup_path = os.path.join(os.path.dirname(__file__), '../data/next_link_backup.txt')
            with open(nextlinkbackup_path, 'w', encoding='utf-8') as fback:
                fback.write(target_page)
        else:
            # エラーを確認するための保存
            rtnMes = 'エラーありでストップ'
            await learning.upload_disclosure_from_error(df)
    
    return {
        'message': rtnMes
        }

# 開示から学習を行う
@router.post('/learning')
async def learning_disclosure(item: LearningItem):
    
    # 現在学習中の日付を取得
    target_file_path = os.path.join(os.path.dirname(__file__), '../data/next_learndate.txt')
    if os.path.exists(target_file_path):
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_date = f.read().strip()      
    
    # 日付ファイル毎に
    await learning.learning_from_save_data(target_date, work_load=item.work_load)
    
    return {
        'message': item.date
        }

# 開示とその要約したリストを取得
# @router.post('/summarizelist')
# async def confirm_summarize():
#     return await learning.get_summarize_list()
@router.post('/summarizelist')
async def confirm_summarize(request: Request):
    # ストリーミングレスポンスを返す
    async def data_streamer():
        # ここではすぐにデータを返し始めるよ
        # 例えば、最初の部分だけ返しながらバックグラウンドタスクが処理を続けるとかｗｗｗ
        async for item in get_partial_data():  # get_partial_data()は途中データを返す非同期関数
            yield json.dumps(item) + "\n"
    
    return StreamingResponse(data_streamer(), media_type="application/x-ndjson")

# データ分割取得
async def get_partial_data():
    # page = 34
    # page_size = 1  # 1ページあたり5件、ページングのサイズを設定
    # is_debug = True
    page = 0
    page_size = 5  # 1ページあたり5件、ページングのサイズを設定
    is_debug = False
    
    while True:
        summarize_list = await learning.get_summarize_list(page=page, page_size=page_size)
        
        if not summarize_list:  # 取得したデータが空なら終了
            break
        
        for item in summarize_list:
            yield item
            
        if is_debug:
            break

        page += page_size  # 次のページへ進む


# 学習前データリストを取得
@router.post('/workdatalist')
async def confirm_work_data():
    return await learning.get_work_data_list()

# 予想株価リストを取得
@router.post('/evallist')
async def get_eval_list(item: LearningItem):
    return await learning.eval_target_list(item.date)

# ワークデータの削除
@router.post('/deleteworkdata')
async def delete_work_data():
    
    blob_path = f'{gcs_work_csv_path}/work_data.json'
    
    googleapi.delete_data(blob_path)
    
    return {}

# 予測元データの削除
@router.post('/deleteevaldata')
async def delete_eval_data():
    blob_path = f'{gcs_work_csv_path}/eval_target.json'
    
    googleapi.delete_data(blob_path)
    
    return {}


# MLPモデルの削除(作業中データも対象日も全部クリア)
@router.post('/deletemlpmodel')
async def delete_mlp_model():
    
    # モデルクリア
    mlp.model_delete()
    
    # 作業データクリア
    blob_path = f'{gcs_work_csv_path}/work_data.json'
    googleapi.delete_data(blob_path)
    
    # 作業対象日をリセット
    date_path = os.path.join(os.path.dirname(__file__), '../data/next_learndate.txt')
    if os.path.exists(date_path):
        with open(date_path, 'w', encoding='utf-8') as w:
            w.write('2022-11-28')
    
    return {}
    
# 現在の状態を取得
@router.get('/state', response_model=StatItem)
async def get_now_state():
    
    # 学習中日付
    target_date = ''
    date_path = os.path.join(os.path.dirname(__file__), '../data/next_learndate.txt')
    if os.path.exists(date_path):
        with open(date_path, 'r', encoding='utf-8') as r:
            target_date = r.read()
            
    # 学習中ワークデータ確認
    list_key = f'{gcs_work_csv_path}/work_data.json'
    data_list = googleapi.download_list(list_key)
    
    is_work_data = True
    if not data_list:
        is_work_data = False
        
    # 学習モデルあり
    is_model_data = True
    model = mlp.model_load()
    if model is None:
        is_model_data = False 
        
        
    # 評価用作成データあり
    list_key = f'{gcs_work_csv_path}/eval_target.json'
    data_list = googleapi.download_list(list_key)
    is_eval_data = True
    if not data_list:
        is_eval_data = False
            
    return StatItem(
        target_date=target_date,
        is_work_data=is_work_data,
        is_model_data=is_model_data,
        is_eval_data = is_eval_data
    )