# プレスリリースに関連するAPI
import os
import re
import json
import requests
import random
import time
from datetime import datetime
import pandas as pd

# PRTIMES api
pressrelease_api = os.getenv('PRESSRELEASE_API', '')
# 処理対象ページを取得
scraping_max_page = int(os.getenv('SCRAPING_MAX_PAGE', 1))
# 収集データを保存して次のページへ行くかどうか
not_next = os.getenv('NOT_NEXT', 'False').lower() == 'true'

# プレスリリースの一覧を取得して返却
async def get_pressrelease(select_date):
    print('get_pressrelease')
    
    # 上場企業一覧を読み込んでおく
    company_listpath = os.path.join(os.path.dirname(__file__), '../data/data_j.xls')
    comp_df = pd.read_excel(company_listpath, engine='xlrd')
    comp_df.columns = ['Date', 'Code', 'Name', 'Sub1', 'Sub2', 'Sub3', 'Sub4', 'Sub5', 'Sub6', 'Sub7']
    comp_df = comp_df[['Code', 'Name']]
    comp_df['Name'] = comp_df['Name'].apply(lambda x: to_list_link_text(x))
    
    all_data = []
    
    # 最大ページまで回す(*日付が変わったらぬける)
    is_past_date = False
    for page_num in range(1, scraping_max_page):
        
        # キャッシュ回避用
        random_number = ''.join(random.choices('0123456789', k=13))
        
        # ページ取得(apiを発見したのでこいつを利用する)
        #list_page = f'{pressrelease_baseurl}/main/html/searchcorpcate/company_cate_id/001'
        list_page = f'{pressrelease_api}&page={page_num}&random={random_number}'
        
        pr_reponse = requests.get(list_page)
        if pr_reponse.status_code != 200:
            print(f'{list_page}の取得失敗')
            return empty_list()
        
        json_text = re.sub(r"^addReleaseList\((.*)\)$", r"\1", pr_reponse.text)
        
        # json読込
        data = json.loads(json_text)
        
        wk_articles = data.get('articles', [])
        for article in wk_articles:
            time = ''
            place = ''
            history = ''
            
            title = article.get('title', '').strip()
            name = article.get('provider', '').get('name', '').strip()
            # 上場企業リストからコード逆引き
            result = comp_df.loc[comp_df['Name'].str.contains(to_list_link_text(name)), 'Code']
            code = result.iloc[0] if not result.empty else None
            
            # <p>タグを改行に置き換える
            link = article.get('text', '').strip()
            link = re.sub(r'</?p>', '', link) # Pを改行 ブランクに
            link = re.sub(r'<[^>]+>', '', link) # それ以外のタグは削除
            link = re.sub(r'\n+', '', link).strip() # 余計な空白も削除
            link = re.sub(r'\s+', '', link).strip() # 余計な空白も削除
            
            # 日付と時刻を取得
            time = article.get('updated_at', '').get('time_iso_8601', '').strip()
            dt = datetime.fromisoformat(time)
            
            # 日付が変わったら終了
            if dt.strftime('%Y-%m-%d') != select_date:
                is_past_date = True
                break
            
            time = dt.strftime('%H:%M')
            
            all_data.append([time, code, name, title, link, place, history])
            
        # 日付が変わったら終了
        if is_past_date:
            break
        
    return all_data
    
# 空のリスト
def empty_list(select_date):
    return {
            'datalist':[
                {
                    'Time': 'ー',
                    'Code': 'ー',
                    'Name': 'なし',
                    'Title': 'ー',
                    'Link': 'ー',
                    'Place': 'ー',
                    'Place': 'ー'
                }
            ]
        }
    
# 企業名から企業コードの取得のため両方を合わせる
def to_list_link_text(text):
    trans_text = str(text)
    # 株式会社等は外しておく
    trans_text = trans_text.replace('株式会社', '')
    trans_text = trans_text.replace('HD', '')
    trans_text = trans_text.replace('・', '')
    
    trans_text = trans_text.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９"
    ))
    
    clean_text = re.sub(r"\s+", '', trans_text)
    
    return clean_text
