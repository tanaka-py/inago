import os
import requests
import re
import numpy as np
import pandas as pd
import asyncio
from . import process_pdf, yahoofinance, jquants
from datetime import datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from . import pressrelease, googleapi, notice
import weakref

# 環境変数からTdnetの一覧URLを取得
tdnet_listurl = os.getenv('TDNET_LISTURL', '')
# 環境変数からIR Bankの一覧URLを取得
irbank_listurl = os.getenv('IRBANK_LISTURL', '')
# 環境変数から除外会社を取得
exclusion_company = os.getenv('EXCLUSION_COMPANY', '').split(',')
# 環境変数から除外タイトルを取得
exclusion_title = os.getenv('EXCLUSION_TITLE', '').split(',')
# PDF取得要約処理の並列ワーカー数
pdf_summaries_max_workers = int(os.getenv('PDF_SUMMARIES_MAX_WORKERS', 10))

# 環境変数から会社リストの保存GCSパスを取得
gcs_list_csv_path = os.getenv('GCS_LIST_CSV_PATH', '')

# IRBankの最終日
irbank_enddate = pd.to_datetime(os.getenv('IRBANK_ENDDATE', ''))

notice_tasks = weakref.WeakSet() # 自動管理

# 一覧データを取得
async def get_list(
    select_date,
    scraping_max_page
    ):
    # まずは前回データを取得
    # 保存データを取得
    list_key = f'{gcs_list_csv_path}/{select_date}.json'
    org_data_list = googleapi.download_list(list_key)
    org_df = pd.DataFrame(org_data_list)
    
    # TDnet用日付
    tdnet_date = datetime.strptime(select_date, '%Y-%m-%d').strftime('%Y%m%d')
    
    # iframe内のリストパスから開示一覧を取得(全ページいっちゃう)
    base_page_path = "I_list_{:03d}_{}.html"
    
    # 全ページを入れていく
    all_page_data = []
    
    for page_num in range(1, scraping_max_page) :  # 設定ページまで(ページにデータがあるかは判定)
    
        target_list = base_page_path.format(page_num, tdnet_date)
        list_url = f'{tdnet_listurl}{target_list}'
        list_response = requests.get(list_url)
        
        # ページ内データがあるか
        if list_response.status_code != 200 :
            print(f'{page_num}はないよ！{list_url}')
            break
        
        list_response.encoding = 'utf-8'
        list_soup = BeautifulSoup(list_response.text, 'html.parser')
        # 一覧htmlを解析
        main_list = list_soup.find('div', id="main-list")
        if not main_list :
            print(f'{page_num}の中身がないよ！{list_url}')
            break
        
        # pandasでHTML読み込み ヘッダーはなしで(thがなくて１行目を取りこぼす可能性があるから)
        #tables = pd.read_html(str(main_list), header=None)
        #df = tables[0]
        # read_htmlだと<a>が消えてしまうため、ループで読み込んでのちにDataFrameにする
        for row in main_list.find_all('tr'):
            cols = row.find_all('td')
            
            cols = [
                f"{a_tag.text.strip()}:{a_tag['href']}" if (a_tag := ele.find('a')) else ele.text.strip()
                for ele in cols
            ]  # テキスト部分を抽出
            if len(cols) > 0:  # 行が空でない場合
                all_page_data.append(cols)
                
    if not all_page_data:   # データが取得出来てない場合は終了
        return None
                
    df = pd.DataFrame(all_page_data,columns=[
        'Time',
        'Code',
        'Name',
        'Title',
        'Link',
        'Place',
        'History'
    ])
    
    df_filters = df[
        [
            'Time',
            'Code',
            'Name',
            'Title',
            'Link',
            'Place'
        ]
    ]
    
    # 証券コードがないものと東証以外は除外
    df_filters = df_filters[
        df_filters['Code'].str.strip().notna() &
        (df_filters['Place'].str.strip() == '東') &
        (~df_filters['Name'].str.contains('|'.join(exclusion_company), case=False, na=False)) &
        (~df_filters['Title'].str.contains('|'.join(exclusion_title), case=False, na=False))
        ]
    
    # PDFの中身を要約してセット
    df_filters[['Link', 'Title']] = df_filters['Title'].apply(lambda x: pd.Series(extract_pdfurl(x)))
    # 並列で処理を行うように修正 applyではなくmapとなる
    urls = df_filters['Link'].to_list()
    
    # asyncioを使用した非同期サーバ通信取得処理
    #summaries = await process_pdf.in_parallel_asyncio(urls, max_workers=pdf_summaries_max_workers)
    #df_filters['Link'] = summaries
    
    # ThreadPoolExecutorを使用した並列処理
    summaries = process_pdf.in_parallel(urls, max_workers=pdf_summaries_max_workers)
    df_filters['Link'] = df_filters['Link'].map(lambda x: re.sub(r"\s+", " ", str(summaries.get(x, ""))).strip())
    
    #summaries = process_pdf.in_parallel_multiprocessing(urls, max_workers=pdf_summaries_max_workers)
    #df_filters['Link'] = summaries
    
    #　単発
    #df_filters['Link'] = df_filters['Link'].apply(lambda x: process_pdf.summarize_pdf(x))
    
    # プレスリリースからの分も加える
    if pressreleaselist := await pressrelease.get_pressrelease(select_date):
        press_df = pd.DataFrame(pressreleaselist,columns=[
            'Time',
            'Code',
            'Name',
            'Title',
            'Link',
            'Place',
            'History'
        ])
        
        press_df = press_df[[
            'Time',
            'Code',
            'Name',
            'Title',
            'Link',
            'Place'
        ]]
        
        # 扶養なものを省く
        press_df = press_df[
            press_df['Code'].str.strip().notna() &
            (~press_df['Name'].str.contains('|'.join(exclusion_company), case=False, na=False)) &
            (~press_df['Title'].str.contains('|'.join(exclusion_title), case=False, na=False))
            ]
        
        # 連結
        df_filters = pd.concat([df_filters, press_df], ignore_index=True)
    
    # 5桁のものはコードを修正
    df_filters['Code'] = df_filters['Code'].apply(lambda code: code.rstrip('0') if len(code) == 5 else code) 
        
    # 最後に登録済みと結合
    df_new = None
    if org_df.empty:
        df_new = df_filters
        merge_df = df_filters
    else:
        # 登録済みにないものだけを抽出
        df_new = df_filters[~df_filters[['Code', 'Title']].apply(tuple, axis=1).isin(
            org_df[['Code', 'Title']].apply(tuple, axis=1)
        )]
        df_new['Date'] = select_date    # 日付をセット
        merge_df = pd.concat([org_df, df_new], ignore_index=True)
    
    # 最終処理
    # codeの最後が0の場合は0を省く念のため
    merge_df['Code'] = merge_df['Code'].apply(lambda code: code.rstrip('0') if len(code) == 5 else code) 
    merge_df['Date'] = select_date    # 日付をセット
    merge_df = merge_df.replace([np.nan, np.inf, -np.inf], None)
    
    # ここで新規に追加するものに対して処理を行う！
    # modelから予測を取得してメールなりで送信といった感じで
    if df_new is not None and not df_new.empty:
        task = asyncio.create_task(notice.send(df_new.copy()))  # コピーを送信する
        notice_tasks.add(task)
        task.add_done_callback(lambda t: notice_tasks.discard(t))    # タスクが完了したらsetから削除
    
    return merge_df

# Title内からPDFへのURLを取得
def extract_pdfurl(title) :
    match = re.search(r':[^\s]+\.pdf', title)
    url = f'{tdnet_listurl}{match.group(0).replace(':', '')}' if match else None
    company_name = re.sub(r':[^\s]+\.pdf', '', title)
    
    return url, company_name

#　コードからその日から一か月の株価をjsonで返却
def get_amonth_finance(
    code,
    start_date,
    is_past = False
):
    start_str_date = ''
    end_str_date = ''
    if is_past:
        # 開始、終了日
        end_str_date = start_date.to_pydatetime().strftime('%Y-%m-%d')
        start_str_date = get_200days_past(start_date)
    else:
        # 開始、終了日
        start_str_date = start_date.to_pydatetime().strftime('%Y-%m-%d')
        end_str_date = get_3month_later(start_date)
    
    # まずは会社を ないものは空のデータを yfinaceは株価のほうが失敗が多い
    # j-quants→yfinance→yahoo financeのスクレイピングを順に試していく
    comp = jquants.get_stock_from_cd(code, start_date, is_past)
    if not comp:
        print(f'{code}:jquantsがないためyfinance')
        stock = yf.Ticker(f'{code}.T')
        df = stock.history(start=start_str_date, end=end_str_date)
        comp = {}
        if not df.empty:
            df['Date'] = df.index.date.astype(str)  # 日付部分だけ取り出す
            comp = df.to_json(orient='records', lines=False, force_ascii=False)
        else:
            print(f'{code}:最後のyahoo finance')
            comp = yahoofinance.get_yahoofinance_stocllist(code, start_date, is_past)
            if not comp:
                print(f'{code}:どこにもない')
            
    
    # 日経
    stockn = yf.Ticker('^N225')
    dfn = stockn.history(start=start_str_date, end=end_str_date)
    dfn['Date'] = dfn.index.date.astype(str)  # 日付部分だけ取り出す
    n225 = dfn.to_json(orient='records', lines=False, force_ascii=False)
    
    # グロース(マザーズ) 
    growth = {}
    if is_past:
        # 過去はCSVから
        growth_file = os.path.join(os.path.dirname(__file__), '../data/growth.csv')
        growth_df = pd.read_csv(growth_file)
        growth_df.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'rate']
        growth_df = growth_df[(start_str_date <= growth_df['Date']) & (growth_df['Date'] < end_str_date)]
        growth_df= growth_df.sort_values('Date')
        growth = growth_df.to_json(orient='records', lines=False, force_ascii=False)
    
    return comp, n225, growth

# 3か月後を取得
def get_3month_later(start_date):
    month_later = ''
    
    # 文字列の日付をdatetimeオブジェクトに変換
    date_obj = start_date.to_pydatetime()

    # 3か月後の日付を求める
    new_date_obj = date_obj + relativedelta(months=3)

    # 新しい日付を文字列に変換
    month_later = new_date_obj.strftime('%Y-%m-%d')
    
    return month_later

# 200日past
def get_200days_past(start_date):
    month_later = ''
    
    # 文字列の日付をdatetimeオブジェクトに変換
    date_obj = start_date.to_pydatetime()

    # 3か月後の日付を求める
    new_date_obj = date_obj - relativedelta(days=200)

    # 新しい日付を文字列に変換
    month_later = new_date_obj.strftime('%Y-%m-%d')
    
    return month_later

# 一覧データを取得(IR Bankから)
async def get_irbank_list(
    scraping_max_page,
    init_page
    ):
    
    # 対象ページ
    target_page = init_page
    
    # 全ページを入れていく
    all_page_data = []
    
    for page_num in range(1, scraping_max_page) :  # 設定ページまで(ページにデータがあるかは判定)
    
        list_url = f'{irbank_listurl}{target_page}'
        list_response = requests.get(list_url)
        
        # ページ内データがあるか
        if list_response.status_code != 200 :
            print(f'{page_num}はないよ！{list_url}')
            break
        
        list_response.encoding = 'utf-8'
        list_soup = BeautifulSoup(list_response.text, 'html.parser')
        # 一覧htmlを解析
        main_list = list_soup.find('table', class_="cs")
        if not main_list :
            print(f'{page_num}の中身がないよ！{list_url}')
            break
        
        # read_htmlだと<a>が消えてしまうため、ループで読み込んでのちにDataFrameにする
        for row in main_list.find_all('tr'):
            cols = row.find_all('td')
            
            cols = [
                f"{a_tag.text.strip()}=>{a_tag['href']}" 
                if (a_tag := ele.find('a')) and not any(cls in ele.get('class', []) for cls in ['ct', 'wmx300'])
                else ele.text.strip()
                for ele in cols
            ]  # テキスト部分を抽出
            if len(cols) >= 3:  # 行が空でない場合
                all_page_data.append(cols)
                
        # 次のリンクへ
        target_page = list_soup.find('tr', id='loading').attrs['data-nx'].replace('/', '').replace('&pg=true', '')
                
    df = pd.DataFrame(all_page_data,columns=[
        'Date',
        'Code',
        'Name',
        'Title'
    ])
    
    check_df = df[pd.to_datetime(df['Date']) <= irbank_enddate]
    if not check_df.empty:
        print(f'{irbank_enddate}のデータあり')
        
    df = df[pd.to_datetime(df['Date']) > irbank_enddate]
    
    if df.empty:
        print(f'{irbank_enddate}以前のデータしかなかったため終了')
        return target_page, None
    
    # 証券コードがないものと東証以外は除外
    df_filters = df[
        df['Code'].str.strip().notna() &
        (~df['Name'].str.contains('|'.join(exclusion_company), case=False, na=False)) &
        (~df['Title'].str.contains('|'.join(exclusion_title), case=False, na=False)) &
        (~df['Name'].str.isdigit()) 
        ]
    
    # PDFの中身を要約してセット
    df_filters[['Link', 'Title']] = df_filters['Title'].apply(lambda x: pd.Series(extract_detailurl(x)))
    # 詳細ページから日付とpdfリンクを取得
    df_filters[['Date', 'Link']] = df_filters['Link'].apply(lambda x: pd.Series(get_detail_data(x))) 
    # 並列で処理を行うように修正 applyではなくmapとなる
    urls = df_filters['Link'].to_list()
    
    # ThreadPoolExecutorを使用した並列処理
    summaries = process_pdf.in_parallel(urls, max_workers=pdf_summaries_max_workers)
    df_filters['Link'] = df_filters['Link'].map(lambda x: re.sub(r"\s+", " ", str(summaries.get(x, ""))).strip())
    
    return target_page, df_filters

# Title内から詳細ページURLを取得
def extract_detailurl(title) :
    match = re.search(r'=>[^\s]+', title)
    url = f'{irbank_listurl}{match.group(0).replace('=>/', '')}' if match else None
    company_name = re.sub(r'=>[^\s]+', '', title)
    
    return url, company_name

# Link先の詳細ページから時間とpdfへのリンクを取得
def get_detail_data(link) :
    detail_response = requests.get(link)
    
    if detail_response.status_code != 200:
        return None, None
    
    detail = BeautifulSoup(detail_response.text, 'html.parser')
    
    # 資料情報の <dl> を取得
    info_dl = detail.find('dl', id='info')
    
    if not info_dl:
        return None, None

    # 日時を取得
    date_time = None
    pdf_link = None
    dt_tags = info_dl.find_all('dt')

    for dt in dt_tags:
        if dt.text.strip() == "【日時】":
            # 次の <dd> を取得
            date_dd = dt.find_next_sibling('dd')
            if date_dd:
                date_time_str = date_dd.text.strip()

        if dt.text.strip() == "【資料】":
            # 次の <dd> の中の <a> の href を取得
            material_dd = dt.find_next_sibling('dd')
            if material_dd and (a_tag := material_dd.find('a')):
                pdf_link = a_tag.get('href')
    
    date_time = None
    if date_time_str:
        try:
            date_time = datetime.strptime(date_time_str, "%Y年%m月%d日 %H時%M分").strftime('%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            print(f"日時の変換に失敗: {e}")
    
    return date_time, pdf_link