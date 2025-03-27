import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

mail_address = os.getenv('MAIL_ADDRESS')
password = os.getenv('PASSWORD')

# リフレッシュトークン取得
data={"mailaddress":mail_address, "password":password}
r_post = requests.post("https://api.jquants.com/v1/token/auth_user", data=json.dumps(data))
REFRESH_TOKEN = r_post.json()["refreshToken"]

# IDトークン取得
r_post = requests.post(f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={REFRESH_TOKEN}")
ID_TOKEN = r_post.json()["idToken"]

# j-quantsから株価を取得する
def get_stock_from_cd(
    code,
    start_date,
    is_past = False
):
    
    # 開始日と終了日(3か月 or 200日前)
    from_date = ''
    to_date = ''
    if is_past:
        # 過去200日
        to_date = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
        from_date = (to_date - timedelta(days=200)).strftime('%Y-%m-%d')
        to_date = to_date.strftime('%Y-%m-%d')
    else:
        # 未来３か月
        from_date = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
        to_date = (from_date + timedelta(days=90)).strftime('%Y-%m-%d')
        from_date = from_date.strftime('%Y-%m-%d')
    
    # IDトークンをヘッダーに設定
    headers = {'Authorization': 'Bearer {}'.format(ID_TOKEN)}
    
    # URLに期間を指定してリクエスト
    url = f"https://api.jquants.com/v1/prices/daily_quotes?code={code}&from={from_date}&to={to_date}"
    
    # データを取得
    response = requests.get(url, headers=headers)

    # レスポンスの確認
    rtn_data = {}
    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data['daily_quotes'])
        
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        df = df.rename(columns=lambda x: x.strip())  # 空白削除

        if not all(col in df.columns for col in expected_columns):
            print(f"{code}データが期待したカラムと異なります:", df.columns)
        else:
            df_filters = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
            df_filters[['Dividends', 'Stock Splits']] = '0.0', '0.0'
            rtn_data = df_filters.to_json(orient='records', lines=False, force_ascii=False, date_format='iso')
    else:
        print(f"{code}エラーが発生しました: {response.status_code}")
        
    return rtn_data

# 当時の財務情報を取得する(※ 12週前の分しか使えないからこれは使えない気がする)
def get_finance_from_cd(
    code,
    target_date,
    target_stock
):
    # IDトークンをヘッダーに設定
    headers = {'Authorization': 'Bearer {}'.format(ID_TOKEN)}
    
    # URLに期間を指定してリクエスト
    #url = f"https://api.jquants.com/v1/fins/statements?code={code}&date={date}"
    url = f"https://api.jquants.com/v1/fins/statements?code={code}"
    
    # データを取得
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        finance_df = pd.DataFrame(data['statements'])
        
        # 直近の最新を取得
        date = datetime.strptime(target_date, '%Y-%m-%d') if isinstance(target_date, str) else target_date
        finance_df['DisclosedDate'] = pd.to_datetime(finance_df['DisclosedDate'])
        finance_df_filters = finance_df[finance_df['DisclosedDate'] < date]
        # ソートして、直近のデータを取得
        finance_df_sorted = finance_df_filters.sort_values(by='DisclosedDate', ascending=False)
        brandnew_df = finance_df_sorted.iloc[0]
        
        # EPS（Earnings Per Share）
        eps = brandnew_df.get('EarningsPerShare', None)
        eps = float(eps) if eps not in [None, ''] else None
        
        # 期末発行済株式数 & 期末自己株式数
        total_shares = brandnew_df.get('NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock', None)
        treasury_shares = brandnew_df.get('NumberOfTreasuryStockAtTheEndOfFiscalYear', None)
        
        try:
            total_shares = float(total_shares) if total_shares not in [None, ''] else None
            treasury_shares = float(treasury_shares) if treasury_shares not in [None, ''] else 0.0
        except ValueError:
            total_shares, treasury_shares = None, 0.0
        
        # 発行済株式数（自己株式を除く）
        shares_outstanding = (total_shares - treasury_shares) if total_shares is not None else None
        
        print(f'期末発行済株式数: {total_shares}')
        print(f'期末自己株式数: {treasury_shares}')
        print(f'発行済株式数（自己株式控除後）: {shares_outstanding}')
        
        # BPS（Book Value Per Share）
        bps = brandnew_df.get('BookValuePerShare', None)
        if bps in [None, '']:
            equity = brandnew_df.get('Equity', None)
            try:
                equity = float(equity) if equity not in [None, ''] else None
                bps = (equity / shares_outstanding) if equity is not None and shares_outstanding not in [None, 0] else None
            except ValueError:
                bps = None
        else:
            try:
                bps = float(bps)
            except ValueError:
                bps = None
        
        # PER（株価収益率）
        per = (target_stock / eps) if eps not in [None, 0] else None
        
        # PBR（株価純資産倍率）
        pbr = (target_stock / bps) if bps not in [None, 0] else None
        
        # 時価総額（Market Cap）
        market_cap = (target_stock * shares_outstanding) if shares_outstanding is not None else None
        
        # 計算結果を出力
        print(f"株価: {target_stock}円")
        print(f"EPS: {eps}円")
        print(f"BPS: {bps}円")
        print(f"PER: {per}")
        print(f"PBR: {pbr}")
        print(f"時価総額: {market_cap}円")
        
        return target_stock, eps, bps, per, pbr, market_cap
    
    return 0,0,0,0,0,0
    