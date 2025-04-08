import os
import json
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from . import googleapi, disclosure, finance, mlp, summarize, summarize_work
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import re

import warnings # いったん
# 特定の警告を無視する
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# 環境変数から会社リストの保存GCSパスを取得
gcs_list_csv_path = os.getenv('GCS_LIST_CSV_PATH', '')
gcs_work_csv_path = os.getenv('GCS_WORK_CSV_PATH', '')

is_debug = os.getenv('is_debug', 'False').lower() == 'true'

# 対象外開示リスト
exclude_title_path = os.path.join(os.path.dirname(__file__), '../data/exclude_title.csv')
exclude_title_df = pd.read_csv(exclude_title_path, header=None)
exclude_title = exclude_title_df.iloc[:,0].to_list()

# 保存データから学習を行う
async def learning_from_save_data(
    target_date,
    work_load
    ):
    
    # 特徴量素材
    features = []
    targets = []
    documents = []
    document_summaries = []
    debug_data = []
        
    total_count = 0
    
    if work_load:
        # 作業データ読み込み
        list_key = f'{gcs_work_csv_path}/{target_date}.json'
        data_list = googleapi.download_list(list_key)
        
        load_df = pd.DataFrame(data_list)
        
        features = load_df['features']
        targets = load_df['targets']
        document_summaries = load_df['document_summaries']
        
        total_count = len(features)
        
    else:
        
        # 保存データを取得
        list_key = f'{gcs_list_csv_path}/{target_date}.json'
        data_list = googleapi.download_list(list_key)
        
        # 株価が入ってないものがあれば保存しなおし
        data_list = re_get_stock_data(
                data_list,
                target_date
            )
        
        # 2. 株価データの特徴量作成
        print(f'総件数：{len(data_list)}')
        
        for item in data_list:
            total_count += 1
            
            if is_broken_text(item['Link']):
                print(f'開示文章が文字化けしてるためスルー：{item['Code']}')
                continue
            
            # 過去3か月の株価たちを取得
            past_start_date = pd.to_datetime(item['Date'])
            past_stock_json, past_n225_json, past_growth_json = disclosure.get_amonth_finance(
                item['Code'],
                past_start_date,
                True
            )
            
            # JSONデータをDataFrameに変換
            if not past_stock_json:
                print(f'過去株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            if not past_n225_json:
                print(f'過去日経株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            if not past_growth_json:
                print(f'過去グロース株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            
            df_stock = pd.DataFrame(json.loads(past_stock_json))
            df_nikkei = pd.DataFrame(json.loads(past_n225_json))
            df_mothers = pd.DataFrame(json.loads(past_growth_json))
            df_stock_targets = pd.DataFrame(json.loads(item['Stock']))  # 結果用
            
            if df_stock_targets.empty:
                print(f'予想用株価がないためスルー：{item['Code']}') 
                continue
            
            # 財務情報を取得
            wk_stock = df_stock_targets[df_stock_targets['Date'] == target_date]
            if wk_stock.empty:
                print(f'{item['Code']}：対象日の財務データがとれないのは対象外')
                continue
            
            code = item['Code']
            first_stock_open = wk_stock.iloc[0]['Open']
            # 開示日
            disclosure_date = pd.to_datetime(item["DateKey"])
            
            # 各指標特徴量取得
            aggregated_features = calculate_aggregated_features(
                code,
                first_stock_open,
                df_stock, 
                df_nikkei, 
                df_mothers, 
                df_stock_targets,
                target_date, 
                disclosure_date
                )
            if aggregated_features is None:
                continue
            
            # 開示日から各ターゲットの日数後の変動率を計算(求める結果)
            days_list = [0, 3, 7, 14, 21, 28, 35, 42, 49]
            rates = get_last_valid_change_rate(df_stock_targets, days_list)
            
            if any(rate is None or rate == -9999 for rate in rates):
                print(f'{item['Code']}：結果株価変化率がとれないのは対象外')
                continue
            
            targets.append(rates)

            # 特徴量リストに追加
            features.append(aggregated_features)
            
            # 開示をセット
            documents.append(item["Link"])
        
        # 要約ではなく、元の文章からいらんもんを省くスタイルで
        document_summaries = [summarize.brushup_text(document) for document in documents]
    
    print(f'総件数：{total_count} features件数：{len(features)} targets件数：{len(targets)} documents件数：{len(document_summaries)}')
    
    if len(targets) < 1:
        # 対象がない
        print('対象データなし：終了')
        return
    
    # DataFrameに格納
    work_df = pd.DataFrame({
            'features': features,
            'targets': targets,
            'document_summaries': document_summaries
        })
    
    if work_load:
        # work_load後は学習
        # MLP学習
        if not is_debug:    # デバッグではとらない
            #mlp.mlp_learning(document_summaries[0:1], features[0:1], targets[0:1])
            mlp.mlp_learning(document_summaries, features, targets)
    else:
        # consleに開示をアップロードする
        googleapi.rewrite_list(
                work_df,
                f'{gcs_work_csv_path}/work_data.json'
            )
        
    # 表示用の場合に一応返却
    return work_df
    
# 株価予想一覧取得
async def eval_target_list(
    target_date
    ):
    
    # 特徴量素材
    features = []
    documents = []
    document_summaries = []
        
    total_count = 0
    is_reload = False
    target_df = None
    
    # まずはworkの中を見てからそっちにある場合はそっちから
    try:
        list_key = f'{gcs_work_csv_path}/{target_date}.json'
        data_list = googleapi.download_list(list_key)
        
        if not data_list:
            is_reload = True
        else:
            target_df = pd.DataFrame(data_list)
            features = target_df['features']
            document_summaries = target_df['document_summaries']
            
    except Exception as e:
        print("保存データがないため取得しなおし")
        is_reload = True
        
    # 保存データがないため取得しなおし
    if is_reload:
        # 保存データを取得
        list_key = f'{gcs_list_csv_path}/{target_date}.json'
        data_list = googleapi.download_list(list_key)
        
        # 2. 株価データの特徴量作成
        print(f'総件数：{len(data_list)}')
        
        for item in data_list:
            total_count += 1
            
            if is_broken_text(item['Link']):
                print(f'開示文章が文字化けしてるためスルー：{item['Code']}')
                continue
            
            # 過去3か月の株価たちを取得
            past_start_date = pd.to_datetime(item['Date'])
            past_stock_json, past_n225_json, past_growth_json = disclosure.get_amonth_finance(
                item['Code'],
                past_start_date,
                True
            )
            
            # JSONデータをDataFrameに変換
            if not past_stock_json:
                print(f'過去株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            if not past_n225_json:
                print(f'過去日経株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            if not past_growth_json:
                print(f'過去グロース株価がないため特徴量が入れれないスルー：{item['Code']}') 
                continue
            
            df_stock = pd.DataFrame(json.loads(past_stock_json))
            df_nikkei = pd.DataFrame(json.loads(past_n225_json))
            df_mothers = pd.DataFrame(json.loads(past_growth_json))
            this_date_stock = disclosure.target_date_open_stock(past_start_date, item['Code'])
            df_stock_targets = pd.DataFrame(json.loads(this_date_stock))  # 結果用
            
            if df_stock_targets.empty:
                print(f'予想用株価がないためスルー：{item['Code']}') 
                continue
            
            # 財務情報を取得
            wk_stock = df_stock_targets[df_stock_targets['Date'] == target_date]
            if wk_stock.empty:
                print(f'{item['Code']}：対象日の財務データがとれないのは対象外')
                continue
            
            code = item['Code']
            first_stock_open = wk_stock.iloc[0]['Open']
            # 開示日
            disclosure_date = pd.to_datetime(item["DateKey"])
            
            # 各指標特徴量取得
            aggregated_features = calculate_aggregated_features(
                code,
                first_stock_open,
                df_stock, 
                df_nikkei, 
                df_mothers, 
                df_stock_targets,
                target_date, 
                disclosure_date
                )
            if aggregated_features is None:
                continue

            # 特徴量リストに追加
            features.append(aggregated_features)
            
            # 開示をセット
            documents.append(item["Link"])
        
        # 要約ではなく、元の文章からいらんもんを省くスタイルで
        document_summaries = [summarize.brushup_text(document) for document in documents]
        
        # DataFrameに格納
        target_df = pd.DataFrame({
                'features': features,
                'document_summaries': document_summaries
            })
        # consleに開示をアップロードする
        googleapi.rewrite_list(
                target_df,
                f'{gcs_work_csv_path}/{target_date}.json'
            )
        
  
    print(f'総件数：{total_count} features件数：{len(features)} documents件数：{len(document_summaries)}')
    
    model = mlp.model_load()
    
    if model is None:
        print('modelが読み込めないため予測が出来ません')
    else:
        # 予想変化率を取得
        target_df['targets'] = target_df.apply(lambda row: mlp.targets_from_model(model, row['document_summaries'], row['features']), axis=1) 
    
    return target_df.to_dict(orient="records")
    
       
# 指標特徴量(feature) 
# features：開示文章が出た時点の各指標
#           'EPS'
#           'ROE'
#           'PER'
#           'PBR'
#           'Market Capitalization Log'
#           'NikkeiCorr'
#           'MothersCorr'
#           'RSI'
#           'MovingAverage50'
#           'MovingAverage200'
#           'ATR'
#           'MACD'
#           'Signal'
#           'UpperBand'
#           'LowerBand'
#           'PercentR'
#           'ADX'
def calculate_aggregated_features(
    code,
    first_stock_open,
    df_stock, 
    df_nikkei, 
    df_mothers, 
    df_stock_targets, 
    target_date, 
    disclosure_date
    ):
    # こっちは各種ファンダメンタル指標も
    aggregated_features = finance.get_finance_from_csv(
        code,
        target_date,
        first_stock_open
    )
        
    if not isinstance(aggregated_features, dict):
        print(f'{code}：財務データがとれないのは対象外')
        return None

    # 数値データの変換
    # 数値データの変換を一気に
    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
        df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce')
        df_nikkei[col] = pd.to_numeric(df_nikkei[col], errors='coerce')
        df_mothers[col] = pd.to_numeric(df_mothers[col], errors='coerce')

    df_stock_targets['Open'] = pd.to_numeric(df_stock_targets['Open'], errors='coerce')
    df_stock_targets['Close'] = pd.to_numeric(df_stock_targets['Close'], errors='coerce')

    # 変化率（パーセンテージ）
    df_stock['CloseChange'] = df_stock['Close'].pct_change() * 100  
    df_nikkei['CloseChange'] = df_nikkei['Close'].pct_change() * 100  
    df_mothers['CloseChange'] = df_mothers['Close'].pct_change() * 100
    df_stock_targets['CloseChange'] = df_stock_targets['Close'].pct_change() * 100

    # 開示日の±200日間のデータを抽出(過去のやつだから)
    df_stock['DaysSinceDisclosure'] = (pd.to_datetime(df_stock["Date"]) - disclosure_date).dt.days
    df_stock_recent = df_stock[(df_stock['DaysSinceDisclosure'] >= -200) & (df_stock['DaysSinceDisclosure'] <= 200)]

    df_nikkei['DaysSinceDisclosure'] = (pd.to_datetime(df_nikkei["Date"]) - disclosure_date).dt.days
    df_nikkei_recent = df_nikkei[df_nikkei['DaysSinceDisclosure'].isin(df_stock_recent['DaysSinceDisclosure'])]

    df_mothers['DaysSinceDisclosure'] = (pd.to_datetime(df_mothers["Date"]) - disclosure_date).dt.days
    df_mothers_recent = df_mothers[df_mothers['DaysSinceDisclosure'].isin(df_stock_recent['DaysSinceDisclosure'])]

    # --- 📌 株価 vs 指数の相関関係 ---
    NikkeiCorr = None
    if not df_stock_recent.empty and not df_nikkei_recent.empty:
        NikkeiCorr = df_stock_recent["CloseChange"].corr(df_nikkei_recent["CloseChange"])
        
    aggregated_features["NikkeiCorr"] = 0 if NikkeiCorr is None or pd.isna(NikkeiCorr) else NikkeiCorr 
    
    MothersCorr = None
    if not df_stock_recent.empty and not df_mothers_recent.empty:
        MothersCorr = df_stock_recent["CloseChange"].corr(df_mothers_recent["CloseChange"])
        
    aggregated_features["MothersCorr"] = 0 if MothersCorr is None or pd.isna(MothersCorr) else MothersCorr 

    # --- 📌 テクニカル指標 ---
    df_stock['RSI'] = calculate_rsi(df_stock['Close'])  # RSI（相対力指数）
    df_stock['MovingAverage50'] = df_stock['Close'].rolling(window=50, min_periods=1).mean()  # 50日移動平均線
    df_stock['MovingAverage200'] = df_stock['Close'].rolling(window=200, min_periods=1).mean()  # 200日移動平均線

    # RSI、移動平均をdf_stock_recentにも反映
    df_stock_recent['RSI'] = df_stock['RSI']
    df_stock_recent['MovingAverage50'] = df_stock['MovingAverage50']
    df_stock_recent['MovingAverage200'] = df_stock['MovingAverage200']

    aggregated_features.update({
        "RSI": df_stock_recent["RSI"].dropna().iloc[-1] if not df_stock_recent["RSI"].dropna().empty else 0,
        "MovingAverage50": df_stock_recent["MovingAverage50"].dropna().iloc[-1] if not df_stock_recent["MovingAverage50"].dropna().empty else 0,
        "MovingAverage200": df_stock_recent["MovingAverage200"].dropna().iloc[-1] if not df_stock_recent["MovingAverage200"].dropna().empty else 0,
    })

    # --- 📌 ボラティリティ指標（修正版）---
    df_stock['High-Low'] = df_stock['High'] - df_stock['Low']
    df_stock['High-ClosePrev'] = abs(df_stock['High'] - df_stock['Close'].shift(1))
    df_stock['Low-ClosePrev'] = abs(df_stock['Low'] - df_stock['Close'].shift(1))

    # True Range（TR）の正しい計算
    df_stock['TR'] = df_stock[['High-Low', 'High-ClosePrev', 'Low-ClosePrev']].max(axis=1)

    # ATR（14日間の移動平均）
    df_stock['ATR'] = df_stock['TR'].rolling(window=14, min_periods=1).mean()

    # df_stock_recentにATRを反映
    df_stock_recent['ATR'] = df_stock['ATR']

    # ATRの平均を特徴量として集約
    aggregated_features["ATR"] = df_stock_recent["ATR"].dropna().mean() if not df_stock_recent["ATR"].dropna().empty else 0

    # --- 📌 テクニカル指標 ---
    df_stock = calculate_macd(df_stock)  # MACD
    df_stock = calculate_bollinger_bands(df_stock)  # ボリンジャーバンド
    df_stock = calculate_percent_r(df_stock)  # パーセントレンジ
    df_stock = calculate_adx(df_stock)  # ADX

    # 追加した指標を集計
    aggregated_features.update({
        "MACD": df_stock['MACD'].dropna().iloc[-1] if not df_stock['MACD'].dropna().empty else 0,
        "Signal": df_stock['Signal'].dropna().iloc[-1] if not df_stock['Signal'].dropna().empty else 0,
        "UpperBand": df_stock['UpperBand'].dropna().iloc[-1] if not df_stock['UpperBand'].dropna().empty else 0,
        "LowerBand": df_stock['LowerBand'].dropna().iloc[-1] if not df_stock['LowerBand'].dropna().empty else 0,
        "PercentR": df_stock['PercentR'].dropna().iloc[-1] if not df_stock['PercentR'].dropna().empty else 0,
        "ADX": df_stock['ADX'].dropna().iloc[-1] if not df_stock['ADX'].dropna().empty else 0,
    })
    
    # None、NaNは0に
    aggregated_features = {
        key : 0 if value is None or pd.isna(value) else value
        for key, value in aggregated_features.items()
    }

    return aggregated_features

    
    
# 開示文章の文字化けチェック(文字化けしてる開示はスルー)
def is_broken_text(text):
    try:
        # 文字列をバイト列にエンコードしてから、デコードしてみる
        byte_data = text.encode('utf-8')  # エンコード
        decoded_text = byte_data.decode('utf-8', errors='strict')  # デコード

        # 制御文字や不可視文字が含まれているかを確認
        for char in decoded_text:
            if ord(char) < 32 or ord(char) == 127:  # 制御文字や不可視文字
                return True  # 壊れていると判定

        # 特定の異常なUnicode文字（例えば、ࢆ）をチェック
        if re.search(r'[\u0500-\u05FF\u200B\u200C\u200D\u2060\u2061\u2062ࢆ]', decoded_text):
            return True  # 異常なUnicode文字があれば壊れていると判定

        # 日本語が含まれていない場合は壊れた文字列と判定
        if not re.search(r'[ぁ-んァ-ン一-龯]', text):  # 日本語の文字範囲にマッチしない場合
            return True

        return False  # 壊れていなければ False

    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        # エンコードまたはデコードエラーで壊れていると判定
        return True
    
# 株価情報が入ってなかったら登録しなおし
def re_get_stock_data(
    data_list,
    target_date
    ):
    
    # 全て株価が入っていたらそのまま返す
    modify_list = data_list
    
    # ３か月前チェック
    org_date = datetime.today()
    before_90_date = org_date - timedelta(days=90)
    list_date = pd.to_datetime(target_date)
    
    df = pd.DataFrame(data_list)
    
    
    # 変なとれかたをしてるので取り直す
    df[['Stock', 'N225', 'Growth']] = df.apply(
        lambda row: pd.Series(
            disclosure.get_amonth_finance(row['Code'], pd.to_datetime(row['Date']))
            ),
        axis=1
    )
    
    # 書き込み、読み込みし直し
    key = f'{gcs_list_csv_path}/{target_date}.json'
    googleapi.rewrite_list(
        df,
        key
    )
    
    modify_list = googleapi.download_list(key)
    
    return modify_list

# RSI (Relative Strength Index) を計算
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()  # windowに変更
    avg_loss = loss.rolling(window=window).mean()  # windowに変更
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# MACDの計算
def calculate_macd(df):
    # 短期と長期のEMAを計算
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()  
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    # MACDライン
    df['MACD'] = df['EMA12'] - df['EMA26']
    # Signalライン（MACDの9日間のEMA）
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# ボリンジャーバンドの計算
def calculate_bollinger_bands(df, window=20):
    df['MA20'] = df['Close'].rolling(window=window).mean()  # 20日移動平均
    df['STD20'] = df['Close'].rolling(window=window).std()  # 20日標準偏差
    df['UpperBand'] = df['MA20'] + (df['STD20'] * 2)  # 上限
    df['LowerBand'] = df['MA20'] - (df['STD20'] * 2)  # 下限
    return df

# パーセントレンジ（%R）の計算
def calculate_percent_r(df, window=14):
    df['HighestHigh'] = df['High'].rolling(window=window).max()
    df['LowestLow'] = df['Low'].rolling(window=window).min()
    df['PercentR'] = 100 * (df['HighestHigh'] - df['Close']) / (df['HighestHigh'] - df['LowestLow'])
    return df

# ADXの計算
def calculate_adx(df, window=14):
    # True Rangeの計算
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # Directional Movementの計算
    df['DM+'] = df['High'] - df['High'].shift(1)
    df['DM-'] = df['Low'].shift(1) - df['Low']

    df['DM+'] = np.where(df['DM+'] > 0, df['DM+'], 0)
    df['DM-'] = np.where(df['DM-'] > 0, df['DM-'], 0)

    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['PDI'] = (df['DM+'].rolling(window=window).sum() / df['ATR']) * 100  # Positive Directional Index
    df['NDI'] = (df['DM-'].rolling(window=window).sum() / df['ATR']) * 100  # Negative Directional Index

    # ADXの計算
    df['ADX'] = (abs(df['PDI'] - df['NDI']) / (df['PDI'] + df['NDI'])).rolling(window=window).mean() * 100
    return df

# listからまとめて変化率を取得する
def get_last_valid_change_rate(
    df,
    days_list,
    is_debug = False
    ):
    last_valid_value = None
    change_rates = {}
    
    for days in days_list:
        rate = calculate_change_rate(df, days, is_debug) if not df.empty else None
        if rate is not None:
            last_valid_value = rate  # 最後の有効な値を更新
        else:
            rate = last_valid_value if last_valid_value is not None else -9999  # 欠損なら最後の有効値 or -9999
        change_rates[f"ChangeRate_{days}"] = rate
    
    return change_rates

# 株価変動率を計算する関数
def calculate_change_rate(
    df, 
    days,
    is_debug = False
    ):
    df_copy = df.copy()
    df_copy['ChangeRate'] = (
        df_copy['Close'].shift(-days) - df_copy['Close'] 
        if days != 0 else 
        df_copy['Close'] - df_copy['Open']
        ) / df_copy['Close'] * 100  
    
    # 最初の行（0日目）の `days` 日後の変動率だけ取得
    if len(df_copy) > days:
        if is_debug:
            # デバッグで欲しいのは株価と株価率
            return f'{df_copy['Close'].shift(-days).iloc[0]}:{df_copy['ChangeRate'].iloc[0]}'
        else:
            # 通常は株価率のみ
            return df_copy['ChangeRate'].iloc[0]
    else:
        return 
  
# 学習前データ確認
async def get_work_data_list(
):
    # 保存データを取得
    list_key = f'{gcs_work_csv_path}/work_data.json'
    data_list = googleapi.download_list(list_key)
    
    # 取得jsonを
    data_df = pd.DataFrame(data_list)
    
    return data_df.to_dict(orient="records")

# 開示文章とその要約を取得  デバッグ確認用
async def get_summarize_list(
):
    # 現在作業中の対象日を取得
    target_date_path = os.path.join(os.path.dirname(__file__), '../data/next_learndate.txt')
    if os.path.exists(target_date_path):
        with open(target_date_path, 'r', encoding='utf-8') as f:
            target_date = f.read().strip()
    
    # 保存データを取得
    list_key = f'{gcs_list_csv_path}/{target_date}.json'
    data_list = googleapi.download_list(list_key)
    
    # 取得jsonを
    data_df = pd.DataFrame(data_list)
    data_df = data_df[['Link']]
    
    data_df = data_df[~data_df['Link'].apply(is_broken_text)]
    #data_df = data_df[0:1]
    
    links = data_df['Link'].tolist()
    
    #data_df['Summarize'] = summarize_work.summarize_in_parallel(links)
    #data_df['Summarize'] = summarize.brush_in_parallel(links)
    data_df['Summarize'] = [summarize.brushup_text(link) for link in links]
    
    return data_df.to_dict(orient="records")

# 一覧から取得した
async def upload_disclosure_from_list(
    df,
    is_today = False
):
    # 保存用の日付のみのキー取得
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DateKey'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # NaNをNoneに変換
    df = df.where(pd.notnull(df), None)
    
    # 日付がないものはここで省く
    df = df[df['Date'].notna()]
    
    # 一か月の株価情報を取得
    if is_today:
        # 本日の場合データがないので空で
        df[['Stock', 'N225', 'Growth']] = df.apply(lambda x: pd.Series({'Stock':{}, 'N225':{}, 'Growth':{}}), axis=1)
    else:
        # 株価セット
        df[['Stock', 'N225', 'Growth']] = df.apply(
            lambda row: pd.Series(disclosure.get_amonth_finance(row['Code'], row['Date'])), axis=1
            )
    
    # 日付別にアップロード
    grouped_dfs = [group for _, group in df.groupby('DateKey')]
    for item_df in grouped_dfs:
        date_key = item_df['DateKey'].iloc[0]
        
        #jsonをGCSに保存
        if is_today:
            # 完全上書き
            googleapi.rewrite_list(
                item_df,
                f'{gcs_list_csv_path}/{date_key}.json'
            )
        else:
            # 既存データそのままで保存する
            googleapi.upload_list(
                item_df,
                f'{gcs_list_csv_path}/{date_key}.json'
            )
    