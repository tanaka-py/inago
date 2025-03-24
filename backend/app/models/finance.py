# 通期からのfinance情報を特徴量に加えるためCSVから算出
import os
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

columns_needed = [
    "Code", "Year", "Total Assets", "Net Assets", "Shareholders' Equity", 
    "Retained Earnings", "Short-Term Loans", "Long-Term Loans", "BPS", 
    "Equity Ratio", "Sales", "Operating Profit", "Ordinary Profit", 
    "Net Profit", "EPS", "ROE", "ROA"
]

# replaceの挙動を将来用に活かす
pd.set_option('future.no_silent_downcasting', True)

# 直近の通期の財務情報を取得
def get_latest_full_year_data(
    code,
    target_date,
):
    # Load financial data from CSV
    finance_path = os.path.join(os.path.dirname(__file__), '../data/finance2021-2024.csv')
    finance_df = pd.read_csv(finance_path)

    # ヘッダに合わせて列名を変更
    finance_df = finance_df[columns_needed]

    # target_date を datetime 型に変換
    target_date = datetime.strptime(target_date, "%Y-%m-%d")

    # "Year" 列を datetime 型に変換（"YYYY/03" → YYYY-03-31）
    finance_df["Year Date"] = finance_df["Year"].apply(
        lambda x: get_last_day_of_month(int(x[:4]), int(x[-2:])) if isinstance(x, str) else np.nan
    )
    # "Year Date" の欠損値を削除
    finance_df = finance_df.dropna(subset=["Year Date"])

    # 指定された銘柄コードのデータを抽出
    finance_df_code = finance_df[finance_df["Code"] == code]

    # target_date より前の最新の年度データを取得
    finance_df_code = finance_df_code[finance_df_code["Year Date"] <= target_date]
    finance_df_code = finance_df_code.sort_values("Year Date", ascending=False).head(1)
    
    return finance_df_code

# ファンダメンタルズ指標を取得
def get_finance_from_csv(
    code,
    target_date,
    target_stock
):
    # 直近の通期財務情報を取得
    finance_df_code = get_latest_full_year_data(
        code,
        target_date
    )

    # データが存在しない場合の処理
    if finance_df_code.empty:
        print(f"No matching data found: {code} before {target_date}")
        return None

    # "-" を NaN に置換
    finance_df_code = finance_df_code.replace("-", np.nan).infer_objects(copy=False)

    # データ型を適切に変換（数値データに変換）
    for col in [
        "Total Assets", "Net Assets", "Shareholders' Equity", "Net Profit", 
        "BPS", "EPS", "ROE", "ROA"
    ]:
        finance_df_code[col] = pd.to_numeric(finance_df_code[col], errors='coerce')

    # 発行済株式数を計算（Shareholders' Equity ÷ BPS）
    finance_df_code["Outstanding Shares"] = finance_df_code["Shareholders' Equity"] / finance_df_code["BPS"]

    # EPS（1株当たり利益）が欠損している場合、計算して補完
    finance_df_code["EPS"] = finance_df_code.apply(
        lambda row: row["Net Profit"] / row["Outstanding Shares"] if pd.isna(row["EPS"]) and row["Outstanding Shares"] > 0 else row["EPS"], axis=1
    )

    # ROE（自己資本利益率）が欠損している場合、計算して補完
    finance_df_code["ROE"] = finance_df_code.apply(
        lambda row: (row["Net Profit"] / row["Shareholders' Equity"]) * 100 if pd.isna(row["ROE"]) and row["Shareholders' Equity"] > 0 else row["ROE"], axis=1
    )

    # ROA（総資産利益率）が欠損している場合、計算して補完
    finance_df_code["ROA"] = finance_df_code.apply(
        lambda row: (row["Net Profit"] / row["Total Assets"]) * 100 if pd.isna(row["ROA"]) and row["Total Assets"] > 0 else row["ROA"], axis=1
    )

    # BPS（1株当たり純資産）が欠損している場合、計算して補完
    finance_df_code["BPS"] = finance_df_code.apply(
        lambda row: row["Shareholders' Equity"] / row["Outstanding Shares"] if pd.isna(row["BPS"]) and row["Outstanding Shares"] > 0 else row["BPS"], axis=1
    )

    # 株価データを追加
    finance_df_code["Stock Price"] = target_stock
    #print(f'{code}: Stock Price → {target_stock}  Type → {target_stock.dtype}')

    # PER（株価収益率）を計算（Stock Price ÷ EPS）
    finance_df_code["PER"] = finance_df_code.apply(
        lambda row: row["Stock Price"] / row["EPS"] if pd.notna(row["Stock Price"]) and pd.notna(row["EPS"]) and row["EPS"] > 0 else np.nan, axis=1
    )

    # PBR（株価純資産倍率）を計算（Stock Price ÷ BPS）
    finance_df_code["PBR"] = finance_df_code.apply(
        lambda row: row["Stock Price"] / row["BPS"] if pd.notna(row["Stock Price"]) and pd.notna(row["BPS"]) and row["BPS"] > 0 else np.nan, axis=1
    )

    # 時価総額を計算（Stock Price × Outstanding Shares）
    finance_df_code["Market Capitalization"] = finance_df_code.apply(
        lambda row: row["Stock Price"] * row["Outstanding Shares"] if pd.notna(row["Stock Price"]) and pd.notna(row["Outstanding Shares"]) else np.nan, axis=1
    )
    finance_df_code["Market Capitalization Log"] = np.log1p(finance_df_code["Market Capitalization"])

    # 結果を表示
    finance_needed = finance_df_code[[
        "EPS", "ROE", "PER", "PBR", "Market Capitalization Log"
    ]]

    return finance_needed.to_dict(orient='records')[0]

# 時価総額のみ取得する
def get_financecapitalization_from_csv(
    code,
    target_date,
    target_stock
):
    # 直近の通期財務情報を取得
    finance_df_code = get_latest_full_year_data(
        code,
        target_date
    )

    # データが存在しない場合の処理
    if finance_df_code.empty:
        print(f"No matching data found: {code} before {target_date}")
        return None

    # "-" を NaN に置換
    finance_df_code = finance_df_code.replace("-", np.nan).infer_objects(copy=False)

    # データ型を適切に変換（数値データに変換）
    for col in [
        "Shareholders' Equity", "BPS"
    ]:
        finance_df_code[col] = pd.to_numeric(finance_df_code[col], errors='coerce')

    # 発行済株式数を計算（Shareholders' Equity ÷ BPS）
    finance_df_code["Outstanding Shares"] = finance_df_code["Shareholders' Equity"] / finance_df_code["BPS"]

    # 株価データを追加
    finance_df_code["Stock Price"] = target_stock
    #print(f'{code}: Stock Price → {target_stock}  Type → {target_stock.dtype}')

    # 時価総額を計算（Stock Price × Outstanding Shares）
    finance_df_code["Market Capitalization"] = finance_df_code.apply(
        lambda row: row["Stock Price"] * row["Outstanding Shares"] if pd.notna(row["Stock Price"]) and pd.notna(row["Outstanding Shares"]) else np.nan, axis=1
    )

    # 結果を表示
    finance_needed = finance_df_code[[
        "Stock Price","Market Capitalization"
    ]]

    return finance_needed.to_dict(orient='records')[0]

# 指定された年と月の月末を返す
def get_last_day_of_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, last_day)
