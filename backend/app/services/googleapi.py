import os
import json
from dotenv import load_dotenv
from google.cloud import storage
import pandas as pd
from io import BytesIO
import joblib
import torch
from ..models import mlp
import torch.serialization

# MLPModel を安全なグローバルとして追加
torch.serialization.add_safe_globals([mlp.MLPModel])

# 環境変数から認証情報を取得
google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
bucket_name = os.getenv('GCS_BUCKET_NAME', '')

# 対象日の開示と株価データをCSVとしてバケット内に保存
def upload_list(
    df,
    brob_name
):
    # Google Cloud Storage クライアントを作成
    storage_client = storage.Client.from_service_account_json(google_creds_path)
    # GCSクライアントを作成
    client = storage.Client()

    # バケットを取得
    bucket = client.get_bucket(bucket_name)
    
    # DataFrameをCSV形式に変換してメモリに書き込む
    json_data = df.to_json(orient='records', lines=False, force_ascii=False, date_format='iso')

    # バケットにアップロード
    blob = bucket.blob(brob_name)
    
    if blob.exists():
        # ファイルが存在する場合、その内容を読み込み
        existing_data = blob.download_as_text()

        # データが空でない場合に追記
        if existing_data.strip():  # 既存データが空でない場合
            # 既存データをJSONとしてパースし、リストに変換
            existing_json = json.loads(existing_data)
            new_json = json.loads(json_data)
            
            # 新規データを既存リストに追加
            existing_json.extend(new_json)
            
            # CodeとTitleの組み合わせをキーとして重複を削除
            unique_data = { (item["Code"], item["Title"]): item for item in existing_json }.values()
            
            result = list(unique_data)
            
            # json形式へ
            update_data = json.dumps(result, ensure_ascii=False)
        else:
            update_data = json_data  # 空の場合、データはそのまま
    else:
        # ファイルが存在しない場合、新規作成
        update_data = json_data
        
    # jsonの内容をバイト列にエンコード
    json_data_bytes = update_data.encode('utf-8')
    
    blob.upload_from_file(BytesIO(json_data_bytes), content_type='application/json', timeout=300)
    
# 対象日の開示と株価データをCSVとしてバケット内に保存
def rewrite_list(
    df,
    brob_name
):
    # Google Cloud Storage クライアントを作成
    storage_client = storage.Client.from_service_account_json(google_creds_path)
    # GCSクライアントを作成
    client = storage.Client()

    # バケットを取得
    bucket = client.get_bucket(bucket_name)
    
    # DataFrameをCSV形式に変換してメモリに書き込む
    json_data = df.to_json(orient='records', lines=False, force_ascii=False, date_format='iso')

    # バケットにアップロード
    blob = bucket.blob(brob_name)
    
    # 存在するファイル内に上書き
    update_data = json_data
        
    # jsonの内容をバイト列にエンコード
    json_data_bytes = update_data.encode('utf-8')
    
    blob.upload_from_file(BytesIO(json_data_bytes), content_type='application/json', timeout=300)
    
# 一つのファイルをjson形式で取得
def download_list(
    blob_name
):
    # Google Cloud Storage クライアントを作成
    storage_client = storage.Client.from_service_account_json(google_creds_path)
    # GCSクライアントを作成
    client = storage.Client()
    # バケットを作成
    bucket = client.get_bucket(bucket_name)
    # Blobを取得
    blob = bucket.get_blob(blob_name)
    
    return json.loads(blob.download_as_text()) if blob and blob.exists() else {}

# 学習モデルの保存(torch LSTM)
def upload_model_torch(
    model,
    blob_name
):
    
    # modelをバイトストリームに保存
    model_bytes = BytesIO()
    torch.save(model, model_bytes)
    model_bytes.seek(0)
    
    # Google Cloud Storage クライアントを作成
    storage_client = storage.Client.from_service_account_json(google_creds_path)
    
    # バケット取得
    bucket = storage_client.get_bucket(bucket_name)
    
    # Blob検索
    blob = bucket.blob(blob_name)
    
    blob.upload_from_file(model_bytes, content_type='application/octet-stream', timeout=300)
    
# 学習モデルの読み込み(torch LSTM)
def load_model_torch(
    blob_name
):
    # Google Cloud Storage クライアントを作成
    storage_client = storage.Client.from_service_account_json(google_creds_path)
    
    # バケット取得
    bucket = storage_client.get_bucket(bucket_name)
    
    # Blob検索
    blob = bucket.get_blob(blob_name)
    
    # Blobの内容をバイトストリームとして読み込み
    model_bytes = blob.download_as_bytes()
    
    # メモリ内でモデルを復元
    return torch.load(BytesIO(model_bytes), weights_only=False)