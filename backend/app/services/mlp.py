# MLPを使って学習
# MLP（多層パーセプトロン） は全結合層を積み重ねて、与えられた入力から予測を行うタイプのモデル
# 時系列的な情報を考慮しない
# 主に全結合層（nn.Linear）と活性化関数（ReLUなど）を使って学習する

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from . import googleapi, summarize
from ..models import mlp

# データ読み込み (例としてjson読み込みの方法)
import json
import os

gcs_model_torch_path = os.getenv('GCS_MODEL_TORCH_PATH', '')

# MLPの学習　メイン処理
# 呼び出し元より
# documents: 開示リスト(この中で特徴量に変える)
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
# targets：３日後～７週間の変化率(予測)
#           ChangeRate_0
#           ChangeRate_3
#           ChangeRate_7
#           ChangeRate_14
#           ChangeRate_21
#           ChangeRate_28
#           ChangeRate_35
#           ChangeRate_42
#           ChangeRate_49
# このLISTでくるように
# PyTorchにはこの順番で学習させるため
def mlp_learning(documents, features, targets):
    # 文章を特徴量に変換
    x_text = summarize.embed_in_parallel(documents=documents)

    # データセットとデータローダの作成
    dataset = mlp.CustomDataset(x_text, features, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # MLPモデルのロード
    model = model_load()
    
    if model is None:
        features_tensor = [torch.tensor(list(item.values()), dtype=torch.float32) for item in features]
        features_array = torch.stack(features_tensor)
        print(f'開示の特徴量→{x_text.shape[1]}')
        print(f'指標の特徴量→{features_array.shape[1]}')
        model = mlp.MLPModel(input_text_size=x_text.shape[1], input_features_size=features_array.shape[1])

    # ロス関数とオプティマイザの設定
    criterion = nn.MSELoss()  # 回帰タスクならMSELossを使用
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # モデルの学習
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_text_batch, features_batch, targets_batch in dataloader:
            optimizer.zero_grad()
            
            # フォワードパス
            outputs = model(x_text_batch, features_batch)
            
            # ロス計算
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            optimizer.step()  # パラメータの更新
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # モデルの保存
    try:
        googleapi.upload_model_torch(model, gcs_model_torch_path)
        print(f"モデルが{gcs_model_torch_path}に保存されました！")
    except Exception as e:
        print(f"モデル保存に失敗しました。エラーメッセージ: {str(e)}")
     
# 保存MLPモデルを取得
# 予測を取得する際は共通的に一回読んで使いまわすように
def model_load():
    model = None
    
    try:
        model = googleapi.load_model_torch(gcs_model_torch_path)
        print("モデルをロードしました。")
    except Exception as e:
        print(f"モデルのロードに失敗しました。エラーメッセージ: {str(e)}")

    return model

# 保存MLPモデルを削除
def model_delete():
    googleapi.delete_data(gcs_model_torch_path)

# MLPモデルから株価予測を取得
def targets_from_model(
    model,
    document,   # 新規開示文章
    feature     # 開示時点の指標
):
    
    # 開示文章を特徴量に変換
    x_text = summarize.get_text_embeddings(document)
    x_text_tensor = torch.tensor(x_text, dtype=torch.float32).unsqueeze(0)
    
    # 指標を特徴量に変換
    # unsqueeze(0)バッチサイズ1のテンソルに変換
    feature_tensor = torch.tensor(list(feature.values()), dtype=torch.float32).unsqueeze(0)
    
    print(f'開示の特徴量→{x_text_tensor.shape[1]}')
    
    # modelを評価モードに
    # これによって結果を得るようの動きになる
    model.eval()
    
    # 勾配計算は不要なので、no_gradで無効化
    with torch.no_grad():
        # 予測を取得
        targets = model(x_text_tensor, feature_tensor)
        
    predictions = [t.item() for t in targets[0]]
    predictions_dic = dict(zip(mlp.CustomDataset.TARGET_KEYS, predictions))
        
    return predictions_dic
    
    



