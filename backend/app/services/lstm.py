# LSTMを使って学習
# LSTM（Long Short-Term Memory）は、**時系列データや自然言語処理（NLP）などの
# シーケンシャルなデータを扱うための特殊なRNN（再帰型ニューラルネットワーク）**の一種
# LSTMは、情報を長期間記憶できるように設計されたRNNで、特に長期的な依存関係を持つデータを扱うのが得意

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from . import googleapi, summarize

# データ読み込み (例としてjson読み込みの方法)
import json
import os

gcs_model_torch_path = os.getenv('GCS_MODEL_TORCH_PATH', '')

# LSTMの学習　メイン処理
# 呼び出し元より
# documents: 開示リスト(この中で特徴量に変える)
# features：開示から3か月の各指標特徴量
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
def lstm_learning(
    documents,
    features,
    targets
    ):
    
    # 文章を特徴量に変換
    x_text = [summarize.get_text_embeddings(document) for document in documents]
    
    # データセットとデータローダの作成
    dataset = CustomDataset(x_text, features, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # モデルのロード (学習済みモデルをロードする場合)
    try:
        model = googleapi.load_model_torch(gcs_model_torch_path)
        print("モデルをロードしました。")
    except Exception as e:
        print("モデルのロードに失敗しました。新しいモデルを初期化します。")
        # モデルの初期化
        # サイズを取得するためfeaturesを2次元配列にしてから 開示の特徴量は100で　指標は20になるはず
        features_tensor = [torch.tensor(list(item.values()), dtype=torch.float32) for item in features]
        features_array = torch.stack(features_tensor)
        print(f'開示の特徴量→{x_text.shape[1]}')
        print(f'指標の特徴量→{features_array.shape[1]}')
        model = LSTMModel(input_text_size=x_text.shape[1], input_features_size=features_array.shape[1])
        
    # ロス関数とオプティマイザの設定
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # モデルの学習または予測
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_text_batch, features_batch, targets_batch in dataloader:
            optimizer.zero_grad()
            
            # フォワードパス
            outputs = model(x_text_batch, features_batch)
            
            # ロス計算
            loss = criterion(outputs.squeeze(), targets_batch)
            loss.backward()
            optimizer.step()    # ここがモデルのパラメータ(重み)更新　これが積み重ね？
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # モデルの保存
    googleapi.upload_model(model, gcs_model_torch_path)


# PyTorch用のデータセットクラス
class CustomDataset(torch.utils.data.Dataset):
    
    FEATURE_KEYS = [
        'EPS', 'ROE', 'PER', 'PBR', 'Market Capitalization Log', 'NikkeiCorr', 'MothersCorr',
        'RSI', 'MovingAverage50', 'MovingAverage200', 'ATR', 'MACD', 'Signal',
        'UpperBand', 'LowerBand', 'PercentR', 'ADX'
    ]
    
    TARGET_KEYS = [
        'ChangeRate_0', 'ChangeRate_3', 'ChangeRate_7', 'ChangeRate_14', 'ChangeRate_21',
        'ChangeRate_28', 'ChangeRate_35', 'ChangeRate_42', 'ChangeRate_49'
    ]
    
    def __init__(self, x_text_tfidf, features, targets):
        self.x_text_tfidf = torch.tensor(x_text_tfidf, dtype=torch.float32)
        self.features = [torch.tensor([target[key] for key in self.FEATURE_KEYS], dtype=torch.float32) for target in features]
        self.targets = [torch.tensor([target[key] for key in self.TARGET_KEYS], dtype=torch.float32) for target in targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.x_text_tfidf[idx], self.features[idx], self.targets[idx]

# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_text_size, input_features_size):
        super(LSTMModel, self).__init__()
        
        # x_textの処理 (Dense層に相当する部分)
        self.text_fc1 = nn.Linear(input_text_size, 128)
        self.text_fc2 = nn.Linear(128, 64)
        
        # featuresの処理
        self.features_fc = nn.Linear(input_features_size, 32)
        
        # LSTM層（影響を抑えるため少なめ）
        self.lstm = nn.LSTM(input_features_size, 64, batch_first=True)
        
        # 統合後の出力層
        self.fc_out = nn.Linear(64 + 64 + 32, 1)  # 64 + 64 (LSTM + Dense) + 32 (features)

    def forward(self, x_text, features):
        # x_textの処理
        x_text = torch.relu(self.text_fc1(x_text))
        x_text = torch.relu(self.text_fc2(x_text))
        
        # featuresの処理
        features_out = torch.relu(self.features_fc(features))
        
        # LSTM処理 (featuresの次元を調整してLSTMに入力)
        lstm_out, _ = self.lstm(features)  # featuresの次元が適切ならunsqueezeは不要
        lstm_out = lstm_out[:, -1, :]  # 最後のタイムステップの出力を使用
        
        # 統合
        combined = torch.cat((x_text, lstm_out, features_out), dim=1)
        
        # 出力
        output = torch.sigmoid(self.fc_out(combined))
        return output

