import torch.nn as nn
import torch

# 回帰学習モデル(MLP)
class MLPModel(nn.Module):
    def __init__(self, input_text_size=768, input_features_size=17):
        super().__init__()
        # テキスト特徴量（BERT等の出力）を処理する層を定義
        self.text_fc = nn.Sequential(
            # 入力の特徴量次元（input_text_size）から128次元に変換する線形層
            nn.Linear(input_text_size, 128),    # ここでBERT出力の768次元などを128次元に圧縮
            # 非線形性を導入して、学習の幅を広げる
            nn.ReLU(),  # ReLU関数で負の値を0にし、ネットワークに非線形性を持たせる
            # 128次元から64次元にさらに圧縮する線形層
            nn.Linear(128, 64),  # 128次元から64次元に圧縮して、より抽象的な特徴を捉える
        )

        # 入力された指標（EPSやROEなど）を処理する層を定義
        self.features_fc = nn.Sequential(
            # 特徴量（input_features_size）を32次元に圧縮する線形層
            nn.Linear(input_features_size, 32),  # 入力された指標を32次元に圧縮
            # 非線形性を導入して学習を効率的にする
            nn.ReLU(),  # ReLU関数を適用して非線形な変換を加える
        )

        self.output_layer = nn.Linear(64 + 32, 9)  # 最終的に9つの変化率出す

    def forward(self, x_text, x_features):
        text_out = self.text_fc(x_text)     # 開示文のエンベディング処理！
        features_out = self.features_fc(x_features) # 数値指標の処理！
        concat = torch.cat((text_out, features_out), dim=1) # 結合ッ…！！
        return self.output_layer(concat)    # 9個の未来変化率を出力ッ！
    
    
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