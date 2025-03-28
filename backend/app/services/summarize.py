import re
import torch
import time
import numpy as np
import pandas as pd
import os
from transformers import BertJapaneseTokenizer, BertModel
from janome.tokenizer import Tokenizer
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import networkx as nx
from ..models import header_replace

# モデルとトークナイザーを設定
model_name = 'cl-tohoku/bert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 。に変換する区切りワードを読み込み
replace_list_path = os.path.join(os.path.dirname(__file__), '../data/summarize_replace.csv')
replace_list_df = pd.read_csv(replace_list_path, header=None)
replace_list = replace_list_df.iloc[:,0].to_list()

#省くフッターりすと
replace_footer_list_path = os.path.join(os.path.dirname(__file__), '../data/summarize_footer_replace.csv')
replace_footer_list_df = pd.read_csv(replace_footer_list_path, header=None)
replace_footer_list = replace_footer_list_df.iloc[:,0].to_list()

# 重要単語リスト
important_keywords_list_path = os.path.join(os.path.dirname(__file__), '../data/important_keywords.csv')
important_keywords_list_df = pd.read_csv(important_keywords_list_path, header=None)
important_keywords = important_keywords_list_df.iloc[:,0].to_list()

# 重要度低いリスト
low_priority_keywords_list_path = os.path.join(os.path.dirname(__file__), '../data/low_priority_keywords.csv')
low_priority_keywords_list_df = pd.read_csv(low_priority_keywords_list_path, header=None)
low_priority_keywords = low_priority_keywords_list_df.iloc[:,0].to_list()

# 並列で要約処理を回す
def summarize_in_parallel(documents, max_workers=5):

    start_time = time.time()
    
    results = []
    # プロセスごとに処理を並列化
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_document = {executor.submit(summarize_long_document, document): document for document in documents}
        for future in future_document:
            try:
                results.append(future.result())  # 結果をリストに追加
            except Exception as e:
                results.append(f'error:{e}')
                
    dif_time = time.time() - start_time
    minutes = int(dif_time // 60)
    seconds = dif_time % 60
    print(f'summarize_in_parallel:{minutes}分{seconds}秒かかった')
    
    return results

# ヘッダー部分を削除
def clean_header(text):
    
    # 不要なブランクを削除
    text = re.sub(r'\s+', '', text).strip()
    
    # 各パターンで処理
    for pattern, label in header_replace.PATTERNS:
        match = re.search(pattern, text)
        if match and match.start() <= 200:
            # マッチした部分までを削除
            text = re.sub(r'^.*?' + pattern, '', text, count=1)

    return text

# footer部分を削除
def clean_footer(text):
    
    # すべての "以上" の位置を取得
    matches = list(re.finditer('|'.join(map(re.escape, replace_footer_list)), text))
    
    if matches:
        last_match = matches[-1]  # 最後の "以上" を取得
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()] if last_match.start() > 0 else text  # 最後の "以上" の直前までを取得

    return text

# 開示文章内から不要な文章を削除
def clean_text(text):
    
    # ヘッダー部分を削除
    text = clean_header(text)
    
    # フッター部分を削除
    text = clean_footer(text)
    
    # summarize_replace.csvに登録されてるものを。に置き換える(区切り文字として扱う)
    text = re.sub('|'.join(map(re.escape, replace_list)), '', text)

    # 最終クリーン
    text = re.sub(r'。+', '。', text)
    text = re.sub(r'\(\)|（）|[\(\)]{1}', '', text)
    text = re.sub(r'\s+', '', text).strip()
    text = re.sub(r'^（代表）', '', text)
    text = re.sub(r'^[）\)]', '', text)
    text = text.strip()

    return text

# 文ごとのBERT埋め込みをバッチ処理で取得（mean pooling を使用）
def get_sentence_embeddings(sentences, model, tokenizer, device, max_token_length=512, batch_size=32):
    embeddings = []

    # バッチ処理で文を分ける
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]

        # トークナイズを一括で処理
        inputs = tokenizer(batch_sentences, return_tensors="pt", truncation=True, 
                           max_length=max_token_length, padding="longest")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # mean pooling
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

    # すべてのバッチの結果を結合
    return np.concatenate(embeddings, axis=0)  # 直接2D numpy配列を返す

# janomeを使用して文をある程度に分割
def split_sentences_with_janome(text):
    tokenizer = Tokenizer()
    sentences = []
    sentence = []
    
    # 括弧内フラグ
    inside_parentheses = 0  # 0: 外、1: ( または （ の中、2: ) または ）の中

    for token in tokenizer.tokenize(text):
        token_lower = token.surface.lower()  # 小文字変換
        current_sentence = ''.join(sentence).lower()  # 現在の文の小文字変換
        
        # 括弧の判定
        if re.search(r'[（(]', token.surface):
            inside_parentheses += 1
        elif re.search(r'[）)]', token.surface):
            inside_parentheses -= 1
        
        # 括弧内でない場合のみ分割トリガー
        if token_lower in ['。']:
            test = ''
            
        if inside_parentheses == 0 and token_lower in ['。', '！', '？']:
            sentence.append(token.surface)

            # ここまでを確定
            append_sentence = ''.join(sentence).strip()
            # 20文字より多いもののみ
            if (len(append_sentence) > 20):
                sentences.append(append_sentence)
                
            sentence = []
        else:
            sentence.append(token.surface)
    
    if sentence:
        append_sentence = ''.join(sentence).strip()
        if len(append_sentence) > 20:
            sentences.append(append_sentence)  # 最後の文を追加
    
    return sentences

# 要約文字数調整
def adjust_summary_length(summary, sentences, target_length=2000):
    # もしsummaryが短すぎる場合、追加の文章を連結
    if len(summary) < target_length:
        remaining_sentences = [s for s in sentences if s not in summary]
        for s in remaining_sentences:
            if len(summary) + len(s) <= target_length:
                summary += s
            else:
                break

    # もしsummaryが長すぎる場合、カット
    summary = summary[:target_length]

    return summary

# 要約メイン処理
def summarize_long_document(document, num_sentences=5, max_token_length=512, stride=256):
    #print(f'{document[:100]}の開始')
    
    """
    k-meansクラスタリングを使った文書要約
    """
    
    # 文書をクリーンアップ
    document = clean_text(document)
    
    if len(document) <= 2000:
        print('2000文字以内のためそのまま返却')
        return document
    
    # 文の分割
    sentences = split_sentences_with_janome(document)
    
    if not sentences:
        return ""
    
    # 1. BERTの文埋め込みを取得
    sentence_embeddings = get_sentence_embeddings(sentences, model, tokenizer, device, max_token_length)
    
    # 2. コサイン類似度行列を計算
    sentence_embeddings_torch = F.normalize(torch.tensor(sentence_embeddings, dtype=torch.float32, device=device), dim=1)
    similarity_matrix = (sentence_embeddings_torch @ sentence_embeddings_torch.T).cpu().numpy()
    
    # 3. グラフを作成し、PageRank（TextRank）を適用
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # 4. スコアの高い順に文を選択 
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # 重要な文を抽出
    important_sentences = [s for s in sentences if any(keyword in s for keyword in important_keywords)]

    # 目次関連の文を低ランクに調整
    low_priority_sentences = [s for s in sentences if any(keyword in s for keyword in low_priority_keywords)]

    # 重要な文を2つまでに制限
    important_sentences = important_sentences[:2]

    # 重要な文とランキング文の重複を避ける
    important_sentences_set = set(important_sentences)
    non_important_ranked_sentences = [s[1] for s in ranked_sentences if s[1] not in important_sentences_set]

    # summary_sentencesを初期化
    summary_sentences = []  # 空リストとして初期化

    # 重要な文とランキング文の重複を避けた後に、残りの文を選定
    remaining_sentences_count = num_sentences - len(important_sentences)
    remaining_sentences = non_important_ranked_sentences[:remaining_sentences_count]

    # 目次関連の文を最後に追加して、低ランクで加える
    low_priority_sentences = [s for s in low_priority_sentences if s not in summary_sentences]
    summary_sentences += low_priority_sentences

    # 重要な文とランキング文を結合
    summary_sentences = important_sentences + remaining_sentences

    # 重要な文とTextRankの文を元の順番に並べるために、インデックスマップを作成
    sentence_indices = {s: i for i, s in enumerate(sentences)}

    # summary_sentencesの順番を元のsentencesのインデックスに基づいて並べ替え
    summary = ''.join(sorted(summary_sentences, key=lambda s: sentence_indices[s]))

    # 使われていない文を取得してadjust_summary_lengthに渡す
    unused_sentences = [s[1] for s in ranked_sentences if s[1] not in summary_sentences]

    return adjust_summary_length(summary, unused_sentences)