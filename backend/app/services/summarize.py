# 作業用　こっちでブラッシュアップして本体にもっていく

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch.nn.functional as F
import networkx as nx
import unicodedata
from . import cleantext

# モデルとトークナイザーを設定
model_name = 'cl-tohoku/bert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 重要単語リスト
important_keywords_path = os.path.join(os.path.dirname(__file__), '../data/important_keywords.csv')
important_keywords_df = pd.read_csv(important_keywords_path, header=None)
important_keywords = important_keywords_df.iloc[:,0].to_list()

# 重要度低いリスト
low_priority_keywords_path = os.path.join(os.path.dirname(__file__), '../data/low_priority_keywords.csv')
low_priority_keywords_df = pd.read_csv(low_priority_keywords_path, header=None)
low_priority_keywords = low_priority_keywords_df.iloc[:,0].to_list()

# 重要度高めリスト
high_priority_words_path = os.path.join(os.path.dirname(__file__), '../data/high_priority_keywords.csv')
high_priority_words_df = pd.read_csv(high_priority_words_path, header=None)
high_priority_words = high_priority_words_df.iloc[:, 0].to_list()

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

# 特徴量変換を同時起動
def embed_in_parallel(documents, max_workers=3):
    
    result = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_documents = [executor.submit(get_text_embeddings, document) for document in documents]
        
        for future in future_documents:
            try:
                result.append(future.result())
            except Exception as e:
                result.append(f'error:{e}')
    
    return np.array(result)

# ブラッシュアップ文章処理を並列化
def brush_in_parallel(documents, max_workers=3):
    result = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_documents = [executor.submit(brushup_text, document) for document in documents]
        
        for future in future_documents:
            try:
                result.append(future.result())
            except Exception as e:
                print(f'error:{e}')
    
    return result


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

# 特徴量を取得(model学習用)
def get_text_embeddings(text, max_token_length=512, batch_size=32):
    # トークン化（特殊トークンを加える）
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # トークン化後のトークン数を確認
    print(f"トークン数: {len(tokens)}")  # 実際のトークン数を確認
    if len(tokens) > max_token_length:
        print("警告: トークン数が最大長を超えています！")
    
    # トークン数が512を超えていた場合、512トークンずつ分割する
    chunks = []
    while len(tokens) > max_token_length:
        chunk = tokens[:max_token_length]
        chunks.append(chunk)
        tokens = tokens[max_token_length:]  # 残りを次に

    if len(tokens) > 0:
        chunks.append(tokens)  # 残りを最後のチャンクとして追加

    # チャンク数を確認
    print(f"分割後のチャンク数: {len(chunks)}")
    
    embeddings = []

    # 各チャンクをバッチ処理
    for chunk in chunks:
        # 入力の準備（バッチ次元追加）
        input_ids = torch.tensor(chunk).unsqueeze(0).to(device)  # バッチ次元追加
        attention_mask = torch.ones(input_ids.shape, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # チャンクごとの埋め込みを計算
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(chunk_embedding)

    # 全チャンクの埋め込みを平均化
    final_embedding = np.mean(np.concatenate(embeddings, axis=0), axis=0)

    return final_embedding


# janomeを使用して文をある程度に分割
def split_sentences_with_janome(text):
    tokenizer = Tokenizer()
    sentences = []
    sentence = []
    
    split = ['。', '！', '!', '？']
    
    # 括弧内フラグ
    inside_parentheses = False

    for token in tokenizer.tokenize(text):
        token_lower = token.surface.lower()  # 小文字変換
        current_sentence = ''.join(sentence).lower()  # 現在の文の小文字変換
        
        # 括弧の判定
        if re.search(r'[（(]', token.surface):
            inside_parentheses = True
        if re.search(r'[）)]', token.surface):
            inside_parentheses = False
        
        # 括弧内でない場合のみ分割トリガー
        if not inside_parentheses and ( 
            token_lower in split or
            any( s in token_lower for s in split)
        ):
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

# 要約メイン処理
def summarize_long_document(document, max_token_length=512, stride=256, summarize_limit=2000):
    #print(f'{document[:100]}の開始')
    
    """
    k-meansクラスタリングを使った文書要約
    """
    
    # 文書をクリーンアップ
    document = cleantext.clean_text(document)
    
    # 文の分割
    sentences = split_sentences_with_janome(document)
    
    if not sentences:
        return ""
    
    if len(document) <= summarize_limit:
        print(f'{summarize_limit}文字以内のため低クオリティだけ省く')
        non_low_priority_sentences = [
           sentence for sentence in sentences
           if not any(keywords in sentence for keywords in low_priority_keywords)
        ]
        return ''.join(non_low_priority_sentences)
    
    # 1. BERTの文埋め込みを取得
    sentence_embeddings = get_sentence_embeddings(sentences, model, tokenizer, device, max_token_length)
    
    # 2. コサイン類似度行列を計算
    sentence_embeddings_torch = F.normalize(torch.tensor(sentence_embeddings, dtype=torch.float32, device=device), dim=1)
    similarity_matrix = (sentence_embeddings_torch @ sentence_embeddings_torch.T).cpu().numpy()
    
    # 3. グラフを作成し、PageRank（TextRank）を適用
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # 4. 重要度と高いと低いで重みづけ
    ranked_sentences = sorted(
        ((scores[i] 
        + (100 if any(keyword in s for keyword in important_keywords) else 0)  # 重要文なら+100
        + (50 if any(keyword in s for keyword in high_priority_words) else 0)  # 重要度高め+50
        - (100 if any(keyword in s for keyword in low_priority_keywords) else 0), s)  # 低優先なら-100
        for i, s in enumerate(sentences)), 
        reverse=True
    )
    
    # 最大文字数までいれていく
    total_length = 0
    summary_sentences = []
    for score, sentence in ranked_sentences:
        if score < 0:    #優先度がマイナスになってるのは飛ばす(つまり登録しない)
            continue
        
        if summarize_limit < len(sentence): # 一つでマックス超えるのは飛ばす
            continue
        
        if summarize_limit < total_length + len(sentence):
            break
        
        summary_sentences.append(sentence)
        total_length += len(sentence)
        
    summary = ''.join(summary_sentences)
    
    return summary

# 要約ではなく文章から無駄な箇所を省く
def brushup_text(
    document
    ):
    
    # 文書をクリーンアップ
    document = cleantext.clean_text(document)
    
    # 文の分割
    sentences = split_sentences_with_janome(document)
    
    if not sentences:
        return ""
    
    # 不要グループを省いていく
    test = [
        low for low in sentences
        if cleantext.is_exclude_calendar(low)
    ]
    non_low_priority_sentences = [
        sentence for sentence in sentences
        if not any(keywords in sentence for keywords in low_priority_keywords) and  # 登録された不要グループ
        not cleantext.is_exclude_calendar(sentence) # カレンダーっぽい数字の羅列
    ]
    
    # # 元のドキュメントと要約候補たち
    # doc_original = "連結P/L 13 ホビーサーチ事業の成長等に伴い、連結売上総利益率は低下 単位:百万円 22/3期2Q 23/3期2Q 科目 金額 売上比 金額 売上比 前年同期比 主な要因 売上高 2,004 100.0_% 3,427 100.0_% 171.0_% ホビーサーチM&Aに伴う増収 売上総利益 1,284 64.1_% 1,644 48.0_% 128.0_% 低売上総利益率のホビー商材拡大による率低下 販売費及び 1,232 61.5_% 1,408 41.1_% 114.3_% ホビーサーチの子会社化に伴う費用増加 一般管理費 営業利益 52 2.6_% 236 6.9_% 450.2_% - 経常利益 49 2.5_% 226 6.6_% 454.0_% - 平塚梅屋事業所撤退等に伴う受取補償金45_百万円 四半期純利益 11 0.6_% 171 5.0_% - の計上 。"
    # summary_1 = "ホビーサーチのM&Aにより、売上高は2,004百万円 → 3,427百万円に増加。 だが、売上総利益率は64.1% → 48.0%と低下。これは利益率の低いホビー商材の拡大が原因だおｗｗｗ"
    # summary_2 = "売上は前年比171%増と爆増ｗｗｗだが、ホビー商材の比率上昇で売上総利益率は縮小。子会社化によって販管費も増えたけど、営業利益は52 → 236百万円（約4.5倍）に成長！"
    # summary_3 = "ホビーサーチM&Aで売上総利益：1,284 → 1,644百万円に増加。一方で利益率はダウン…。経常利益は49 → 226百万円に激増＆**平塚梅屋の撤退補償金（45百万円）**も計上されてるおｗｗｗ"


    # # ベクトルに変換
    # docs = [doc_original, summary_1, summary_2, summary_3]
    # embeddings = [test_embedding(d) for d in docs]

    # # コサイン類似度で元文との近さを測定
    # similarities = cosine_similarity([embeddings[0]], embeddings[1:])
    # for i, sim in enumerate(similarities[0]):
    #     print(f"要約{i+1}との類似度: {sim:.4f}")
        
    return ''.join(non_low_priority_sentences)

# 特徴量の確認に使用する確認用
def test_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] トークンのベクトルを使用
    return cls_embedding.squeeze().numpy()