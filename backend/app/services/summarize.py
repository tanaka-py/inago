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
from ..models import headerfooter_replace

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

# ヘッダー部分を削除
def clean_header(text):
    
    is_delete = False
    pattern = '|'.join(pattern[0] for pattern in headerfooter_replace.PATTERNS_HEADER)
    match = re.search(pattern, text)
    if match and match.start() <= 300:
        # マッチした部分までを削除
        #text = re.sub(r'^.*?' + pattern, '', text, count=1)
        text = re.sub(r'^.*?(' + pattern + ')', '', text, count=1)
        is_delete = True
        
    # 不要なブランクを削除
    text = re.sub(r'\s+', '', text).strip()
    
    if not is_delete:
        # TELなしでも、3ブロック以上のやつだけ
        pattern = r'''
            (?<!\d)
            (
                0\d{1,4}
                [-ー−–（）\s]*       # ← ここに en dash 入れたよ！！
                \d{2,4}
                [-ー−–（）\s]*
                \d{3,4}
            )
            (?!\d)
        '''

        # matchオブジェクトが欲しいので finditer を使うお
        for match in re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE):
            digits = re.sub(r'\D', '', match.group())
            if len(digits) >= 10 and match.start() < 200:
                # 有効な電話番号っぽい！そこまでズバーンと削除ｗｗｗ
                text =  text[match.end():]
                break

    return text

# footer部分を削除
def clean_footer(text):
    
    # すべての "以上" の位置を取得
    matches = list(re.finditer(r'以上', text))
    
    if matches:
        last_match = matches[-1]  # 最後の "以上" を取得
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    # 以上を削除したあとまだあるフッター要素を削除
    patterns = [pattern for pattern, _ in headerfooter_replace.PATTERNS_FOTTER]
    matches = list(re.finditer('|'.join(map(re.escape, patterns)), text))

    if matches:
        last_match = matches[-1]
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    return text

# 開示文章内から不要な文章を削除
def clean_text(text):
    
    # ヘッダー部分を削除
    text = clean_header(text)
    
    # フッター部分を削除
    text = clean_footer(text)
    
    # summarize_replace.csvに登録されてるものを。に置き換える(区切り文字として扱う)
    text = re.sub('|'.join(map(re.escape, replace_list)), '。', text)
    
    # 連続するハイフン（ーまたは-）を1つにする
    text = re.sub(r'[ー\―\-]{2,}', '―', text)

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
        if token_lower in ['。']:
            test = ''
            
        if not inside_parentheses and token_lower in ['。', '！', '？']:
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
    document = clean_text(document)
    
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
    document = clean_text(document)
    
    # 文の分割
    sentences = split_sentences_with_janome(document)
    
    if not sentences:
        return ""
    
    # 低クオリティだけ省く
    non_low_priority_sentences = [
        sentence for sentence in sentences
        if not any(keywords in sentence for keywords in low_priority_keywords)
    ]
        
    return ''.join(non_low_priority_sentences)