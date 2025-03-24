import re
import torch
import time
import numpy as np
import os
from transformers import BertJapaneseTokenizer, BertModel
from janome.tokenizer import Tokenizer
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import networkx as nx

# モデルとトークナイザーを設定
model_name = 'cl-tohoku/bert-base-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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

def clean_text(text):
    
    # URLまでを削除
    match = re.search(r'https?://[^\s()<>]+(?:\([^\)]*\)|[^\s`!()\[\]{};:.,?–_])', text)
    if match and match.start() <= 200:
        # URLまでを削除
        text = re.sub(r'^.*?https?://[^\s()<>]+(?:\([^\)]*\)|[^\s`!()\[\]{};:.,?–_])', '', text, count=1)
        
    match = re.search(r'@[^@\s]+\.[a-z]+(?:\.[a-z]+)?', text)
    if match and match.start() <= 200:
        text = re.sub(r'^.*?(@[^@\s]+\.[a-z]+(?:\.[a-z]+)?)', '', text, count=1)
    
    text = re.sub(r'\s+', '', text).strip()
    
    # 電話番号ヘッダまでを削除
    match = re.search(r'(TEL|ＴＥＬ|T E L|電話番号|電話|電 話)', text)
    if match and match.start() <= 200:
        text = re.sub(r'^.*?(TEL|ＴＥＬ|T E L|電話番号|電話|電 話)', '', text, count=1)
    
    # 電話番号までを削除
    match = re.search(r'(\（?[\d０-９]{2,5}\）?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}|\bTEL\b[\s\-−－]?\(?[\d０-９]{2,5}\)?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4})', text)
    if match and match.start() <= 200:
        text = re.sub(r'^.*?(\（?[\d０-９]{2,5}\）?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}|\bTEL\b[\s\-−－]?\(?[\d０-９]{2,5}\)?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4})', '', text, count=1)
        
    # 証券会社までを削除
    match = re.search(r'(東証|名証|札証|大証|福証)', text)
    if match and match.start() <= 200:
        text = re.sub(r'^.*?(東証|名証|札証|大証|福証)', '', text, count=1)
        
    # all right reserved系は全部消す
    pattern = r'[\s\W]*?(Ｒｅｓｅｒｖｅｄ|ｒｅｓｅｒｖｅｄ|ＲＩＧＨＴ|ｒｉｇｈｔ|ＲＩＧＨＴＳ|ｒｉｇｈｔｓ|' \
          r'Ａｌｌ|ａｌｌ|Ｌｔｄ．|ｌｔｄ．|ＣＯ．|ｃｏ．|Ｒｉｇｈｔｓ|ｒｉｇｈｔｓ|' \
          r'Reserved|Reserved\.|reserved|RIGHT|right|Right|RIGHTS|rights|All|all|Ltd\.|ltd\.|CO\.|co\.)[\s\W]*?'

    # すべての該当単語を削除
    text = re.sub(pattern, '', text)
    
    # 以上以降を削除
    text = re.sub(r'以上.*$', '', text)
    
    # copyright、sectionを。に
    replace_phrases = ['©', 'copyright', 'Copyright', 'section', 'Section']
    text = re.sub('|'.join(map(re.escape, replace_phrases)), '。', text)

    # 残った部分の前後の不要な空白を削除
    text = text.strip()
    text = re.sub(r'\(\)|（）|[\(\)]{1}', '', text)
    
    # 7. 余分なスペースや空行を削除
    text = re.sub(r'\s+', '', text).strip()
    text = re.sub(r'[・「」〈〉]', '', text)  # 特殊記号を削除（必要に応じて追加）
    
    # 先頭が)とか）とかの場合
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
        if token.surface in ['(', '（']:
            inside_parentheses += 1
        elif token.surface in [')', '）']:
            inside_parentheses -= 1
        
        # 括弧内でない場合のみ分割トリガー
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
    print(f'{document[:100]}の開始')
    
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

    # 重要なキーワードを含む文を優先的に選ぶ
    important_keywords = ['通期', '上場廃止', '大幅', '世界初', '史上初', '業界初']

    # 重要な文を抽出
    important_sentences = [s for s in sentences if any(keyword in s for keyword in important_keywords)]

    # 重要な文を2つまでに制限
    important_sentences = important_sentences[:2]

    # 重要な文とランキング文の重複を避ける
    important_sentences_set = set(important_sentences)
    non_important_ranked_sentences = [s[1] for s in ranked_sentences if s[1] not in important_sentences_set]

    # 重要な文を追加した後に、num_sentencesに収まるように調整
    remaining_sentences_count = num_sentences - len(important_sentences)

    # 残りの文を選ぶ
    remaining_sentences = non_important_ranked_sentences[:remaining_sentences_count]

    # 重要な文とランキング文を結合
    summary_sentences = important_sentences + remaining_sentences

    # 重要な文とTextRankの文を元の順番に並べるために、インデックスマップを作成
    sentence_indices = {s: i for i, s in enumerate(sentences)}

    # summary_sentencesの順番を元のsentencesのインデックスに基づいて並べ替え
    summary = ''.join(sorted(summary_sentences, key=lambda s: sentence_indices[s]))

    # 使われていない文を取得してadjust_summary_lengthに渡す
    unused_sentences = [s[1] for s in ranked_sentences if s[1] not in summary_sentences]

    #print(f'これ返すよー？：{document[:100]}')

    return adjust_summary_length(summary, unused_sentences)