import time
import re
import requests
import asyncio
import aiohttp
import multiprocessing
import fitz  # PyMuPDF
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import os
import sys
import pandas as pd

table_header_path = os.path.join(os.path.dirname(__file__), '../data/pdf_table_header.csv')
table_header_df = pd.read_csv(table_header_path, header=None)
table_headers = table_header_df.iloc[:,0].to_list()

# fitz や pdfplumber などが出す stderr を握りつぶす
sys.stderr = open(os.devnull, 'w')

# URLのPDFを取得
def download_pdf(url):
    start_time = time.time()
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        pdf_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            pdf_data.write(chunk)
        
        pdf_data.seek(0)  # メモリ位置をリセット
        elapsed_time = time.time() - start_time
        print(f"[download_pdf] {url} - {elapsed_time:.2f}秒")
        return pdf_data
    
    print(f"[download_pdf] {url} - Failed")
    return None


# (非同期版)
async def download_pdf_async(url):
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                pdf_data = BytesIO()
                async for chunk in response.content.iter_any():
                    pdf_data.write(chunk)
                
                pdf_data.seek(0)  # メモリ位置をリセット
                elapsed_time = time.time() - start_time
                print(f"[download_pdf_async] {url} - {elapsed_time:.2f}秒")
                return pdf_data
    
    print(f"[download_pdf_async] {url} - Failed")
    return None

# PDFのバイトデータから中のテキストデータを読込
# def extract_text_from_pdf(pdf_data):
#     start_time = time.time()

#     text = ""
#     try:
#         # pdf_dataがbytesの場合でも対応できるように修正
#         if isinstance(pdf_data, bytes):
#             pdf_data = BytesIO(pdf_data)
        
#         with fitz.open(stream=pdf_data.getvalue(), filetype='pdf') as doc:
#             for page in doc:
#                 text += page.get_text("text", sort=True) + "\n"
#     except RuntimeError:
#         text = 'PDFの解析に失敗しました'

#     elapsed_time = time.time() - start_time
#     print(f"[extract_text_from_pdf] - {elapsed_time:.2f}秒")
#     return text

# --- ラベル補正関数 ---
# 「（数字）」が見つかったときだけ処理、それ以外はそのまま
# 正規化処理（ラベルが付いている場合）
def normalize_label(cell):
    try:
        if not cell:
            return '*'
        cell = str(cell).strip()
        match = re.search(r'（\d+）', cell)
        if match:
            num = match.group(0)
            rest = cell.replace(num, '').strip()
            return f"{num} {rest}"
        return cell
    except Exception as e:
        print(f"normalize_labelでエラーが発生しました: {e}")
        return '*'

# セル掃除係
def clean_cell(cell):
    try:
        if not cell:
            return '*'
        text = str(cell).replace("\n", " ").strip()
        return normalize_label(text)
    except Exception as e:
        print(f"clean_cellでエラーが発生しました: {e}")
        return '*'

def guess_header_rows(table):
    likely_keywords = table_headers
    header_flags = []

    try:
        # 最初の3行をチェック！
        for i, row in enumerate(table[:3]):
            keyword_hits = sum(1 for cell in row if any(k in str(cell) for k in likely_keywords))
            non_numeric = sum(1 for cell in row if not str(cell).strip().replace(',', '').replace('.', '').isdigit())

            # 条件判定コポォｗｗｗ
            is_header = (
                keyword_hits >= 1 and      # ←ここがポイント！
                non_numeric >= len(row) * 0.6
            )
            header_flags.append(is_header)

        # 最後にTrueが出た位置までをヘッダーとみなす！
        if True in header_flags:
            last_index = max(i for i, flag in enumerate(header_flags) if flag)
            return last_index + 1  # 行数なので+1

    except Exception as e:
        print(f"guess_header_rowsでエラーが発生しました: {e}")

    return 0



# ヘッダー結合魔法
def merge_headers(table, header_rows):
    try:
        if header_rows == 0:
            return [f"列{i+1}" for i in range(len(table[0]))]

        header_parts = table[:header_rows]
        max_len = max(len(row) for row in header_parts)

        # 各行の長さを max_len にそろえる（足りないところは空文字追加）
        for i in range(header_rows):
            header_parts[i] += [''] * (max_len - len(header_parts[i]))

        # 前回の値を記憶しておく用（空白だったときに引き継ぐ）
        parts = [''] * max_len

        for row_idx in range(header_rows):
            for col_idx in range(max_len):
                val = (header_parts[row_idx][col_idx] or '').strip()
                if not val and row_idx == 0:
                    # 最初の行であれば上のを引き継げないから前の列のを
                    val = (parts[col_idx -1] if 0 < col_idx else ''  or '').strip()
                        
                parts[col_idx] += (' ' if 0 < row_idx else '') + val # 連結
                
        return parts

    except Exception as e:
        print(f"merge_headersでエラーが発生しました: {e}")
        return []


# テーブル整形関数（自動判定式！）
def format_table(table):
    try:
        if not table or not table[0]:
            return "{'table': 『*』}"

        header_rows = guess_header_rows(table)
        headers = merge_headers(table, header_rows)
        content = table[header_rows:]

        formatted_headers = "『" + " | ".join([clean_cell(header) for header in headers]) + "』"
        formatted_content = []
        for row in content:
            cleaned_row = [clean_cell(cell) for cell in row]
            formatted_row = " | ".join(cleaned_row)
            formatted_content.append(f"「{formatted_row}」")

        return "{'table': " + " " + formatted_headers + " " + " ".join(formatted_content) + "}"
    except Exception as e:
        print(f"format_tableでエラーが発生しました: {e}")
        return "{'table': 'エラーが発生しました'}"

# --- PDF抽出メイン関数 ---
def remove_duplicate_table_text(text, table_rows):
    """テーブルの最初と最後の行がテキスト内に含まれてたら除去するマン"""
    if not text or not table_rows:
        return text

    first_line = ' '.join([cell for cell in table_rows[0] if cell])
    last_line = ' '.join([cell for cell in table_rows[-1] if cell])

    # 両端の10文字を拾ってワイルドに検索ｗ
    pattern = re.escape(first_line[:10]) + r'.*?' + re.escape(last_line[-10:])
    new_text = re.sub(pattern, '', text, flags=re.DOTALL)

    return new_text.strip()

def safe_within_bbox_text(page, top, bottom):
    """
    高さが0の場合に落ちないようにするためのラッパー関数なんだが？ｗｗｗ
    """
    if abs(bottom - top) < 1e-3:  # 超小さい差も無視（精度誤差対策）
        return ""
    try:
        text = page.within_bbox((0, top, page.width, bottom)).extract_text()
        return text or ""
    except Exception as e:
        print(f"フォカヌポウｗｗｗ within_bboxエラー回避したおｗｗｗ: {e}")
        return ""


def extract_text_from_pdf(pdf_data):
    start_time = time.time()
    
    try:
        if isinstance(pdf_data, bytes):
            pdf_data = BytesIO(pdf_data)
        
        final_content = ""

        with pdfplumber.open(pdf_data) as pdf:
            table = None
            for page_num, page in enumerate(pdf.pages):
                table_objects = page.find_tables()
                table_infos = sorted(table_objects, key=lambda x: x.bbox[1])

                content = ""
                last_y = 0

                for table_index, table_obj in enumerate(table_infos):
                    table = table_obj.extract()
                    top_y, bottom_y = table_obj.bbox[1], table_obj.bbox[3]

                    safe_top = min(last_y, top_y)
                    safe_bottom = max(last_y, top_y)

                    top_text = ""
                    top_text = safe_within_bbox_text(page, safe_top, safe_bottom)

                    # 重複排除処理キタコレｗｗｗｗ
                    if top_text:
                        top_text = remove_duplicate_table_text(top_text, table)
                        content += top_text.strip() + "\n\n"

                    # テーブル挿入するおｗｗｗ
                    content += format_table(table) + "\n\n"
                    last_y = bottom_y

                safe_top = min(last_y, page.height)
                safe_bottom = max(last_y, page.height)
                bottom_text = safe_within_bbox_text(page, safe_top, safe_bottom)

                if bottom_text:
                    if table is not None:
                        bottom_text = remove_duplicate_table_text(bottom_text, table)
                    content += bottom_text.strip()

                final_content += content.strip() + "\n"

    except RuntimeError as e:
        final_content = f'★Error PDFの解析に失敗しました: {e}'
    except Exception as e:
        final_content = f'★Error 想定外のエラーが発生しました: {e}'

    elapsed_time = time.time() - start_time
    print(f"[extract_text_from_pdf] - {elapsed_time:.2f}秒")
    return final_content.strip()


# 要約は諦める
# ええものは金かかる
# 長文読み込めるのが少ない
# 要約しょぼい
# 要約で削れるのは果たしてどうか
# 要約処理(sumyを使用)
# def summarize_text(text):
#     start_time = time.time()

#     try:
#         # テキストの前処理: 改行や余分な空白を削除
#         cleaned_text = re.sub(r'\n+', ' ', text)  # 改行をスペースに置き換え
#         cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 複数の空白を1つにする
#         cleaned_text = cleaned_text.strip()  # 両端の余分な空白を削除

#         # Hugging Faceの生成型要約モデルをロード
#         summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#         # 要約処理
#         summary = summarizer(cleaned_text, max_length=200, min_length=50, do_sample=False)

#         # summaryが空でないかを確認
#         if not summary or 'summary_text' not in summary[0]:
#             raise ValueError("Summary is empty or missing 'summary_text' key")

#         elapsed_time = time.time() - start_time
#         print(f"[summarize_text] - {elapsed_time:.2f}秒")

#         return summary[0]['summary_text']

#     except Exception as e:
#         # エラーが発生した場合、エラーの内容を表示
#         print(f"[ERROR] An error occurred: {e}")
#         return None  # エラーが発生した場合はNoneを返す


# PDFをダウンロードして要約して返す
def summarize_pdf(url):
    start_time = time.time()

    pdf_data = download_pdf(url)
    if not pdf_data:
        return "Failed to download PDF"
    
    text = extract_text_from_pdf(pdf_data)
    if not text:
        return "No readable text found"

    # 要約諦め
    #summary = summarize_text(text)
    summary = text

    elapsed_time = time.time() - start_time
    print(f"[summarize_pdf] {url} - Total: {elapsed_time:.2f}秒")
    
    return summary

# (非同期版)
async def summarize_pdf_async(url):
    start_time = time.time()

    pdf_data = await download_pdf_async(url)
    if not pdf_data:
        return "Failed to download PDF"
    
    text = extract_text_from_pdf(pdf_data)
    if not text:
        return "No readable text found"

    # 要約諦め
    #summary = summarize_text(text)
    summary = text

    elapsed_time = time.time() - start_time
    print(f"[summarize_pdf_async] {url} - Total: {elapsed_time:.2f}秒")

    return summary

# 要約ゲットまでを並列処理で(ThreadPoolExecutor)
def in_parallel(urls, max_workers=10):
    start_time = time.time()

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(summarize_pdf, url) : url for url in urls}
        for future in future_to_url:
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = f"Error: {e}"

    elapsed_time = time.time() - start_time
    print(f"[in_parallel] Total processing time: {elapsed_time:.2f}秒")
    
    return results

# 要約ゲットまでを並列処理で(asyncio)
async def in_parallel_asyncio(urls, max_workers=10):
    start_time = time.time()

    # asyncio.gatherで並列処理
    tasks = [summarize_pdf_async(url) for url in urls]
    summaries = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time
    print(f"[in_parallel_asyncio] Total processing time: {elapsed_time:.2f}秒")

    return summaries

# 要約ゲットまでを並列処理で(multiprocessing)
def in_parallel_multiprocessing(urls, max_workers=10):
    start_time = time.time()

    with multiprocessing.Pool(processes=max_workers) as pool:
        summaries = pool.map(summarize_pdf, urls)

    elapsed_time = time.time() - start_time
    print(f"[in_parallel_multiprocessing] Total processing time: {elapsed_time:.2f}秒")

    return summaries
