import time
import re
import requests
import asyncio
import aiohttp
import multiprocessing
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import os
import sys

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

# テーブル整形関数（自動判定式！）
def format_table(table):
    try:
        if not table or not table[0]:
            return "{'table': 『*』}"
        
        formatted_content = []
        for row in table:
            cleaned_row = [clean_cell(cell) for cell in row]
            formatted_row = " | ".join(cleaned_row)
            formatted_content.append(f"「{formatted_row}」")

        return "{'table': " + " ".join(formatted_content) + "}"
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

# PDFをpdflumberを使って読込み(tableの部分をある程度ちゃんと把握させときたいため)
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

# PDFをダウンロードして要約して返す
def summarize_pdf(url):
    start_time = time.time()
    
    print(f'今からsummarize_pdf{url}とるよー')

    pdf_data = download_pdf(url)
    if not pdf_data:
        return "Failed to download PDF"
    
    print(f'終わったsummarize_pdf{url}')
    
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
    print(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(summarize_pdf, url) : url for url in urls}
        for future in future_to_url:
            url = future_to_url[future]
            if url == "https://f.irbank.net/pdf/20241031/140120241001592334.pdf":
                test = 'saaaa'
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
