import time
import re
import requests
import asyncio
import aiohttp
import multiprocessing
import fitz  # PyMuPDF
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

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
def extract_text_from_pdf(pdf_data):
    start_time = time.time()

    text = ""
    try:
        # pdf_dataがbytesの場合でも対応できるように修正
        if isinstance(pdf_data, bytes):
            pdf_data = BytesIO(pdf_data)
        
        with fitz.open(stream=pdf_data.getvalue(), filetype='pdf') as doc:
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
    except RuntimeError:
        text = 'PDFの解析に失敗しました'

    elapsed_time = time.time() - start_time
    print(f"[extract_text_from_pdf] - {elapsed_time:.2f}秒")
    return text

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
