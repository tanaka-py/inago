import os
import pandas as pd
import asyncio
import aiohttp
import re
from . import summarize

# 重要度の高い文言検索
important_words_path = os.path.join(os.path.dirname(__file__), '../data/important_keywords.csv')
important_words_df = pd.read_csv(important_words_path, header=None)
important_words = important_words_df.iloc[:, 0].to_list()

slack_url = os.getenv('SLACK_WEBHOOK_URL', '')

# 通知を送信する(slackで)
async def send(
    df
):
    # ここはmodelで予想したものであがりそうなものを送信する予定
    # 今は重要ワードが含まれてたら
    filterd_df = df[df['Link'].str.contains('|'.join(important_words), na=False)]
    
    if not filterd_df.empty:
        
        tasks_list = [summarize_send(row) for row in filterd_df.itertuples(index=False)]
        
        if tasks_list:
            print('送信開始')
            # 並列で行う
            results = await asyncio.gather(*tasks_list, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"エラー発生：{result}")
                    
            print('全送信')
            
# 要約して通知
async def summarize_send(
    row
):
    #print(f'{row}を要約して送信')
    # 要約
    text = summarize.summarize_long_document(row.Link, summarize_limit=1000)
    
    for ward in important_words:
        text = re.sub(f'({ward})', r" `\1` ", text, flags=re.IGNORECASE)

    # メッセージ作成（マークダウン形式）
    payload = {
        "text": f'*{getattr(row, "Date", "N/A")} {getattr(row, "Time", "N/A")}*  \n*code:* `{getattr(row, "Code", "N/A")}`  \n---  \n*社名:* {getattr(row, "Name", "N/A")}  \n{text}',
        "mrkdwn": True  # これによって装飾を反映させる
    }

    # 送信
    async with aiohttp.ClientSession() as session:
        async with session.post(slack_url, json=payload) as response:
            if response.status == 200:
                print(f'code:{getattr(row, "Code", "N/A")} 社名:{getattr(row, "Name", "N/A")} 送信OK')
            else:
                print(f'code:{getattr(row, "Code", "N/A")} 社名:{getattr(row, "Name", "N/A")} 送信失敗 {response.text}')