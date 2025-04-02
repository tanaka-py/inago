import os
import pandas as pd
import asyncio
import aiohttp
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
        
        tasks_list = [summarize_send(row.Code, row.Name, row.Link) for row in filterd_df.itertuples()]
        
        if tasks_list:
            # 並列で行う
            results = await asyncio.gather(*tasks_list, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"エラー発生：{result}")
            
# 要約して通知
async def summarize_send(
    code,
    name,
    link
):
    # 要約
    text = summarize.summarize_long_document(link, summarize_limit=1000)
    
    #　メッセージ作成
    message={
        'text': f'code:{code}\n社名:{name}\n{text}'
    }
    
    # 送信
    async with aiohttp.ClientSession() as session:
        async with session.post(slack_url, json=message) as response:
            if response.status == 200:
                print(f'code:{code} 社名:{name} 送信OK')
            else:
                print(f'code:{code} 社名:{name} 送信失敗')