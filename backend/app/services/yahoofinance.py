
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

yahoofinance_listurl = os.getenv('YAHOOFINANCE_LISTURL')

#　yfinanceが取得出来ないパターンが多いのでyahoofinance日本語版のページをスクレイピングする
def get_yahoofinance_stocllist(
    code,
    target_date,
    is_past = False
):
    start_str_date = ''
    end_str_date = ''
    if is_past:
        # 過去200日
        end_date = datetime.strptime(target_date, '%Y-%m-%d') if isinstance(target_date, str) else target_date
        end_str_date = end_date.strftime('%Y%m%d')
        start_date = end_date - timedelta(days=200)
        start_str_date = start_date.strftime('%Y%m%d')
    else:
        # 未来３か月
        start_date = datetime.strptime(target_date, '%Y-%m-%d') if isinstance(target_date, str) else target_date
        start_str_date = start_date.strftime('%Y%m%d')
        end_date = start_date + timedelta(days=90)
        end_str_date = end_date.strftime('%Y%m%d')
    
    data = []
    stock_json = {}
    for page_num in range(1, 20):
        # URL作成
        list_url = f'{yahoofinance_listurl.replace('{0}', code + '.T')}&from={start_str_date}&to={end_str_date}&page={page_num}'
        
        sub = get_list_data(list_url)
        if sub:
            # 見つかった場合
            data.extend(sub)
        else :
            # Tで見つからなかったらもうあきらめる　東証以外はどうでもいいし負荷かかってもあれやし
            break
            
            # # 見つからなかった場合 Sで試す　これでもダメならもう無理
            # list_url = f'{yahoofinance_listurl.replace('{0}', code + '.S')}&from={start_str_date}&to={end_str_date}&page={page_num}'
            # sub = get_list_data(list_url)
            # if sub:
            #     data.extend(sub)
            # else:
            #     # 見つからなかった場合 Nで試す　これでもダメならもう無理
            #     list_url = f'{yahoofinance_listurl.replace('{0}', code + '.N')}&from={start_str_date}&to={end_str_date}&page={page_num}'
            #     sub = get_list_data(list_url)
            #     if sub:
            #         data.extend(sub)
            #     else:
            #         # 見つからなかった場合 Fで試す　これでもダメならもう無理
            #         list_url = f'{yahoofinance_listurl.replace('{0}', code + '.F')}&from={start_str_date}&to={end_str_date}&page={page_num}'
            #         sub = get_list_data(list_url)
            #         if sub:
            #             data.extend(sub)
            #         else:
            #             print(f'打つ手なし:{code}')
            #             break
            
    # dfに
    if data:
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df[['Dividends', 'Stock Splits']] = 0.0, 0.0
        
        # 日付の昇順で並べ直し
        df = df.sort_values(by='Date', ascending=True)
        
        stock_json = df.to_json(orient='records', force_ascii=False, lines=False, date_format='iso')
        
    return stock_json
        
def get_list_data(url):
    rtn = []
    
    finance_response = requests.get(url)
    if finance_response.status_code == 200:
        finance_soup = BeautifulSoup(finance_response.text, 'html.parser')
        mainlist = finance_soup.find('table', class_='StocksEtfReitPriceHistory__historyTable__13C_')
        
        if mainlist:
            for row in mainlist.find_all('tr', class_='HistoryTable__row__2ZqX'):
                columns = row.find_all('td')
                
                # 分割行がある
                row_divide = row.find('td', class_='HistoryTable__detail__UV2e HistoryTable__detail--joined__20yO')
                
                if columns and not row_divide:
                    date = row.find('th', class_='HistoryTable__date__1whp').text.strip()
                    date = datetime.strptime(date, '%Y年%m月%d日').strftime('%Y-%m-%d')
                    open = pd.to_numeric(columns[0].find('span', class_='StyledNumber__value__3rXW').text.strip().replace(',', ''), downcast='float', errors='coerce')
                    high = pd.to_numeric(columns[1].find('span', class_='StyledNumber__value__3rXW').text.strip().replace(',', ''), downcast='float', errors='coerce')
                    low = pd.to_numeric(columns[2].find('span', class_='StyledNumber__value__3rXW').text.strip().replace(',', ''), downcast='float', errors='coerce')
                    close = pd.to_numeric(columns[3].find('span', class_='StyledNumber__value__3rXW').text.strip().replace(',', ''), downcast='float', errors='coerce')
                    volume = pd.to_numeric(columns[4].find('span', class_='StyledNumber__value__3rXW').text.strip().replace(',', ''), downcast='float', errors='coerce')

                    # 取得したデータをリストに追加
                    rtn.append([date, open, high, low, close, volume])
        
    return rtn    
        
    