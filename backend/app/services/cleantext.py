# 文章中のヘッダーとフッターのパターン設定
from typing import List, Tuple
import re
import unicodedata
import os
import pandas as pd

# 。に変換する区切りワードを読み込み
replace_list_path = os.path.join(os.path.dirname(__file__), '../data/summarize_replace.csv')
replace_list_df = pd.read_csv(replace_list_path, header=None)
replace_list = replace_list_df.iloc[:,0].to_list()

# PATTERNS_HEADER: List[Tuple[str, str]] = [
#         (r'https?://[a-zA-Z0-9.-]+\.(?:com|jp|net|org|co\.jp|biz|info|gov|edu|io|ai)(?:/[A-Za-z/]*)?(?=[^A-Za-z/]|$)', 'url'),  # URLを適切に切る
#         (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}', 'email'),  # メールアドレス
#         (r'(TEL|ＴＥＬ|T E L|電話番号|電話|電 話)', 'phone_header'),  # 電話ヘッダ
#         (r'(\（?[\d０-９]{2,5}\）?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}|\bTEL\b[\s\-−－]?\(?[\d０-９]{2,5}\)?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4})', 'phone_number'),  # 電話番号
#         (r'(プライム市場、福証)', '福証')  # 福証ヘッダ
#     ]

_PATTERNS_HEADER: List[Tuple[str, str]] = [
        (r'(ついて)', 'title1'),  # ついて
        (r'(お知らせ)', 'title2'),  # お知らせ
        (r'(OVERVIEW)', 'title3'), # OVERVIEW
        (r'(業績サマリー Point)', 'title4'), # 業績サマリー Point
        (r'(百万円未満切捨て)', 'title5'), # 百万円未満切捨て
    ]

_PATTERNS_FOTTER: List[Tuple[str, str]] = [
    (r'■本件に関するお問い合わせ先', 'fotter1'),
    (r'【本件に関する問合せ】', 'fotter2'),
    (r'本件に関するお問合せ', 'fotter3'),
    (r'お問い合わせ', 'fotter4'),
    (r'本資料に関するお問合せ先', 'fotter5')
]

_PHONE_PATTERN = r'''
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
        
_DATE_PATTERN = r'''
    (?:
        [0-9０-９]{4}[ \u3000]?年[ \u3000]?[0-9０-９]{1,2}[ \u3000]?月[ \u3000]?[0-9０-９]{1,2}[ \u3000]?日 |  # yyyy年mm月dd日 の形式
        [0-9０-９]{4}[/-][0-9０-９]{1,2}[/-][0-9０-９]{1,2} |  # yyyy/mm/dd や yyyy-mm-dd の形式
        [0-9０-９]{1,2}[ \u3000]?月[ \u3000]?[0-9０-９]{1,2}[ \u3000]?日 |  # mm月dd日 の形式
        [0-9０-９]{4}[ \u3000]?年[ \u3000]?[0-9０-９]{1,2}[ \u3000]?月 |   # yyyy年mm月 の形式（2022年3月期のような）
        [0-9０-９]{1,2}[ \u3000]?年[ \u3000]?[0-9０-９]{1,2}[ \u3000]?月   # yy年mm月 の形式（2022年3月期のような）
    )
'''

# ハイフン・長音・水平バー系＋空白を許容して2回以上連続
_HYPHEN_PATTERN = r'(?:[ \u3000]*[―ー‐－\-–—﹘=#＃◎✓〇﹣][ \u3000]*){2,}'

_URL_PATTERN = r"""
    (?:
        https?:// |
        www\.
    )
    [\w\-._~:/?#\[\]@!$&'()*+,;=%]+
"""

_DAY_OF_WEEK_PATTERN = r'[（(][月火水木金土日][）)]'

# 単位リスト（あとで追加し放題ｗｗｗ）
_UNITS = ["百万円", "億円", "百万", "万円", "円", "人", "株", "%", "件", "社", "号", "倍"]


# ヘッダー部分を削除
def _clean_header(text):
    
    is_delete = False
    pattern = '|'.join(pattern[0] for pattern in _PATTERNS_HEADER)
    match = re.search(pattern, text)
    if match and match.start() <= 300:
        # マッチした部分までを削除
        #text = re.sub(r'^.*?' + pattern, '', text, count=1)
        text = re.sub(r'^.*?(' + pattern + ')', '', text, count=1)
        is_delete = True
    
    if not is_delete:
        # matchオブジェクトが欲しいので finditer を使うお
        for match in re.finditer(_PHONE_PATTERN, text, re.VERBOSE | re.IGNORECASE):
            digits = re.sub(r'\D', '', match.group())
            if len(digits) >= 10 and match.start() < 200:
                # 有効な電話番号っぽい！そこまでズバーンと削除ｗｗｗ
                text =  text[match.end():]
                break

    return text

# footer部分を削除
def _clean_footer(text):
    
    # すべての "以上" の位置を取得
    matches = list(re.finditer(r'以[\s　]*上', text))
    
    if matches:
        last_match = matches[-1]  # 最後の "以上" を取得
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    # 以上を削除したあとまだあるフッター要素を削除
    patterns = [pattern for pattern, _ in _PATTERNS_FOTTER]
    matches = list(re.finditer('|'.join(map(re.escape, patterns)), text))

    if matches:
        last_match = matches[-1]
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    return text


# 年としてありえる連番がガチガチに詰まってるのを探す
def _insert_spaces_between_years(text, min_years=3):
    # 年パターン
    year_pattern = r'(19[8-9]\d|20[0-2]\d)'
    # 年の連続（最低3つ以上）に直前の文字列をセットで取得
    pattern = re.compile(rf'(.+?)((?:{year_pattern}){{{min_years},}})')

    def replacer(match):
        before = match.group(1)
        chunk = match.group(2)
        years = [chunk[i:i+4] for i in range(0, len(chunk), 4)]
        return before + ' ' + ' '.join(years)

    return pattern.sub(replacer, text)


# 数値と単位を_でつなげる
def _combine_number_and_unit(text):
    unit_pattern = '|'.join(sorted(_UNITS, key=len, reverse=True))
    
    # 単位と数値の間に_を入れるためのパターン
    pattern_all = rf'(\d[\d,\.]*)(\s*)({unit_pattern})(?=\b|[^ぁ-んァ-ン一-龥a-zA-Z])'
    
    # スペースありのパターンを最初に適用
    text = re.sub(pattern_all, r'\1_\3', text)  # スペースがあれば _ を挿入

    return text

# 開示文章内から不要な文章を削除
def clean_text(text):
    # 「異体字セレクタ」や「制御文字」に該当するやつをゴッソリ除去
    text = re.sub(r'[\u2000-\u200F\uFE0F\u2028\u2029\u2060]+', '', text)
    
    # 全角英数を半角に
    text = unicodedata.normalize('NFKC', text)
    
    # ヘッダー部分を削除
    text = _clean_header(text)
    
    # フッター部分を削除
    text = _clean_footer(text)
    
    # URLを<URL>タグに置き換える
    text = re.sub(_URL_PATTERN, '<URL>', text, flags=re.VERBOSE | re.IGNORECASE)
    
    # 電話番号を<PHONE>タグに置き換える
    text = re.sub(_PHONE_PATTERN, '<PHONE>', text, flags=re.VERBOSE | re.IGNORECASE)
    
    # 日付を<DATE>タグに置き換える
    text = re.sub(_DATE_PATTERN, '<DATE>', text, flags=re.VERBOSE | re.IGNORECASE)
    
    # 曜日を<DOW>タグに置き換える
    text = re.sub(_DAY_OF_WEEK_PATTERN, '<DOW>', text, flags=re.VERBOSE | re.IGNORECASE)
    
    # 数値と単位を結合
    text = _combine_number_and_unit(text)
    
    # 連続年度にスペースを
    text = _insert_spaces_between_years(text)
    
    # 連続するハイフン（ーまたは-）を1つにする
    text = re.sub(_HYPHEN_PATTERN, '―', text, flags=re.VERBOSE | re.IGNORECASE)
    
    # summarize_replace.csvに登録されてるものを。に置き換える(区切り文字として扱う)
    text = re.sub('|'.join(map(re.escape, replace_list)), '。', text, flags=re.VERBOSE | re.IGNORECASE)

    # 最終クリーン
    text = re.sub(r'。+', '。', text)
    text = re.sub(r'[.。,、]{2,}', '', text)  # 連続した記号をまとめて削除
    #text = re.sub(r'\(\)|（）|[\(\)]{1}', '', text)
    text = re.sub(r'^（代表）', '', text)
    text = re.sub(r'^）のお知らせ', '', text)
    text = re.sub(r'^のお知らせ', '', text)
    text = re.sub(r'^[）\)]', '', text)
    text = text.strip()

    return text

#### ↓　開示内にて表とだったであろうグループを特定 #############################################
def is_exclude_calendar(line):
    return re.search(r'\b(?:1[0-2]|[1-9])(?:\s+(?:1[0-2]|[1-9])){5,}\b', line)

