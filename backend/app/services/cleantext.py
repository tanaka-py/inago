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
        (r'(お知らせの一部訂正について)', 'title0'),  # お知らせ
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
    (?<![.,+\-△▲\d])  # ← 記号と数字の直前を除外！
    (
        [（(]?\s*0\d{1,4}\s*[）)]?        # ← 全角/半角カッコ付き市外局番対応
        [-ー－−–（）\s]*                   # ← 区切り文字を許容（en dashも！）
        \d{2,4}
        [-ー－−–（）\s]*
        \d{3,4}
    )
    (?!\d)
'''

_DATE_PATTERN = r'''
    (?<![.,+\-△▲])
    (?:
        (?:19[0-9]{2}|20[0-3]\d|2040)[ \u3000]?年[ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月[ \u3000]?
            (?:0?[1-9]|[12][0-9]|3[01])[ \u3000]?日 |
        (?:19[0-9]{2}|20[0-3]\d|2040)[/-]
            (?:1[0-2]|0?[1-9])[/-]
            (?:0?[1-9]|[12][0-9]|3[01]) |
        (?:1[0-2]|0?[1-9])[ \u3000]?月[ \u3000]?
            (?:0?[1-9]|[12][0-9]|3[01])[ \u3000]?日 |
        (?:19[0-9]{2}|20[0-3]\d|2040)[ \u3000]?年[ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月 |
        (?:19[0-9]{2}|20[0-3]\d|2040)[/-／－]
            (?:1[0-2]|0?[1-9]) |
        (?:[0-9０-９]{2})[ \u3000]?年[ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月 |
        (?:[0-9]{2})[ \u3000]*年[ \u3000]*(?:1[0-2]|0?[1-9])(?=[ \u3000月.,)）\n]|$)
    )
'''

_FISCAL_PATTERN = r'''
    (?<![.,+\-△▲])
    (?:
        (?:19[0-9]{2}|20[0-3]\d|2040)/(?:1[0-2]|0?[1-9])期(?:\s*(?:[1-4]Q))? |
        (?:[0-9０-９]{2})/(?:1[0-2]|0?[1-9])期(?:\s*(?:[1-4]Q))? |
        (?:19[0-9]{2}|20[0-3]\d|2040)[ \u3000]?年[ \u3000]?(?:1[0-2]|0?[1-9])[ \u3000]?月期 |
        (?:[0-9０-９]{2})[ \u3000]?年[ \u3000]?(?:1[0-2]|0?[1-9])[ \u3000]?月期
    )
'''

_PERIOD_PATTERN = r'''
    (?<![.,+\-△▲])
    (?:
        (?:19[0-9]{2}|20[0-3]\d|2040)[ \u3000]?年[ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?[-～〜][ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月(?!\s*\d{1,2}\s*日) |

        (?:19[0-9]{2}|20[0-3]\d|2040)[/／－][ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?[-～〜][ \u3000]?
            (?:1[0-2]|0?[1-9])(?!\s*\d{1,2}\s*日) |

        (?:1[0-2]|0?[1-9])[ \u3000]?[-～〜][ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月(?!\s*\d{1,2}\s*日) |

        (?:19[0-9]{2}|20[0-3]\d|2040)[ \u3000]?年[ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月[ \u3000]?[-～〜][ \u3000]?
            (?:1[0-2]|0?[1-9])[ \u3000]?月(?!\s*\d{1,2}\s*日)
    )
'''

_MONTH_PATTERN = r'''
    (?<![.,+\-△▲0-9０-９])
    (?:1[0-2]|0?[1-9])
    [ \u3000]*
    月
    (?![日0-9０-９])
'''

_YEAR_PATTERN = r'''
    (?<![.,+\-△▲0-9０-９])
    (?:19[0-9]{2}|20[0-3]\d|2040)
    年(?:度)?
    (?![月期0-9０-９])
'''

_DAY_PATTERN = r'''
    (?<![.,+\-△▲0-9０-９])
    ((?:0?[1-9]|[12][0-9]|3[01])      # ← 日付の数字
    [ \u3000]*日)                     # ← 日までキャプチャ
    (?=[ \u3000（(])                  # ← 日の直後がスペースやカッコのときだけヒット！
'''

_ESTIMATE_PATTERN = r'''
    (?<![.,+\-△▲])
    (?:19[0-9]{2}|20[0-3]\d|2040)
    (?:年(?:度)?)?
    [\(\（]\s*
    (?:見込み|見通し|予測)
    \s*[\)\）]
'''

_TERM_PATTERN = r'''
    (?:
        (?<![.,+\-△▲])
        (?:19[0-9]{2}|20[0-2]\d)
        [\u3000\s\-ー〜～～]?
        (?:19[0-9]{2}|20[0-2]\d)
        年(?:度)?
    )
'''

_URL_PATTERN = r"""
    (?:
        https?:// |
        www\.
    )
    [\w\-._~:/?#\[\]@!$&'()*+,;=%]+
"""

_DAY_OF_WEEK_PATTERN = r'[（(](月曜日|火曜日|水曜日|木曜日|金曜日|土曜日|日曜日|月|火|水|木|金|土|日)[）)]'

_TIME_PATTERN = r'''
    (?:
        (午前|午後)?              # 午前 or 午後（省略可）
        \s*                       # スペース（0回以上）
        ([0-9]|1[0-9]|2[0-4])     # 時（0〜24）
        \s*時\s*                  # 「時」の前後にスペース許容
        ([0-9]|[1-5][0-9]|60)     # 分（0〜60）
        \s*分                     # 「分」の後にもスペース許容
    )
'''

# 単位リスト（あとで追加し放題ｗｗｗ）
_UNITS = ["百万円", "千万円", "億円", "百万", "万円", "千円", "万人",
          "円", "株", "%", "％", "件", "号", "倍", "社", "人" "ポイント", "pt"]
_UNITS_NONE = [
    # 株
    "株主", "株式", "株当たり", "株数", "株価", "株元", "株券",
    # 社
    "社長", "社名", "社外", "社員", "社数", "社宅", "社内", "社歴",
    # 人
    "人件費", "人材", "人口", "人数", "人員", "人月", "人時", "人手", "人物",
    # 件
    "件数", "件名", "事件", "案件",
    # 倍・ポイント・他
    "倍率", "倍増", "倍減",
    "ポイント還元", "ポイント制", "ポイント数",
    "％表示", "％比率", "％変動"
]

_UNIT_PATTERN = '|'.join(sorted(map(re.escape, _UNITS), key=len, reverse=True))

_SIGNED_NUMBER_PATTERN = r'''
    (?:                                # 非キャプチャグループで包むお
        (?P<plus>[＋+])                # 全角＋ or 半角+（命名キャプチャ）
        (?P<plusnum>\d+)              # 数字（1回以上）
      |
        (?P<minus>[△▲−ー－-])         # 多様なマイナス記号（全角・長音含む）
        (?P<minusnum>\d+)             # 数字（1回以上）
    )
'''

_Q_SPLIT_PATTERN = r'''
    (?<!\s)
    (?=[1-4]Q)
'''

_DAY_PATTERN = r'''
    (?<![.,+\-△▲0-9０-９])
    ((?:0?[1-9]|[12][0-9]|3[01])      # ← 日付の数字
    [ \u3000]*日)                     # ← 日までキャプチャ
    (?=[ \u3000（(])                  # ← 日の直後がスペースやカッコのときだけヒット！
'''

# 事前コンパイルパターンたち（お守り装備コポォ）
_RE_REMOVE_CHARS = re.compile(r'[\u2000-\u200F\uFE0F\u2028\u2029\u2060]+')
_RE_URL = re.compile(_URL_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_PHONE = re.compile(_PHONE_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_ESTIMATE = re.compile(_ESTIMATE_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_FISCAL = re.compile(_FISCAL_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_TERM = re.compile(_TERM_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_PERIOD = re.compile(_PERIOD_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_DATE = re.compile(_DATE_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_YEAR = re.compile(_YEAR_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_MONTH = re.compile(_MONTH_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_DAY = re.compile(_DAY_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_DOW = re.compile(_DAY_OF_WEEK_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_TIME = re.compile(_TIME_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_SIGNED_NUM = re.compile(_SIGNED_NUMBER_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_REPLACE_LIST = re.compile('|'.join(map(re.escape, replace_list)), flags=re.VERBOSE | re.IGNORECASE)
_RE_Q_SPLIT = re.compile(_Q_SPLIT_PATTERN, flags=re.VERBOSE | re.IGNORECASE)
_RE_MULTIPLE_PERIODS = re.compile(r'。+')
_RE_SYMBOLS = re.compile(r'[.。,、]{2,}')
_RE_HEAD_REP = re.compile(r'^（代表）')
_RE_HEAD_NOTICE = re.compile(r'^）のお知らせ')
_RE_HEAD_NOTICE2 = re.compile(r'^のお知らせ')
_RE_HEAD_CLOSING = re.compile(r'^[）\)]')
_COMBINE_UNIT_RE = re.compile(rf'(\d+(?:[,.]\d+)?)[\s\u3000]*({_UNIT_PATTERN})')

_HEADER_PATTERN = re.compile('|'.join(pattern[0] for pattern in _PATTERNS_HEADER))

_FOOTER_PATTERN_1 = re.compile(r'以[\s　]*上')
_FOTTER_PATTERN_2 = re.compile('|'.join(map(re.escape, [p[0] for p in _PATTERNS_FOTTER])))

_ZENKAKU = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
_HANKAKU = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ZEN2HAN_TABLE = str.maketrans(_ZENKAKU, _HANKAKU)

# ヘッダー部分を削除
def _clean_header(text):
    
    is_delete = False
    match = re.search(_HEADER_PATTERN, text)
    if match and match.start() <= 300:
        # マッチした部分までを削除
        text = re.sub(r'^.*?(' + _HEADER_PATTERN.pattern + ')', '', text, count=1)
        is_delete = True
    
    if not is_delete:
        # matchオブジェクトが欲しいので finditer を使うお
        for match in re.finditer(_RE_PHONE, text):
            digits = re.sub(r'\D', '', match.group())
            if len(digits) >= 10 and match.start() < 200:
                # 有効な電話番号っぽい！そこまでズバーンと削除ｗｗｗ
                text = text[match.end():]
                break

    return text

# footer部分を削除
def _clean_footer(text):
    
    # すべての "以上" の位置を取得
    matches = list(re.finditer(_FOOTER_PATTERN_1, text))
    
    if matches:
        last_match = matches[-1]  # 最後の "以上" を取得
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    # 以上を削除したあとまだあるフッター要素を削除
    matches = list(re.finditer(_FOTTER_PATTERN_2, text))

    if matches:
        last_match = matches[-1]
        if last_match.start() >= len(text) - 200:
            text = text[:last_match.start()]    #直前までをリセット

    return text


# 年としてありえる連番がガチガチに詰まってるのを探す
def _replace_year_sequence_with_token(text, min_years=3):
    # 年パターン（西暦1980〜2029）
    year_pattern = r'(19[8-9]\d|20[0-2]\d)'

    # そして残った連続年列（最低 min_years 個）を対象に <YEAR> に置換
    # 連続した年だけ抽出（4桁ごとに分割できるように）
    year_seq_pattern = re.compile(rf'(.+?)((?:{year_pattern}){{{min_years},}})')
    year_seq_pattern = re.compile(
        rf'(?P<before>.+?)'
        rf'(?P<years>(?:{year_pattern}){{{min_years},}})'
        )

    def replacer(match):
        before = match.group('before')
        chunk = match.group('years')
        years = [chunk[i:i+4] for i in range(0, len(chunk), 4)]
        return before + ' ' + ' '.join(['<YEAR>'] * len(years))

    return year_seq_pattern.sub(replacer, text)


# 数値と単位を_でつなげる
def _combine_number_and_unit(text):
    
    def unit_replacer(match):
        start, end = match.span()
        unit = match.group(2)  # ← マッチした単位だけ取り出すぞｗｗｗ
        after = text[end:end + 10]  # ← 後ろの文字列10文字くらい見とくと安心ｗｗｗ
        
        # 例えば「株主」っていう例外ワードが「株」に続くかどうか判定するんだおｗｗｗ
        for ex in _UNITS_NONE:
            suffix = ex[len(unit):]
            if suffix and after.startswith(suffix):
                return match.group(0)  # 除外
        return f"{match.group(1)}_{unit}"
    
    return _COMBINE_UNIT_RE.sub(unit_replacer, text)

# 全角英数を半角英数に
def _convert_zenkaku_alnum_to_hankaku(text):
    return text.translate(_ZEN2HAN_TABLE)

# +-と数値を_でつなげる
def _normalize_signed_number(text):
    def signed_replacer(match):
        if match.group("plus"):
            return f"+{match.group('plusnum')}"
        elif match.group("minus"):
            return f"-{match.group('minusnum')}"
        return match.group(0)
    return _RE_SIGNED_NUM.sub(signed_replacer, text)

def clean_text(text):
    # 「異体字セレクタ」や「制御文字」に該当するやつをゴッソリ除去
    text = _RE_REMOVE_CHARS.sub('', text)

    # 全角英数を半角に
    text = _convert_zenkaku_alnum_to_hankaku(text)

    # ヘッダー部分を削除
    text = _clean_header(text)

    # フッター部分を削除
    text = _clean_footer(text)

    # URLを<URL>タグに置き換える
    text = _RE_URL.sub('<URL>', text)

    # 電話番号を<PHONE>タグに置き換える
    text = _RE_PHONE.sub('<PHONE>', text)
    
    # 見込みを<ESTIMATE>タグに置き換える
    text = _RE_ESTIMATE.sub('<ESTIMATE>', text)

    # 連続年度にスペースを
    text = _replace_year_sequence_with_token(text)

    # クオーターを<FISCAL>タグに置き換える
    text = _RE_FISCAL.sub('<FISCAL>', text)

    # 期間を<PERIOD>タグに置き換える
    text = _RE_TERM.sub('<PERIOD>', text)

    # 年度期間を<TERM>タグに置き換える
    text = _RE_PERIOD.sub('<PERIOD>', text)

    # 日付を<DATE>タグに置き換える
    text = _RE_DATE.sub('<DATE>', text)

    # 年を<YEAR>タグに置き換える
    text = _RE_YEAR.sub('<YEAR>', text)

    # 月を<MONTH>タグに置き換える
    text = _RE_MONTH.sub('<MONTH>', text)
    
    # 日を<DAY>タグに置き換える
    text = _RE_DAY.sub('<DAY>', text)

    # 曜日を<DOW>タグに置き換える
    text = _RE_DOW.sub('<DOW>', text)
    
    # 時間を<TIME>タグに置き換える
    text = _RE_TIME.sub('<TIME>', text)

    # 数値と単位を結合
    text = _combine_number_and_unit(text)
    
    # +-と数値を結合
    text = _normalize_signed_number(text)
    
    # 1~4Qの前のスペースを
    text = _RE_Q_SPLIT.sub(' ', text)

    # 連続するハイフン（ーまたは-）を1つにする
    #text = _RE_HYPHEN.sub('―', text)

    # summarize_replace.csvに登録されてるものを。に置き換える(区切り文字として扱う)
    text = _RE_REPLACE_LIST.sub('。', text)

    # 最終クリーン
    text = _RE_MULTIPLE_PERIODS.sub('。', text)
    text = _RE_SYMBOLS.sub('', text)
    text = _RE_HEAD_REP.sub('', text)
    text = _RE_HEAD_NOTICE.sub('', text)
    text = _RE_HEAD_NOTICE2.sub('', text)
    text = _RE_HEAD_CLOSING.sub('', text)
    text = text.strip()

    return text


#### ↓　開示内にて表とだったであろうグループを特定 #############################################
def is_exclude_calendar(line):
    return re.search(r'\b(?:1[0-2]|[1-9])(?:\s+(?:1[0-2]|[1-9])){5,}\b', line)

