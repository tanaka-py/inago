# 文章中のヘッダーとフッターのパターン設定
from typing import List, Tuple

PATTERNS_HEADER: List[Tuple[str, str]] = [
        (r'https?://[a-zA-Z0-9.-]+\.(?:com|jp|net|org|co\.jp|biz|info|gov|edu|io|ai)(?:/[A-Za-z/]*)?(?=[^A-Za-z/]|$)', 'url'),  # URLを適切に切る
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}', 'email'),  # メールアドレス
        (r'(TEL|ＴＥＬ|T E L|電話番号|電話|電 話)', 'phone_header'),  # 電話ヘッダ
        (r'(\（?[\d０-９]{2,5}\）?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}|\bTEL\b[\s\-−－]?\(?[\d０-９]{2,5}\)?[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4}[\s\-−－]?[0-9０-９]{1,4})', 'phone_number'),  # 電話番号
        (r'(プライム市場、福証)', '福証')  # 福証ヘッダ
    ]

PATTERNS_FOTTER: List[Tuple[str, str]] = [
    (r'■本件に関するお問い合わせ先', 'fotter1'),
    (r'【本件に関する問合せ】', 'fotter2'),
    (r'本件に関するお問合せ', 'fotter3')
]