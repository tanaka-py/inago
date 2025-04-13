#環境変数読む用
import os
from dotenv import load_dotenv

#環境変数読込
load_dotenv()

# fastapiをimport
from fastapi import FastAPI, Request
#CORS設定をするためのimport
from fastapi.middleware.cors import CORSMiddleware
# ルーターを読み込み
from .routers import disclosure, pressrelease

#fastapiインスタンスを作成
app = FastAPI(debug=True)

#環境変数からフロントエンドのURLを取得
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:100")

#CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url], # vueサーバを指定
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"], # 必要なHTTPメソッドを指定
    allow_headers=["*"]
)

#各種ルーターを登録
app.include_router(disclosure.router, prefix='/disclosure', tags=['disclosure'])
app.include_router(pressrelease.router, prefix='/pressrelease', tags=['pressrelease'])

@app.get('/')
def start():
    return{'status': 'ok'}





