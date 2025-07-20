# 建具表抽出アプリ

PDFから建具表を自動抽出し、CSV形式で出力するStreamlitアプリケーションです。

## 機能

- PDFファイルから建具表を自動検出・抽出
- Google Gemini APIを使用した高精度なテーブル解析
- 並列処理による高速化
- 正解データとの比較・精度評価
- CSV形式でのエクスポート

## セットアップ

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/kazuma-identity/bunka_demo_ocr_2.git
cd bunka_demo_ocr_2

# Conda環境の作成（推奨）
conda create -n bunka_demo_2 python=3.9
conda activate bunka_demo_2

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. システム依存関係

PDFの画像変換にはpoppler-utilsが必要です：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# 1. https://github.com/oschwartz10612/poppler-windows/releases/ から最新版をダウンロード
# 2. 解凍してC:\poppler などに配置
# 3. C:\poppler\Library\bin をシステムのPATHに追加
# 4. コマンドプロンプトを再起動

# WSL (Windows Subsystem for Linux)
sudo apt-get update
sudo apt-get install poppler-utils
```

インストール確認：
```bash
pdftoppm -v
```

### 3. 環境変数の設定

`.env`ファイルを作成し、Gemini APIキーを設定：

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## 使用方法

### Streamlitアプリの起動

```bash
streamlit run streamlit_app.py
```

### コマンドラインツール

個別のPDF処理：

```bash
# LLMを使用した建具表抽出
python src/llm_table.py --pdf data/image.pdf --output fixtures.csv

# PDFからテキスト抽出
python src/pdf_text_extractor.py

# PDFをCSVに変換（画像ベース）
python pdf_to_csv_converter.py
```

## ファイル構成

```
bunka_demo_ocr_2/
├── streamlit_app.py          # メインのStreamlitアプリ
├── streamlit_app_pdf2image.py # pdf2image版アプリ
├── pdf_to_csv_converter.py    # PDF→CSV変換スクリプト
├── src/
│   ├── llm_table.py          # LLMベースの建具表抽出
│   ├── llm_table_simple.py   # 簡易版
│   ├── pdf_text_extractor.py # PDFテキスト抽出
│   └── simple_pdf_to_csv.py  # シンプルなPDF→CSV変換
├── data/
│   ├── image.pdf             # サンプルPDF
│   └── answer.csv            # 正解データ
├── requirements.txt          # Python依存関係
└── .env                      # 環境変数（要作成）
```

## 文字化け対策

PDFの表示で文字化けが発生する場合：

1. スクリーンショット方式：
   - `data/pdf_screenshots/`ディレクトリを作成
   - 各ページのスクリーンショットを`page_1.png`, `page_2.png`...として保存
   - アプリが自動的にスクリーンショットを使用

2. フォント埋め込みの確認：
   ```bash
   pdffonts data/image.pdf
   ```

## トラブルシューティング

### pdf2imageエラー
- poppler-utilsがインストールされているか確認
- `conda install -c conda-forge poppler`でインストール

### Gemini APIエラー
- APIキーが正しく設定されているか確認
- APIの利用制限に達していないか確認

### メモリ不足
- PDFのページ数が多い場合は、DPIを下げる（200→150）
- 並列処理数を減らす

## ライセンス

このプロジェクトはMITライセンスで公開されています。