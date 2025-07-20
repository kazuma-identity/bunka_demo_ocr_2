#!/usr/bin/env python3
"""
LLMを使用してpdfplumberで読み取った建具表を解析
各ページを個別に処理
"""

import argparse
import pdfplumber
import csv
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class LLMFixtureParser:
    def __init__(self, api_key=None):
        # Gemini設定
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            raise ValueError("GEMINI_API_KEY環境変数が設定されていません")

        # プロンプトテンプレート
        self.system_prompt = """あなたは建築図面の建具表を解析する専門家です。
与えられたテキストから建具情報を抽出し、構造化されたJSON形式で出力してください。

建具表の各項目は以下の情報を含みます：
- 記号: 建具の識別記号（例: SS-3, AD-1） 必ず ** 大文字英字大文字英字-数字の形式 ** で出力してください。
- 数量: 個数
- 室名: 設置場所
- 型式: 建具の型式や仕様、
- 見込：見込寸法
- 姿図: 空で出力してください
- 仕上げ: 表面仕上げ
- ガラス: ガラス仕様
- 付属金物: 付属する金物
- 備考: メーカー名やその他の情報

- *-*は空で出力してください
- 見込は*空*か*数値と単位*であり、一つしかありません。姿図での数値を混同しないように気を付けてください。
"""

        self.extract_prompt_template = """以下の建具表テキストから、すべての建具情報を抽出してください。

テキスト:
{text}

以下のJSON形式で出力してください：
{{
    "fixtures": [
        {{
            "記号": "SS-3",
            "数量": 2,
            "室名": "１Ｆ：研修実習室",
            "型式": "電動シャッター",
            "見込": "",
            "姿図": "",
            "仕上げ": "",
            "ガラス": "",
            "付属金物": "標準金物一式",
            "備考": "文化シャッター：御前様"
        }},
        ...
    ]
}}
"""

    def extract_with_gemini(self, text: str) -> List[Dict[str, Any]]:
        """Gemini APIを使用して抽出"""
        try:
            # プロンプトを結合
            prompt = self.system_prompt + "\n\n" + self.extract_prompt_template.format(text=text)

            response = self.model.generate_content(prompt)
            result_text = response.text

            # JSON部分を抽出
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('fixtures', [])

        except Exception as e:
            print(f"⚠️ Gemini API エラー: {e}")

        return []

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """PDFを処理して建具情報を抽出（ページごとに処理）"""
        all_fixtures = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"\n=== ページ {page_num} を処理中 ===")

                # テキスト抽出
                text = page.extract_text()
                if not text:
                    print(f"ページ {page_num}: テキストが見つかりません")
                    continue

                # 建具表セクションを識別
                if '建具表' in text or '記号' in text:
                    print("建具表を検出しました")

                    # Geminiで抽出
                    fixtures = self.extract_with_gemini(text)

                    # ページ番号を追加
                    for fixture in fixtures:
                        fixture['ページ'] = page_num

                    all_fixtures.extend(fixtures)

                    # 結果表示
                    for fixture in fixtures:
                        print(f"  - {fixture['記号']}: 数量={fixture['数量']}")

        return all_fixtures

    def validate_and_clean(self, fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽出結果の検証とクリーニング"""
        cleaned = []

        for fixture in fixtures:
            # 記号の検証
            if not fixture.get('記号') or not re.match(r'[A-Z]{2,3}-\d+', fixture['記号']):
                continue

            # 数量の検証
            try:
                fixture['数量'] = int(fixture.get('数量', 1))
            except:
                fixture['数量'] = 1

            # 空文字列の処理
            for key in fixture:
                if fixture[key] == '-' or fixture[key] == '－':
                    fixture[key] = ''
                elif isinstance(fixture[key], str):
                    fixture[key] = fixture[key].strip()

            cleaned.append(fixture)


        return cleaned

    def save_to_csv(self, fixtures: List[Dict[str, Any]], output_path: str):
        """CSVファイルに保存"""
        if not fixtures:
            print("⚠️ 保存する建具データがありません")
            return

        fieldnames = ['記号', '数量', '室名', '型式', '見込', '姿図', '仕上げ', 'ガラス', '付属金物', '備考', 'ページ']

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(fixtures)

        print(f"✅ CSV保存完了: {output_path}")

    def save_to_json(self, fixtures: List[Dict[str, Any]], output_path: str):
        """JSONファイルに保存（デバッグ用）"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON保存完了: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Geminiを使用した建具表解析")
    parser.add_argument("--pdf", required=True, help="入力PDFファイル")
    parser.add_argument("--output", default="fixtures_gemini.csv", help="出力CSVファイル")
    parser.add_argument("--api-key", help="Gemini APIキー")
    parser.add_argument("--json", action='store_true', help="JSON形式でも出力")
    args = parser.parse_args()

    try:
        # LLMパーサーを初期化
        llm_parser = LLMFixtureParser(api_key=args.api_key)

        # PDFを処理
        fixtures = llm_parser.process_pdf(args.pdf)

        # 結果をクリーニング
        fixtures = llm_parser.validate_and_clean(fixtures)

        if fixtures:
            # CSV保存
            llm_parser.save_to_csv(fixtures, args.output)

            # JSON保存（オプション）
            if args.json:
                json_path = args.output.replace('.csv', '.json')
                llm_parser.save_to_json(fixtures, json_path)

            # 簡単なサマリーのみ
            print(f"\n✅ 解析完了: {len(fixtures)}件の建具を抽出しました")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
