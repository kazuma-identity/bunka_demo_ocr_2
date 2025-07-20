#!/usr/bin/env python3
"""
pdfplumberで読み取った建具表テキストを解析してCSV形式に変換
Usage: python parse_fixture_table.py --pdf <PDF_PATH> [--output fixtures.csv]
"""

import argparse
import pdfplumber
import re
import csv
from collections import OrderedDict
import pandas as pd

class FixtureTableParser:
    def __init__(self):
        self.fixtures = []
        self.field_mapping = {
            '記号': 'symbol',
            '数量': 'quantity',
            '室名': 'room_name',
            '室 名': 'room_name',
            '型式': 'model',
            '見込': 'depth',
            '姿図': 'figure',
            '姿 図': 'figure',
            '仕上': 'finish',
            '仕 上': 'finish',
            'ガラス': 'glass',
            '付属金物': 'hardware',
            '備考': 'remarks'
        }

    def extract_text_with_pdfplumber(self, pdf_path):
        """pdfplumberでPDFからテキストを抽出"""
        all_pages_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    all_pages_text.append({
                        'page': page_num,
                        'text': text
                    })

                # テーブルも試して抽出
                tables = page.extract_tables()
                if tables:
                    all_pages_text[-1]['tables'] = tables

        return all_pages_text

    def parse_fixture_table_text(self, text):
        """建具表のテキストを解析"""
        lines = text.split('\n')
        fixtures = []

        # 現在処理中の建具データを保持
        current_fixtures = []
        current_field = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 記号行の検出（複数の建具が横並び）
            if '記号' in line and '数量' in line:
                # 記号と数量を抽出
                symbols_quantities = self._extract_symbols_quantities(line)

                # 既存の建具データがあれば保存
                if current_fixtures:
                    fixtures.extend(current_fixtures)

                # 新しい建具データを初期化
                current_fixtures = []
                for symbol, quantity in symbols_quantities:
                    current_fixtures.append({
                        '記号': symbol,
                        '数量': quantity,
                        '型式': '',
                        '見込': '',
                        '姿図': '',
                        '仕上げ': '',
                        'ガラス': '',
                        '付属金物': '',
                        '備考': ''
                    })
                current_field = '記号'

            # 他のフィールドの処理
            else:
                field_name = self._detect_field_name(line)
                if field_name:
                    current_field = field_name
                    values = self._extract_field_values(line, field_name, len(current_fixtures))

                    # 各建具に値を設定
                    for i, value in enumerate(values):
                        if i < len(current_fixtures):
                            current_fixtures[i][field_name] = value

                # フィールド名がない行は前のフィールドの続き
                elif current_field and current_fixtures:
                    # 数値や記号を含む行を値として処理
                    if re.search(r'[0-9,]+|FL|mm|-', line):
                        values = self._split_values(line, len(current_fixtures))
                        for i, value in enumerate(values):
                            if i < len(current_fixtures) and value:
                                # 既存の値に追加または上書き
                                if current_fixtures[i].get(current_field):
                                    current_fixtures[i][current_field] += ' ' + value
                                else:
                                    current_fixtures[i][current_field] = value

        # 最後の建具データを追加
        if current_fixtures:
            fixtures.extend(current_fixtures)

        return fixtures

    def _extract_symbols_quantities(self, line):
        """記号と数量を抽出"""
        # パターン: 記号 数量 SS-3 2 記号 数量 SS-4 2
        symbols_quantities = []

        # 記号パターン（例: SS-3, AD-1）
        pattern = r'([A-Z]{2,3}-\d+)\s+(\d+)'
        matches = re.findall(pattern, line)

        for match in matches:
            symbol = match[0]
            quantity = int(match[1])
            symbols_quantities.append((symbol, quantity))

        return symbols_quantities

    def _detect_field_name(self, line):
        """行からフィールド名を検出"""
        for jp_name, en_name in self.field_mapping.items():
            if line.startswith(jp_name):
                return jp_name.replace(' ', '')
        return None

    def _extract_field_values(self, line, field_name, num_fixtures):
        """フィールドの値を抽出（複数の建具分）"""
        # フィールド名を除去
        for jp_name in self.field_mapping.keys():
            if jp_name in line:
                line = line.replace(jp_name, '').strip()
                break

        # 値を分割
        return self._split_values(line, num_fixtures)

    def _split_values(self, line, num_fixtures):
        """行を複数の値に分割"""
        if not line.strip():
            return [''] * num_fixtures

        # 特殊な区切り文字で分割を試みる
        if '：' in line:
            parts = [p.strip() for p in line.split('：')]
        elif '  ' in line:  # 複数スペース
            parts = [p.strip() for p in line.split('  ') if p.strip()]
        elif num_fixtures == 1:
            parts = [line.strip()]
        else:
            # 均等に分割を試みる
            parts = self._split_equally(line, num_fixtures)

        # 必要な数に調整
        while len(parts) < num_fixtures:
            parts.append('')

        return parts[:num_fixtures]

    def _split_equally(self, text, num_parts):
        """テキストを均等に分割"""
        if num_parts <= 1:
            return [text]

        # 数値を含む部分で分割を試みる
        numbers = re.findall(r'\d+[,\d]*', text)
        if len(numbers) >= num_parts:
            return numbers[:num_parts]

        # それ以外は文字数で均等分割
        length = len(text)
        part_length = length // num_parts
        parts = []

        for i in range(num_parts):
            start = i * part_length
            end = start + part_length if i < num_parts - 1 else length
            parts.append(text[start:end].strip())

        return parts

    def parse_table_format(self, tables):
        """pdfplumberのテーブル形式を解析"""
        fixtures = []

        for table in tables:
            if not table:
                continue

            # テーブルの構造を分析
            headers = []
            data_rows = []

            for row in table:
                if any('記号' in str(cell) for cell in row if cell):
                    headers = row
                else:
                    data_rows.append(row)

            # データを構造化
            if headers and data_rows:
                # 記号の列を見つける
                symbol_indices = [i for i, h in enumerate(headers) if h and '記号' in str(h)]

                for idx in symbol_indices:
                    fixture = self._extract_fixture_from_column(headers, data_rows, idx)
                    if fixture:
                        fixtures.append(fixture)

        return fixtures

    def _extract_fixture_from_column(self, headers, data_rows, start_col):
        """テーブルの列から建具情報を抽出"""
        fixture = {}

        # ヘッダーとデータのマッピング
        for row_idx, row in enumerate(data_rows):
            if start_col < len(row) and row[start_col]:
                # 対応するフィールド名を推測
                if row_idx < len(headers) and headers[row_idx]:
                    field = str(headers[row_idx])
                    for jp_name, en_name in self.field_mapping.items():
                        if jp_name in field:
                            fixture[jp_name.replace(' ', '')] = str(row[start_col])
                            break

        return fixture if fixture else None

    def merge_and_clean_fixtures(self, fixtures):
        """建具データをマージしてクリーニング"""
        cleaned = []

        for fixture in fixtures:
            # 記号が有効なもののみ
            if fixture.get('記号') and re.match(r'[A-Z]{2,3}-\d+', fixture['記号']):
                # 数量を整数に変換
                if fixture.get('数量'):
                    try:
                        fixture['数量'] = int(re.search(r'\d+', str(fixture['数量'])).group())
                    except:
                        fixture['数量'] = 1
                else:
                    fixture['数量'] = 1

                # 値のクリーニング
                for key, value in fixture.items():
                    if isinstance(value, str):
                        # 不要な文字を除去
                        value = value.replace('\n', ' ').strip()
                        value = re.sub(r'\s+', ' ', value)
                        fixture[key] = value

                cleaned.append(fixture)

        # 重複を除去
        unique_fixtures = []
        seen = set()

        for fixture in cleaned:
            key = fixture['記号']
            if key not in seen:
                seen.add(key)
                unique_fixtures.append(fixture)

        return unique_fixtures

    def save_to_csv(self, fixtures, output_path):
        """建具データをCSVに保存"""
        if not fixtures:
            print("⚠️ 建具データが見つかりませんでした。")
            return

        # フィールド名の順序
        fieldnames = ['記号', '数量', '室名', '型式', '見込', '姿図', '仕上げ', 'ガラス', '付属金物', '備考']

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for fixture in fixtures:
                # 空のフィールドを追加
                for field in fieldnames:
                    if field not in fixture:
                        fixture[field] = ''

                writer.writerow(fixture)

        print(f"✅ CSV保存完了: {output_path}")

    def process_pdf(self, pdf_path):
        """PDFを処理して建具表を抽出"""
        print(f"📄 PDFを読み込み中: {pdf_path}")

        # pdfplumberでテキスト抽出
        pages_data = self.extract_text_with_pdfplumber(pdf_path)

        all_fixtures = []

        for page_data in pages_data:
            page_num = page_data['page']
            text = page_data['text']

            print(f"\n=== ページ {page_num} ===")

            # テキスト形式の解析
            fixtures = self.parse_fixture_table_text(text)

            # テーブル形式も試す
            if 'tables' in page_data:
                table_fixtures = self.parse_table_format(page_data['tables'])
                fixtures.extend(table_fixtures)

            # ページ番号を追加
            for fixture in fixtures:
                fixture['ページ'] = page_num

            all_fixtures.extend(fixtures)

            # 検出結果を表示
            for fixture in fixtures:
                print(f"  - {fixture.get('記号', '?')}: 数量={fixture.get('数量', '?')}")

        # データのクリーニングとマージ
        cleaned_fixtures = self.merge_and_clean_fixtures(all_fixtures)

        # ソート
        cleaned_fixtures.sort(key=lambda x: (x['記号'].split('-')[0], int(x['記号'].split('-')[1])))

        return cleaned_fixtures

def main():
    parser = argparse.ArgumentParser(description="pdfplumberで読み取った建具表を解析")
    parser.add_argument("--pdf", required=True, help="入力PDFファイル")
    parser.add_argument("--output", default="fixtures_parsed.csv", help="出力CSVファイル")
    parser.add_argument("--debug", action='store_true', help="デバッグ情報を表示")
    args = parser.parse_args()

    # パーサーを初期化
    table_parser = FixtureTableParser()

    try:
        # PDF処理
        fixtures = table_parser.process_pdf(args.pdf)

        if fixtures:
            # CSV保存
            table_parser.save_to_csv(fixtures, args.output)

            # サマリー表示
            print(f"\n📊 抽出結果:")
            print(f"  建具数: {len(fixtures)}件")
            print(f"  総数量: {sum(f.get('数量', 0) for f in fixtures)}個")

            # デバッグモードでは詳細表示
            if args.debug:
                print("\n【詳細データ】")
                for fixture in fixtures:
                    print(f"\n{fixture['記号']}:")
                    for key, value in fixture.items():
                        if value:
                            print(f"  {key}: {value}")
        else:
            print("⚠️ 建具データを抽出できませんでした。")

    except Exception as e:
        print(f"❌ エラー: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
