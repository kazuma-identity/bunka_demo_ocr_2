#!/usr/bin/env python3
"""
Streamlit app for PDF fixture table extraction using Gemini
"""

import streamlit as st
import pandas as pd
import pdfplumber
import google.generativeai as genai
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from io import BytesIO
import concurrent.futures
import time
import hashlib

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="建具表抽出アプリ",
    page_icon="🏗️",
    layout="wide"
)

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
与えられたテキストから建具情報を抽出し、構造化された**JSON形式**で出力してください。
出力は必ずJSON形式で、jsonのみを出力してください。
"""

        self.extract_prompt_template = """
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

        except Exception:
            # エラーログを軽量化（UI更新なし）
            pass

        return []

    def process_pdf_page(self, page: Any, page_num: int = 0) -> List[Dict[str, Any]]:
        """単一ページを処理"""
        # テキスト抽出
        text = page.extract_text()
        if not text:
            return []

        # 建具表セクションを識別
        if '建具表' in text or '記号' in text:
            # Geminiで抽出
            fixtures = self.extract_with_gemini(text)
            # page_numをログ用に使用（未使用警告を回避）
            _ = page_num
            return fixtures

        return []

    def process_page_parallel(self, page_info: Dict[str, Any]) -> Dict[str, Any]:
        """並列処理用のページ処理関数"""
        page_num = page_info['page_num']
        text = page_info['text']

        start_time = time.time()
        print(f"🔥 スレッド開始: ページ {page_num} スレッド実行開始 ({start_time:.3f})")

        try:
            print(f"🌐 API呼び出し開始: ページ {page_num} Gemini API呼び出し開始 ({time.time():.3f})")
            fixtures = self.extract_with_gemini(text)
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ API完了: ページ {page_num} 処理完了 ({end_time:.3f}) 処理時間: {duration:.3f}秒")
            return {
                'page_num': page_num,
                'fixtures': fixtures,
                'success': True,
                'error': None,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ エラー: ページ {page_num} でエラー ({end_time:.3f}) 処理時間: {duration:.3f}秒")
            return {
                'page_num': page_num,
                'fixtures': [],
                'success': False,
                'error': str(e),
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }

    def process_pages_parallel_batch(self, pages_data: List[Dict[str, Any]], max_workers: int = 3, page_containers: Dict = None, progress_callback=None) -> Dict[int, Dict[str, Any]]:
        """複数ページを並列処理（バッチ処理）"""
        # 全ページを処理対象とする
        pages_to_process = pages_data

        if not pages_to_process:
            return {}

        results = {}
        completed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 並列処理を開始
            future_to_page = {
                executor.submit(self.process_page_parallel, page_info): page_info['page_num']
                for page_info in pages_to_process
            }

            # 結果を収集（完了順にコールバック実行）
            for future in concurrent.futures.as_completed(future_to_page):
                result = future.result()
                page_num = result['page_num']
                results[page_num] = result
                completed_count += 1
                
                # コールバック実行（リアルタイム更新用）
                if progress_callback:
                    progress_callback(page_num, result, completed_count, len(pages_to_process))

        return results

    def extract_all_text_fast(self, pdf_path: str) -> List[Dict[str, Any]]:
        """全ページのテキストを高速抽出（text_extractor.py方式）"""
        text_content = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    text_content.append({
                        'page': page_num,
                        'text': text.strip() if text else ''
                    })
        except Exception:
            pass

        return text_content

    def load_screenshot(self, screenshot_path: Path) -> BytesIO:
        """スクリーンショットを読み込み（並列処理用）"""
        try:
            with open(screenshot_path, 'rb') as f:
                img_buffer = BytesIO(f.read())
                img_buffer.seek(0)
                return img_buffer
        except Exception:
            return BytesIO()

    def load_screenshots_parallel(self, screenshot_files: List[Path], max_workers: int = 4) -> List[BytesIO]:
        """スクリーンショットを並列読み込み"""
        screenshots = [BytesIO()] * len(screenshot_files)  # 順序を保持

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 各スクリーンショットの読み込みを並列実行
            future_to_index = {
                executor.submit(self.load_screenshot, screenshot_file): i
                for i, screenshot_file in enumerate(screenshot_files)
            }

            # 結果を順序通りに格納
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    img_buffer = future.result()
                    screenshots[index] = img_buffer
                except Exception:
                    screenshots[index] = BytesIO()

        return screenshots

    def get_pdf_hash(self, pdf_path: str) -> str:
        """PDFファイルのハッシュを計算（キャッシュ判定用）"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                return hashlib.md5(pdf_bytes).hexdigest()
        except Exception:
            return ""

    def get_cached_texts_or_extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        """キャッシュからテキストを取得するか、新規抽出"""
        # PDFのハッシュを計算
        current_hash = self.get_pdf_hash(pdf_path)

        # キャッシュが有効かチェック
        if (current_hash and
            current_hash == st.session_state.get('pdf_hash', '') and
            st.session_state.get('text_cache', {})):

            st.info("📋 キャッシュからテキストデータを読み込み中...")
            return st.session_state.text_cache

        # 新規抽出（高速）
        st.info("📄 PDFからテキストを高速抽出中...")
        text_content = self.extract_all_text_fast(pdf_path)

        # キャッシュに保存
        if current_hash:
            st.session_state.pdf_hash = current_hash
            st.session_state.text_cache = text_content

        return text_content

    def validate_and_clean(self, fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽出結果の検証とクリーニング"""
        cleaned = []

        for fixture in fixtures:
            # 記号の検証
            if not fixture.get('記号'):
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

def main():
    st.title("🏗️ 建具表抽出アプリ")
    st.markdown("PDFから建具表を自動抽出します")

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {}

    # キャッシュ機能のためのセッション状態初期化
    if 'text_cache' not in st.session_state:
        st.session_state.text_cache = {}
    if 'pdf_hash' not in st.session_state:
        st.session_state.pdf_hash = ""

    # Get specific PDF file
    pdf_file = Path("data/image.pdf")

    # Alternative: Use a fixed version if available
    # pdf_file = Path("data/image_fixed.pdf") if Path("data/image_fixed.pdf").exists() else Path("data/image.pdf")

    if not pdf_file.exists():
        st.error("data/image.pdf が見つかりません")
        return

    # Main area for PDF processing
    st.info(f"📄 処理対象: {pdf_file.name}")

    # Processing mode selection
    col1, col2 = st.columns(2)
    with col1:
        processing_mode = st.radio(
            "処理モード",
            ["🚀 並列処理", "📝 順次処理"],
            index=0,
            help="並列処理: 複数ページを同時に処理（高速）\n順次処理: 1ページずつ処理（安定）"
        )

    with col2:
        if processing_mode == "🚀 並列処理":
            max_workers = st.slider(
                "同時処理数",
                min_value=1,
                max_value=4,
                value=4,
                help="同時に処理するページ数（4ページ対応）"
            )
        else:
            max_workers = 1

    if st.button("🔍 抽出開始", type="primary", use_container_width=True):
        # Initialize parser
        try:
            parser = LLMFixtureParser()

            # Process PDF
            all_fixtures = []
            page_results = {}

            # Use screenshots only
            screenshot_dir = Path("data/pdf_screenshots")
            screenshot_files = sorted(screenshot_dir.glob("page_*.png"))

            if not screenshot_files:
                st.error("スクリーンショットが見つかりません")
                st.info("data/pdf_screenshots/page_1.png, page_2.png... としてスクリーンショットを保存してください")
                return

            st.info("スクリーンショットを使用します...")
            progress_bar = st.progress(0, text="スクリーンショットを読み込み中...")

            total_pages = len(screenshot_files)

            # テキストを高速抽出
            text_content = parser.get_cached_texts_or_extract(str(pdf_file))

            # スクリーンショットを並列読み込み
            progress_bar.progress(0.3, text="スクリーンショットを並列読み込み中...")
            screenshot_workers = min(6, max_workers + 2)  # スクリーンショット読み込み用
            screenshots = parser.load_screenshots_parallel(screenshot_files, screenshot_workers)

            # ページデータを準備（テキスト抽出済み）
            page_data = []
            for i, (text_info, img_buffer) in enumerate(zip(text_content, screenshots)):
                page_num = text_info['page']
                text = text_info['text']

                progress_bar.progress(0.5 + (0.3 * i / len(text_content)), text=f"ページデータ準備中: {page_num}/{len(text_content)}")

                page_data.append({
                    'page_num': page_num,
                    'image_buffer': img_buffer,
                    'text': text,
                    'has_fixtures': True  # 全ページ処理対象
                })

            # ページ数を表示
            col1, col2 = st.columns(2)
            with col1:
                st.metric("総ページ数", f"{len(page_data)} ページ")
            with col2:
                st.metric("処理対象", f"{len(page_data)} ページ")

            # Create page containers for status display
            st.markdown("---")
            st.subheader("📄 ページ別処理状況")

            page_containers = {}
            for page_info in page_data:
                page_num = page_info['page_num']

                # トグルでページ表示を制御
                with st.expander(f"📄 ページ {page_num}", expanded=True):
                    # Create columns for page image and results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown(f"**ページ {page_num} 画像:**")
                        if 'image_buffer' in page_info and page_info['image_buffer']:
                            # Display PNG image from buffer
                            st.image(page_info['image_buffer'], caption=f"Page {page_num}", use_container_width=True)
                        else:
                            st.info("ページ画像を表示できません")

                    with col2:
                        st.markdown("**処理状況:**")
                        # Create placeholder for results
                        page_containers[page_num] = st.container()
                        with page_containers[page_num]:
                            st.info("🔄 処理待機中...")

            # Process pages based on selected mode
            if processing_mode == "🚀 並列処理":
                # Parallel processing
                progress_bar.progress(0, text="並列処理で建具情報を抽出中...")

                # Process all pages in parallel
                start_time = time.time()

                # 並列処理でバッチ実行（テキスト抽出済みのため高速）
                progress_bar.progress(0.8, text=f"🚀 {max_workers}つのスレッドでGemini API並列処理を開始...")

                # リアルタイム更新用のコールバック関数
                def update_page_result(page_num, result, completed, total):
                    # 進捗バー更新
                    progress = 0.8 + (0.2 * completed / total)
                    progress_bar.progress(progress, text=f"処理中: {completed}/{total} ページ完了")
                    
                    # ページコンテナ更新
                    if page_containers and page_num in page_containers:
                        with page_containers[page_num]:
                            page_containers[page_num].empty()
                            
                            duration = result.get('duration', 0)

                            if result['success']:
                                fixtures = result['fixtures']
                                if fixtures:
                                    page_results[page_num] = fixtures
                                    all_fixtures.extend(fixtures)
                                    st.success(f"✅ ページ {page_num}: {len(fixtures)}件の建具を抽出 (処理時間: {duration:.1f}秒)")
                                    
                                    # 抽出結果を即座に表示
                                    with st.expander(f"📋 抽出結果", expanded=False):
                                        fixtures_df = pd.DataFrame(fixtures)
                                        st.dataframe(fixtures_df, use_container_width=True)
                                else:
                                    st.info(f"ページ {page_num}: 建具情報なし (処理時間: {duration:.1f}秒)")
                            else:
                                st.error(f"❌ ページ {page_num}: エラー - {result.get('error', '')} (処理時間: {duration:.1f}秒)")

                # 並列処理実行（コールバック付き）
                parallel_results = parser.process_pages_parallel_batch(
                    page_data, 
                    max_workers=max_workers,
                    page_containers=page_containers,
                    progress_callback=update_page_result
                )

                processing_time = time.time() - start_time

                # Show processing time and statistics
                total_pages = len(page_data)
                processed_pages = len([p for p in page_data if p['has_fixtures']])
                total_fixtures = len(all_fixtures)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("処理時間", f"{processing_time:.2f}秒")
                with col2:
                    st.metric("処理済みページ", f"{processed_pages}/{total_pages}")
                with col3:
                    st.metric("抽出建具数", f"{total_fixtures}件")

                st.success(f"🚀 並列処理完了！")

            else:
                # Sequential processing (original method)
                progress_bar.progress(0, text="建具情報を抽出中...")

                # Process all pages
                total_to_process = len(page_data)

                start_time = time.time()

                for idx, page_info in enumerate(page_data):
                    page_num = page_info['page_num']

                    # Update progress
                    progress_bar.progress((idx + 1) / total_to_process,
                                        text=f"処理中: {idx + 1}/{total_to_process} ページ")

                    # Extract with Gemini (簡潔表示)
                    with page_containers[page_num]:
                        page_containers[page_num].empty()
                        try:
                            fixtures = parser.extract_with_gemini(page_info['text'])

                            if fixtures:
                                page_results[page_num] = fixtures
                                all_fixtures.extend(fixtures)
                                st.success(f"✅ ページ {page_num}: {len(fixtures)}件の建具を抽出")
                            else:
                                st.info(f"ページ {page_num}: 建具情報なし")

                        except Exception as e:
                            st.error(f"❌ ページ {page_num}: エラー")

                processing_time = time.time() - start_time

                # Show processing time and statistics
                total_pages = len(page_data)
                processed_pages = len([p for p in page_data if p['has_fixtures']])
                total_fixtures = len(all_fixtures)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("処理時間", f"{processing_time:.2f}秒")
                with col2:
                    st.metric("処理済みページ", f"{processed_pages}/{total_pages}")
                with col3:
                    st.metric("抽出建具数", f"{total_fixtures}件")

                st.success(f"📝 順次処理完了！")

            # Clean results
            all_fixtures = parser.validate_and_clean(all_fixtures)

            # Store results
            st.session_state.results[pdf_file.name] = all_fixtures

            st.success("✅ 抽出完了！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    # Display results at the bottom (after processing is complete)
    if st.session_state.results:
        st.markdown("---")
        st.header("📊 抽出結果")

        for pdf_name, fixtures in st.session_state.results.items():
            if fixtures:
                # Convert to DataFrame
                df = pd.DataFrame(fixtures)

                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("建具数", f"{len(fixtures)} 件")
                with col2:
                    st.metric("総数量", f"{sum(f.get('数量', 0) for f in fixtures)} 個")

                # Display table
                st.dataframe(df, use_container_width=True)

                # Download button for CSV
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSVをダウンロード",
                    data=csv,
                    file_name=f"{pdf_name.replace('.pdf', '')}_fixtures.csv",
                    mime='text/csv'
                )

                # Calculate accuracy against answer.csv
                st.markdown("---")
                st.subheader("🎯 正解率の評価")

                try:
                    # Load answer.csv
                    answer_df = pd.read_csv("data/answer.csv")

                    # フィールド定義
                    fields = ['数量', '室名', '型式', '見込', '仕上げ', 'ガラス', '付属金物', '備考']

                    # 評価マトリックス用のデータ構造
                    comparison_matrix = []
                    field_accuracy = {field: {'correct': 0, 'total': 0} for field in fields}

                    total_items = len(answer_df)
                    total_cells = total_items * len(fields)
                    correct_cells = 0

                    for _, answer_row in answer_df.iterrows():
                        symbol = answer_row['記号']

                        # Find matching row in extracted data
                        extracted_rows = df[df['記号'] == symbol]

                        row_result = {'記号': symbol}

                        if not extracted_rows.empty:
                            extracted_row = extracted_rows.iloc[0]

                            # 各フィールドを評価
                            for field in fields:
                                answer_val = str(answer_row.get(field, '')).strip()
                                extracted_val = str(extracted_row.get(field, '')).strip()

                                # Handle empty values
                                if answer_val in ['nan', 'NaN', '']:
                                    answer_val = ''
                                if extracted_val in ['nan', 'NaN', '']:
                                    extracted_val = ''

                                # 評価
                                field_accuracy[field]['total'] += 1
                                if answer_val == extracted_val:
                                    row_result[field] = '✅'
                                    field_accuracy[field]['correct'] += 1
                                    correct_cells += 1
                                else:
                                    row_result[field] = '❌'
                        else:
                            # 未抽出の場合、全フィールドが不正解
                            for field in fields:
                                row_result[field] = '❌'
                                field_accuracy[field]['total'] += 1

                        comparison_matrix.append(row_result)

                    # 全体とフィールド別正解率を計算
                    overall_accuracy = (correct_cells / total_cells) * 100 if total_cells > 0 else 0

                    # メトリクス表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("全体正解率", f"{overall_accuracy:.1f}%")
                    with col2:
                        st.metric("正解セル数", f"{correct_cells} / {total_cells}")
                    with col3:
                        st.metric("抽出建具数", f"{len(df)} / {total_items} 件")

                    # フィールド別正解率表示
                    st.subheader("📊 フィールド別正解率")
                    field_accuracy_data = []
                    for field in fields:
                        accuracy = (field_accuracy[field]['correct'] / field_accuracy[field]['total']) * 100 if field_accuracy[field]['total'] > 0 else 0
                        field_accuracy_data.append({
                            'フィールド': field,
                            '正解数': field_accuracy[field]['correct'],
                            '総数': field_accuracy[field]['total'],
                            '正解率(%)': f"{accuracy:.1f}"
                        })

                    field_accuracy_df = pd.DataFrame(field_accuracy_data)
                    st.dataframe(field_accuracy_df, use_container_width=True)

                    # 評価マトリックス表示
                    with st.expander("📋 評価マトリックス（行×列）", expanded=False):
                        st.markdown("**凡例:** ✅ = 正解, ❌ = 不正解")
                        comparison_df = pd.DataFrame(comparison_matrix)
                        st.dataframe(comparison_df, use_container_width=True)

                        # Download comparison CSV
                        comparison_csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 評価マトリックスをダウンロード",
                            data=comparison_csv,
                            file_name="evaluation_matrix.csv",
                            mime='text/csv'
                        )

                        # フィールド別正解率もダウンロード可能に
                        field_csv = field_accuracy_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 フィールド別正解率をダウンロード",
                            data=field_csv,
                            file_name="field_accuracy.csv",
                            mime='text/csv'
                        )

                except Exception as e:
                    st.error(f"正解データの読み込みエラー: {e}")

            else:
                st.info("建具表が見つかりませんでした")


if __name__ == "__main__":
    main()
