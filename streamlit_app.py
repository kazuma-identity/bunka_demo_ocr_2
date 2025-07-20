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
import tempfile
import concurrent.futures
import threading
from io import BytesIO

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
            st.error(f"⚠️ Gemini API エラー: {e}")

        return []

    def process_pdf_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """単一ページを処理"""
        # テキスト抽出
        text = page.extract_text()
        if not text:
            return []

        # 建具表セクションを識別
        if '建具表' in text or '記号' in text:
            # Geminiで抽出
            fixtures = self.extract_with_gemini(text)
            return fixtures
        
        return []

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

    # Get specific PDF file
    pdf_file = Path("data/image.pdf")
    
    # Alternative: Use a fixed version if available
    # pdf_file = Path("data/image_fixed.pdf") if Path("data/image_fixed.pdf").exists() else Path("data/image.pdf")
    
    if not pdf_file.exists():
        st.error("data/image.pdf が見つかりません")
        return

    # Main area for PDF processing
    st.info(f"📄 処理対象: {pdf_file.name}")
    
    if st.button("🔍 抽出開始", type="primary", use_container_width=True):
        # Initialize parser
        try:
            parser = LLMFixtureParser()
            
            # Process PDF
            all_fixtures = []
            page_results = {}
            progress_container = st.container()
            
            # Check if screenshots exist
            screenshot_dir = Path("data/pdf_screenshots")
            use_screenshots = screenshot_dir.exists() and len(list(screenshot_dir.glob("page_*.png"))) > 0
            
            if use_screenshots:
                # Use pre-made screenshots
                st.info("スクリーンショットを使用します...")
                progress_bar = st.progress(0, text="スクリーンショットを読み込み中...")
                
                # Get all screenshot files
                screenshot_files = sorted(screenshot_dir.glob("page_*.png"))
                total_pages = len(screenshot_files)
                
                # Extract text from PDF using pdfplumber
                page_texts = []
                with pdfplumber.open(str(pdf_file)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        page_texts.append(text)
                
                # Prepare page data with screenshots
                page_data = []
                for i, (screenshot_file, text) in enumerate(zip(screenshot_files, page_texts)):
                    page_num = i + 1
                    progress_bar.progress(page_num / total_pages, text=f"ページ {page_num}/{total_pages} を準備中...")
                    
                    # Read screenshot
                    with open(screenshot_file, 'rb') as f:
                        img_buffer = BytesIO(f.read())
                        img_buffer.seek(0)
                    
                    page_data.append({
                        'page_num': page_num,
                        'image_buffer': img_buffer,  # Screenshot buffer
                        'text': text,
                        'has_fixtures': '建具表' in text or '記号' in text if text else False
                    })
            else:
                # Fallback to pdf2image conversion
                st.warning("スクリーンショットが見つかりません。pdf2imageで変換します...")
                progress_bar = st.progress(0, text="PDFをPNG画像に変換中...")
                
                try:
                    from pdf2image import convert_from_path
                    
                    # Convert with optimized settings
                    png_images = convert_from_path(
                        pdf_path=str(pdf_file), 
                        dpi=200,
                        fmt='png',
                        thread_count=1,
                        use_pdftocairo=False,
                        strict=False
                    )
                    
                    total_pages = len(png_images)
                    
                    # Save PNG images in memory
                    png_buffers = []
                    for i, img in enumerate(png_images):
                        progress_bar.progress((i + 1) / total_pages, text=f"PNG画像を保存中... {i + 1}/{total_pages}")
                        buffer = BytesIO()
                        img.save(buffer, format='PNG', optimize=True, quality=95)
                        buffer.seek(0)
                        png_buffers.append(buffer)
                    
                    # Extract text from PDF
                    page_texts = []
                    with pdfplumber.open(str(pdf_file)) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            page_texts.append(text)
                    
                    # Prepare page data
                    page_data = []
                    for page_num, (png_buffer, text) in enumerate(zip(png_buffers, page_texts), 1):
                        page_data.append({
                            'page_num': page_num,
                            'image_buffer': png_buffer,
                            'text': text,
                            'has_fixtures': '建具表' in text or '記号' in text if text else False
                        })
                    
                except Exception as e:
                    st.error(f"PDF変換エラー: {e}")
                    st.info("data/pdf_screenshots/page_1.png, page_2.png... としてスクリーンショットを保存してください")
                    return
            
            # Display page images first
            with progress_container:
                st.subheader("処理状況")
                
                # Create placeholders for each page
                page_containers = {}
                for page_info in page_data:
                    page_num = page_info['page_num']
                    
                    # Create columns for page image and results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"**ページ {page_num}:**")
                        if 'image_buffer' in page_info and page_info['image_buffer']:
                            # Display PNG image from buffer
                            st.image(page_info['image_buffer'], caption=f"Page {page_num}", use_container_width=True)
                        else:
                            st.info("ページ画像を表示できません")
                    
                    with col2:
                        st.markdown("**抽出結果:**")
                        # Create placeholder for results
                        page_containers[page_num] = st.container()
                        with page_containers[page_num]:
                            if page_info['has_fixtures']:
                                st.info("🔄 処理待機中...")
                            else:
                                st.info("このページに建具表はありません")
                    
                    st.divider()
            
            # Process pages with Gemini API in parallel
            progress_bar.progress(0, text="建具情報を抽出中...")
            
            # Process pages with fixtures
            pages_with_fixtures = [p for p in page_data if p['has_fixtures']]
            total_to_process = len(pages_with_fixtures)
            
            # First, send all requests to Gemini in parallel
            with st.spinner(f"Gemini APIに{total_to_process}ページ分のリクエストを送信中..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all extraction tasks
                    future_to_page = {}
                    for page_info in pages_with_fixtures:
                        future = executor.submit(parser.extract_with_gemini, page_info['text'])
                        future_to_page[future] = page_info['page_num']
                    
                    # Collect results as they complete
                    api_results = {}
                    completed = 0
                    
                    for future in concurrent.futures.as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            result = future.result()
                            api_results[page_num] = result
                        except Exception as e:
                            api_results[page_num] = {'error': str(e)}
                        
                        completed += 1
                        progress_bar.progress(completed / total_to_process, 
                                            text=f"API処理完了: {completed}/{total_to_process} ページ")
            
            # Then update UI sequentially
            st.info("UI を更新中...")
            for page_info in pages_with_fixtures:
                page_num = page_info['page_num']
                
                # Get the result from API calls
                result = api_results.get(page_num, None)
                
                # Update the placeholder with results
                with page_containers[page_num]:
                    # Clear the waiting message
                    page_containers[page_num].empty()
                    
                    if isinstance(result, dict) and 'error' in result:
                        st.error(f"エラー: {result['error']}")
                    elif result:
                        # Display extracted data
                        fixtures = result
                        page_results[page_num] = fixtures
                        all_fixtures.extend(fixtures)
                        
                        for i, fixture in enumerate(fixtures):
                            with st.expander(f"建具 {i+1}: {fixture.get('記号', 'N/A')}", expanded=False):
                                for key, value in fixture.items():
                                    st.text(f"{key}: {value}")
                    else:
                        st.warning("建具情報が見つかりませんでした")
            
            # Clean results
            all_fixtures = parser.validate_and_clean(all_fixtures)
            
            # Store results
            st.session_state.results[pdf_file.name] = all_fixtures
            
            st.success("✅ 抽出完了！")
            st.rerun()
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    # Display results at the bottom
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
                    
                    # Calculate accuracy
                    total_correct = 0
                    total_items = len(answer_df)
                    
                    # Create comparison dataframe
                    comparison_data = []
                    
                    for _, answer_row in answer_df.iterrows():
                        symbol = answer_row['記号']
                        
                        # Find matching row in extracted data
                        extracted_rows = df[df['記号'] == symbol]
                        
                        if not extracted_rows.empty:
                            extracted_row = extracted_rows.iloc[0]
                            
                            # Compare each field
                            field_results = {}
                            field_results['記号'] = symbol
                            
                            # Check each field
                            is_correct = True
                            for field in ['数量', '室名', '型式', '見込', '仕上げ', 'ガラス', '付属金物', '備考']:
                                answer_val = str(answer_row.get(field, '')).strip()
                                extracted_val = str(extracted_row.get(field, '')).strip()
                                
                                # Handle empty values
                                if answer_val in ['nan', 'NaN', '']:
                                    answer_val = ''
                                if extracted_val in ['nan', 'NaN', '']:
                                    extracted_val = ''
                                
                                # Compare
                                if answer_val == extracted_val:
                                    field_results[field] = '✅'
                                else:
                                    field_results[field] = f'❌ (正解: {answer_val}, 抽出: {extracted_val})'
                                    is_correct = False
                            
                            if is_correct:
                                total_correct += 1
                            
                            comparison_data.append(field_results)
                        else:
                            # Missing item
                            comparison_data.append({
                                '記号': symbol,
                                '数量': '❌ (未抽出)',
                                '室名': '❌ (未抽出)',
                                '型式': '❌ (未抽出)',
                                '見込': '❌ (未抽出)',
                                '仕上げ': '❌ (未抽出)',
                                'ガラス': '❌ (未抽出)',
                                '付属金物': '❌ (未抽出)',
                                '備考': '❌ (未抽出)'
                            })
                    
                    # Calculate and display accuracy
                    accuracy = (total_correct / total_items) * 100 if total_items > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("正解数", f"{total_correct} / {total_items}")
                    with col2:
                        st.metric("正解率", f"{accuracy:.1f}%")
                    with col3:
                        st.metric("抽出数", f"{len(df)} 件")
                    
                    # Show detailed comparison
                    with st.expander("詳細な比較結果", expanded=False):
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Download comparison CSV
                        comparison_csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 比較結果をダウンロード",
                            data=comparison_csv,
                            file_name="comparison_results.csv",
                            mime='text/csv'
                        )
                    
                except Exception as e:
                    st.error(f"正解データの読み込みエラー: {e}")
                
            else:
                st.info("建具表が見つかりませんでした")

if __name__ == "__main__":
    main()