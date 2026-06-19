# 台股市場資訊儀表板期末報告

## 1. 專案主題

本專案主題為「台股市場資訊儀表板」。系統整合台股股價、三大法人買賣超、Yahoo 股市新聞、PTT Stock 討論與 AI 摘要，讓使用者輸入股票代號後，可以快速掌握個股近期市場資訊。

本系統定位為資訊整理工具，不提供投資建議，也不預測股價。

## 2. 動機

台灣股票市場資訊來源分散，使用者常需要同時查看股價、籌碼、新聞與社群討論，才能理解近期市場變化。然而這些資訊通常分散在不同網站，格式也包含表格、新聞長文與社群討論，整理成本較高。

因此本專案希望建立一個簡單的儀表板，把分散資料整合到同一畫面，並透過 AI 或 fallback 規則式分析產生白話摘要，協助使用者更快理解近期資訊重點。

## 3. 系統功能

主要功能如下：

| 功能 | 說明 |
| --- | --- |
| 個股查詢 | 使用者輸入股票代號，例如 2330 |
| 股價走勢 | 顯示 Yahoo Finance 股價折線圖 |
| 三大法人買賣超 | 顯示外資、投信、自營商買賣超 |
| 新聞與 PTT 情緒 | 整理 Yahoo 新聞與 PTT Stock 討論情緒 |
| AI 近期觀察摘要 | 有 API key 時使用模型分析；無 API key 時使用 fallback 分析 |
| 快取優先 | 預設使用 `data/raw_data/` 內既有資料，避免展示時因網路不穩中斷 |
| 示範資料模式 | 若快取不足，自動載入 `data/demo/2330/`，確保老師能看到完整畫面 |
| 資料更新控制台 | 顯示目前資料來源、最後更新時間與各類資料筆數 |
| 籌碼白話解讀 | 將近五筆三大法人買賣超整理為資金方向與連買/連賣狀態 |
| 重新抓取資料 | 使用者手動按下按鈕才會重新爬蟲 |
| 資料品質摘要 | 顯示股價、法人、新聞、PTT、RAG 文件與錯誤數 |

## 4. 資料來源

| 資料類型 | 來源 |
| --- | --- |
| 股價資料 | Yahoo Finance chart API |
| 三大法人資料 | TWSE；若官方端點不穩，使用 FinMind 公開資料集 fallback |
| 新聞資料 | Yahoo 股市新聞 |
| 社群資料 | PTT Stock 版 |
| AI 摘要 | Google Gemini 或 OpenAI；若沒有 API key 則使用規則式 fallback |

## 5. 資料處理流程

系統流程如下：

```text
輸入股票代號
-> 檢查本機快取資料
-> 若 RAG 文件足夠，直接載入快取
-> 若快取不足，改用 data/demo/2330/ 示範資料
-> 若快取與示範資料皆不足，提示使用者手動重新抓取
-> 爬蟲取得股價、法人、新聞、PTT
-> 轉成 CSV / JSON / JSONL
-> 建立可檢索文件
-> 產生圖表、摘要與資料品質資訊
```

輸出資料主要包含：

```text
price_history.csv
institutional_trading.csv
news.json
ptt_posts.json
rag_documents.jsonl
crawl_summary.json
crawl_errors.json
```

正式繳交版不包含完整 `data/raw_data/`，只保留 `data/demo/2330/` 的少量示範資料。

## 6. RAG / AI 摘要流程

新聞與 PTT 文章會被整理成 `rag_documents.jsonl`。系統會根據使用者問題，從已蒐集資料中找出相關文件，再提供給分析模組。

本期末版本採用本機 TF-IDF 檢索作為輕量化 RAG。此方法不需要額外向量資料庫或 embedding API，較適合課堂專題展示與離線測試。

AI 摘要流程：

```text
使用者問題
-> 檢索相關新聞/PTT 文件
-> 組合股價、法人、情緒與來源資料
-> 若有 API key，使用 Gemini/OpenAI 產生摘要
-> 若無 API key 或 API 失敗，改用 fallback 規則式分析
```

這樣可以避免系統在沒有 API key 時直接中斷。

## 7. API key 使用方式

本專案沒有在程式碼中寫死任何 API key。

方法一：複製 `.env.example`，改名為 `.env`，並填入 API key：

```text
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Optional
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

方法二：執行 Streamlit 後，在左側欄位「API 狀態」區塊輸入 Google AI Studio API key。此方式只會在本次執行期間暫時使用，不會寫入檔案。

若沒有 API key，系統仍會使用 fallback 規則式分析。

## 8. 操作步驟

安裝套件：

```powershell
pip install -r requirements.txt
```

啟動儀表板：

```powershell
streamlit run app.py
```

建議展示案例：

```text
股票代號：2330
股票名稱：台積電
操作：輸入 2330 -> 按「查詢」 -> 使用快取資料展示
```

若需要更新資料，可按左側「重新抓取資料」。

API key 健康檢查：

```powershell
python -m tools.api_health_check --provider gemini
```

離線測試：

```powershell
python -m tools.offline_smoke_test
```

## 9. 成果展示

以 2330 台積電為例，系統可展示：

1. 資料更新控制台：顯示目前使用快取、示範資料或即時爬取，以及各類資料筆數。
2. 個股概覽：最新價、區間趨勢、法人近五筆、新聞/PTT 筆數。
3. 股價走勢：以折線圖呈現近期收盤價，並加入股價與三大法人合計買賣超對照圖。
4. 三大法人買賣超：以長條圖呈現外資、投信、自營商資料，並提供籌碼解讀卡。
5. 新聞與 PTT 情緒：統計正向、中立、負向筆數。
6. AI 近期觀察摘要：整理股價、籌碼、新聞與社群資訊。
7. 資料來源、資料品質摘要與金融名詞小字典：列出資料筆數、限制與基本名詞。

目前 `data/demo/2330/` 保留少量示範資料，方便老師確認資料格式。

## 10. 限制與未來改進

目前限制：

1. PTT 資料可能因關鍵字、時間範圍或網站限制而不足。
2. Yahoo/Google 新聞網頁結構可能改變，爬蟲需要維護。
3. 情緒分析主要以關鍵字與簡單規則判斷，無法完全理解反諷或複雜語氣。
4. 本期末版本使用 TF-IDF RAG，語意理解能力不如正式 embedding + vector database。
5. 系統只整理市場資訊，不提供買賣建議。

未來改進：

1. 串接公開資訊觀測站、月營收與正式財報資料。
2. 將 TF-IDF RAG 升級為 Embedding + FAISS 或 ChromaDB。
3. 加入更完整的快取管理與資料更新時間標示。
4. 增加更多視覺化，例如法人趨勢、新聞來源分布、情緒比例圖。
5. 改善新聞與 PTT 的資料穩定性。

## 11. 繳交檔案說明

期末繳交版保留：

```text
app.py
requirements.txt
README.md
.env.example
.gitignore
ai_core/
scrapers/
tools/
docs/REPORT.md
data/demo/2330/
```

不放入繳交版：

```text
.env
.venv/
.venv312/
.deps/
__pycache__/
*.pyc
*.log
data/raw_data/
```

產生乾淨繳交壓縮檔：

```powershell
python -m tools.export_submission --output Taiwan_Stock_System_submission.zip
```

打包工具會自動排除 API key、虛擬環境、快取與完整爬蟲資料，避免上傳敏感或過大的檔案。
