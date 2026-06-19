# 台股市場資訊解讀系統

> **API Key 設定提醒**
> 本專案 ZIP 已包含 API Key 為空白的 `.env`。
> 老師可直接在專案根目錄的 `.env` 填入 Google Gemini 或 OpenAI API Key，也可以啟動系統後，在左側「API 狀態」欄位輸入。
> 若未提供 API Key，系統仍可顯示資料與 RAG 證據，AI 回答會自動改用規則式 fallback。

## 專案簡介

本系統使用 Streamlit 建立台股市場資訊整合介面，可查詢股價、成交量、三大法人、財經新聞與 PTT 討論，並結合 RAG、Memory、Query Rewriting 與 AI 產生具來源引用的市場觀察摘要。

本系統僅提供市場資訊整理與理解輔助，不構成投資建議，也不預測確切未來股價。

## 主要功能

* 股價走勢與成交量
* 外資、投信、自營商與法人合計
* 財經新聞與 PTT 討論
* TF-IDF RAG、Chunking 與 Top-k 檢索
* 金融名詞與分析規則知識庫
* Memory 與 Query Rewriting
* `[R1]`、`[R2]` 等證據引用
* Gemini／OpenAI 模型支援
* 無 API Key 時自動使用 fallback
* 快取與 `data/demo/2330/` 離線展示

股價與法人屬於結構化資料，由 Pandas 直接計算；RAG 主要負責新聞、PTT、金融名詞與分析規則的檢索。

## 執行方式

建議使用 Python 3.11 或 Python 3.12。

安裝套件：

```powershell
pip install -r requirements.txt
```

啟動系統：

```powershell
streamlit run app.py
```

開啟後可使用以下展示案例：

```text
股票代號：2330
股票名稱：台積電
問題：台積電最近的市場狀況如何？
追問：那外資呢？
```

## API Key 設定

專案根目錄的 `.env` 已包含以下空白欄位：

```env
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

請在等號後填入自己的 API Key。

也可以在 Streamlit 左側「API 狀態」欄位輸入。介面輸入的 Key 只會在本次執行期間使用，不會寫入檔案。

未設定 API Key 時，系統會自動使用規則式 fallback，不會中斷。

## 資料模式

* **即時資料**：按下「重新抓取資料」後取得。
* **快取資料**：使用先前已成功取得的本機資料。
* **Demo 資料**：即時與快取資料不足時，使用 `data/demo/2330/` 的台積電示範資料。

系統會在畫面上顯示目前資料模式，避免將 Demo 資料誤認為即時資料。

## 主要資料來源

* 股價：Yahoo Finance chart API
* 三大法人：TWSE，失敗時使用 FinMind fallback
* 新聞：Yahoo 股市新聞
* 社群：PTT Stock
* 金融知識：`data/knowledge/finance_glossary.json`
* 分析規則：`data/knowledge/analysis_rules.json`

## 測試方式

語法檢查：

```powershell
$files = @('app.py') + (Get-ChildItem ai_core,scrapers,tools -Filter *.py -Recurse | ForEach-Object { $_.FullName })
python -m py_compile @files
```

離線測試：

```powershell
python -m tools.offline_smoke_test
```

Gemini API 測試：

```powershell
python -m tools.api_health_check --provider gemini
```

## 專案結構

```text
Taiwan_Stock_System/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .env
├─ .env.example
├─ ai_core/
├─ scrapers/
├─ tools/
├─ data/
│  ├─ demo/2330/
│  └─ knowledge/
└─ docs/
```

## 使用限制

外部網站可能因改版、網路或反爬蟲機制暫時無法取得資料。新聞與 PTT 僅代表目前實際取得的內容，不能代表整體市場意見。

RAG 可以降低模型幻覺，但不能保證回答完全正確，使用者仍應查閱原始資料並自行判斷。
