<pre>
  
   /$$$$$$                                      /$$            /$$$$$$            /$$            /$$$$$$ 
 /$$__  $$                                    | $$           /$$__  $$          |__/           /$$__  $$
| $$  \__/ /$$  /$$  /$$  /$$$$$$   /$$$$$$  /$$$$$$        | $$  \__/  /$$$$$$  /$$  /$$$$$$ | $$  \__/
|  $$$$$$ | $$ | $$ | $$ /$$__  $$ /$$__  $$|_  $$_/        | $$ /$$$$ /$$__  $$| $$ /$$__  $$| $$$$    
 \____  $$| $$ | $$ | $$| $$$$$$$$| $$$$$$$$  | $$          | $$|_  $$| $$  \__/| $$| $$$$$$$$| $$_/    
 /$$  \ $$| $$ | $$ | $$| $$_____/| $$_____/  | $$ /$$      | $$  \ $$| $$      | $$| $$_____/| $$      
|  $$$$$$/|  $$$$$/$$$$/|  $$$$$$$|  $$$$$$$  |  $$$$/      |  $$$$$$/| $$      | $$|  $$$$$$$| $$      
 \______/  \_____/\___/  \_______/ \_______/   \___/         \______/ |__/      |__/ \_______/|__/      
                                                                                                        

</pre>
# Recall Trading Agent (Sandbox + Dashboard)

Agen trading untuk kompetisi **Recall** dengan 2 cara pakai:

* **Dashboard Web** (FastAPI + HTML/JS): lihat saldo, harga, **Rebalance**, dan **Manual Trade** via browser.
* **CLI Agent** (`trading_agent.py`): jalankan strategi **rebalance** dari terminal (tanpa UI).

Repo ini default-nya ke **Sandbox** agar aman untuk testing. Production cukup switch `.env`. ([GitHub][1])

---

## Fitur

* Liat **balances** & nilai portofolio (USD).
* **Rebalance**: jaga porsi token target (mis. WETH/ETH) ke **Target %** dari total.
* **Manual trade**: buy/sell sederhana dengan nominal USD “human units”.
* Token registry lokal via `tokens.json`.
* (Opsional) Integrasi AI (file `llm.py`) jika memasang `OPENAI_API_KEY`.

---

## Struktur Proyek

```
recall-agent/
├─ app.py                 # server FastAPI (dashboard + proxy ke Recall API)
├─ templates/
│  └─ index.html          # halaman utama (Bootstrap)
├─ static/
│  ├─ app.js              # logic fetch API (balances, rebalance, trade)
│  └─ style.css           # gaya tambahan (opsional)
├─ trading_agent.py       # CLI agent: strategi rebalance dari terminal
├─ tokens.json            # registry token lokal
├─ requirements.txt       # dependencies
├─ .env.sandbox           # contoh env SANDBOX
├─ .env.prod              # contoh env PRODUCTION
└─ README.md
```

(Disesuaikan dengan isi repo kamu saat ini.) ([GitHub][1])

---

## Prasyarat

* **Python 3.11+** (disarankan).
* Git (opsional, untuk kerja via GitHub).

---

## Instalasi

```bash
# 1) Buat & aktifkan virtualenv
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

## Konfigurasi Lingkungan

Buat file `.env` dari template **sandbox** (testing):

```bash
# Windows PowerShell
Copy-Item .env.sandbox .env -Force
# macOS/Linux
cp .env.sandbox .env
```

Isi `.env` kamu minimal seperti ini:

```
RECALL_API_URL=https://api.sandbox.competitions.recall.network
RECALL_API_KEY=pk_sandbox_xxxxxxxxxxxxxxxxx
# Opsional untuk AI
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
# OPENAI_MODEL=gpt-4o-mini
```

> Saat kompetisi live, ganti ke production:
>
> ```
> RECALL_API_URL=https://api.competitions.recall.network
> RECALL_API_KEY=pk_prod_xxxxxxxxxxxxxxxxx
> ```

---

## Menjalankan Dashboard Web

```bash
uvicorn app:app --reload --port 8000
```

Buka browser: `http://localhost:8000`

### Yang tersedia di dashboard

* **Balances**: tabel saldo & nilai (USD).
* **Rebalance**:

  * **Target %**: porsi token target dari total.
  * **Max trade (USD)**: batas nominal per eksekusi.
  * **USDC reserve (USD)**: kas minimum yang disisakan.
  * Tombol **Rebalance** akan beli token target (dengan cash token, biasanya USDC/ETH) hingga mendekati target.
* **Manual Trade**:

  * **Side**: buy (USDC→WETH) / sell (WETH→USDC).
  * **Amount (USD)** + **Reason**.
  * Tombol **Execute**.

> Dashboard mem-proxy panggilan ke Recall API melalui backend FastAPI—**API key tidak pernah disimpan di frontend**. ([GitHub][1])

---

## Menjalankan CLI Agent (Rebalance)

Contoh paling sederhana (target 10% WETH/ETH):

```bash
python trading_agent.py rebalance
```

Parameter umum (opsional):

```bash
python trading_agent.py rebalance ^
  --target-pct 12 ^
  --max-trade-usd 300 ^
  --reserve-usd 50 ^
  --target-token WETH ^
  --cash-token USDC
```

Mode AI (opsional, butuh `OPENAI_API_KEY`):

```bash
python trading_agent.py rebalance --use-llm
```

> CLI ini memakai endpoint Recall yang sama seperti di dashboard (balances/price/trade), dengan default ke **Sandbox**. ([GitHub][1])

---

## Rebalance—cara kerja singkat

1. Hitung **total nilai** portofolio (USD).
2. Target value = **Target % × Total**.
3. Ambil nilai token target saat ini.
4. Delta = Target − Saat ini.

   * **Delta > 0** → beli token target dari **cash token** (mis. USDC/ETH) hingga `min(delta, max trade, cash tersedia − reserve)`.
   * **Delta ≤ 0** → tidak beli (versi dasar tidak menjual; opsi “allow sell” bisa ditambah bila diperlukan).
