<<<<<<< HEAD
import os
import asyncio
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Request, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, timezone
from datetime import datetime, timezone

# === Load env ===
load_dotenv()
API_KEY = os.getenv("RECALL_API_KEY")
BASE    = os.getenv("RECALL_API_URL", "https://api.sandbox.competitions.recall.network")
if not API_KEY:
    raise SystemExit("RECALL_API_KEY belum ada di .env")

USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_ETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"


def hdrs():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _default_specific_for_chain(chain: str | None):
    if not chain:
        return None
    c = str(chain).lower()
    if c == "evm":
        return "eth"
    if c == "solana":
        return "sol"
    if c == "svm":
        return "svm"
    return None


def _execute_trade_payload(payload: dict):
    upstream_text = None
    upstream_status = None

    def _execute(p: dict):
        return requests.post(f"{BASE}/api/trade/execute", json=p, headers=hdrs(), timeout=30)

    r = _execute(payload)
    upstream_text = r.text
    upstream_status = r.status_code
    if r.status_code in (400, 404) and (str(payload.get("fromChain")) == "svm" or str(payload.get("toChain")) == "svm"):
        fallback = dict(payload)
        changed = False
        if str(fallback.get("fromChain")) == "svm":
            fallback["fromChain"] = "solana"
            if not fallback.get("fromSpecificChain"):
                fallback["fromSpecificChain"] = "svm"
            changed = True
        if str(fallback.get("toChain")) == "svm":
            fallback["toChain"] = "solana"
            if not fallback.get("toSpecificChain"):
                fallback["toSpecificChain"] = "svm"
            changed = True
        if changed:
            r2 = _execute(fallback)
            upstream_text = r2.text
            upstream_status = r2.status_code
            r2.raise_for_status()
            return r2.json(), upstream_status, upstream_text
        r.raise_for_status()
        return r.json(), upstream_status, upstream_text
    r.raise_for_status()
    return r.json(), upstream_status, upstream_text




def _get_price(token: str, chain: str = "evm", specific_chain: str | None = "eth") -> float:
    params = {"token": token, "chain": chain}
    if specific_chain:
        params["specificChain"] = specific_chain
    r = requests.get(
        f"{BASE}/api/price",
        params=params,
        headers=hdrs(), timeout=20
    )
    r.raise_for_status()
    data = r.json()
    return float(data.get("price", 0))


# === FastAPI ===
app = FastAPI(title="Recall Agent Dashboard")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str((BASE_DIR / "static").resolve())), name="static")
templates = Jinja2Templates(directory=str((BASE_DIR / "templates").resolve()))

# Optional LLM module (robust local import)
llm_mod = None
try:
    # Try regular import if running inside the folder
    import llm as _llm
    llm_mod = _llm
except Exception:
    try:
        import importlib.util, pathlib, sys
        here = pathlib.Path(__file__).resolve().parent
        llm_path = here / "llm.py"
        if llm_path.exists():
            spec = importlib.util.spec_from_file_location("llm", str(llm_path))
            _mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(_mod)  # type: ignore
            llm_mod = _mod
    except Exception:
        llm_mod = None


# ---------- Pages ----------
@app.get("/")
def home(request: Request):
    # Info dasar tidak mengandung secret (API key hanya di server)
    ctx = {"request": request, "base_url": BASE, "hide_small_default": float(os.getenv("BALANCE_HIDE_USD_THRESHOLD", "1"))}
    return templates.TemplateResponse("index.html", ctx)


# ---------- API Proxies ----------
@app.get("/api/balances")
def api_balances(minUsd: float | None = Query(None)):
    try:
        upstream_text = None
        upstream_status = None
        r = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20)
        upstream_text = r.text
        upstream_status = r.status_code
        r.raise_for_status()
        resp = r.json()
        try:
            if minUsd is None:
                threshold = float(os.getenv("BALANCE_HIDE_USD_THRESHOLD", "1"))
            else:
                threshold = max(0.0, float(minUsd))
            items = []
            for it in resp.get("balances", []):
                val = float(it.get("value", 0) or 0)
                token_addr = str(it.get("tokenAddress", ""))
                kind = str(it.get("kind", "")).lower()
                is_token = bool(token_addr) or kind == "token"
                if is_token and val < threshold:
                    continue
                items.append(it)
            resp["balances"] = items
            resp["minUsd"] = threshold
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "upstream_status": upstream_status,
            "upstream_body": upstream_text,
        }, status_code=500)


@app.get("/api/price")
def api_price(token: str = Query(...), chain: str = "evm", specificChain: str = "eth"):
    try:
        upstream_text = None
        upstream_status = None
        r = requests.get(
            f"{BASE}/api/price",
            params={"token": token, "chain": chain, "specificChain": specificChain},
            headers=hdrs(), timeout=20,
        )
        upstream_text = r.text
        upstream_status = r.status_code
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "upstream_status": upstream_status,
            "upstream_body": upstream_text,
        }, status_code=500)


@app.get("/api/ai/suggest-rebalance")
def api_ai_suggest_rebalance():
    if not llm_mod:
        return JSONResponse({"error": "LLM module not available. Install 'openai' and ensure llm.py present."}, status_code=500)
    try:
        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        balances = rb.get("balances", [])
        data = llm_mod.suggest_rebalance_with_llm(balances, default_target=WETH_ETH, default_cash=USDC_ETH)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ai/status")
def api_ai_status(test: int = 0):
    """Return whether LLM is available and API key present. If test=1, try init client."""
    info = {"available": False, "hasKey": False, "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
    if not llm_mod:
        return JSONResponse(info)
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    info["hasKey"] = has_key
    if not has_key:
        return JSONResponse(info)
    if test:
        try:
            # Only try to construct client; avoid actual network call for speed
            _ = llm_mod._get_openai_client()
            info["available"] = True
        except Exception as e:
            return JSONResponse({**info, "error": str(e)})
        return JSONResponse(info)
    # Without test flag, report optimistic availability if key exists and module present
    info["available"] = True
    return JSONResponse(info)

# ---------- Token Registry (server-side) ----------
import json
from pathlib import Path

TOKENS_FILE = Path(__file__).resolve().parent / "tokens.json"

DEFAULT_TOKENS = [
    {"symbol": "USDC", "address": USDC_ETH, "chain": "evm", "specificChain": "eth"},
    {"symbol": "WETH", "address": WETH_ETH, "chain": "evm", "specificChain": "eth"},
    {"symbol": "USDT", "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "chain": "evm", "specificChain": "eth"},
    {"symbol": "DAI",  "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "chain": "evm", "specificChain": "eth"},
    {"symbol": "UNI",  "address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "chain": "evm", "specificChain": "eth"},
    # EVM: Polygon
    {"symbol": "USDC", "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "WETH", "address": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "USDT", "address": "0xC2132D05D31c914a87C6611C10748AEB04B58e8F", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "DAI",  "address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063", "chain": "evm", "specificChain": "polygon"},
    # EVM: Base (partial known set)
    {"symbol": "WETH", "address": "0x4200000000000000000000000000000000000006", "chain": "evm", "specificChain": "base"},
    # Solana (SPL mints)
    {"symbol": "wSOL", "address": "So11111111111111111111111111111111111111112", "chain": "solana", "specificChain": "sol"},
    {"symbol": "USDC", "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "chain": "solana", "specificChain": "sol"},
]

def _load_tokens():
    if TOKENS_FILE.exists():
        try:
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Merge defaults that are missing by (chain, specific, address)
                    base = list(data)
                    seen = set((str(t.get("chain","")), str(t.get("specificChain","")), str(t.get("address","")) .lower()) for t in base)
                    for d in DEFAULT_TOKENS:
                        key = (str(d.get("chain","")), str(d.get("specificChain","")), str(d.get("address","")) .lower())
                        if key not in seen:
                            base.append(d)
                            seen.add(key)
                    return base
        except Exception:
            pass
    return DEFAULT_TOKENS.copy()

def _save_tokens(tokens):
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

def _ensure_token_known(address: str, symbol_hint: str | None = None, chain: str = "evm", specific: str = "eth"):
    try:
        addr = str(address or "").strip()
        if not addr.lower().startswith("0x") or len(addr) != 42:
            return
        toks = _load_tokens()
        key = (chain, specific, addr.lower())
        for t in toks:
            if t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == addr.lower():
                return  # already known
        # Add with placeholder symbol if hint missing
        sym = (symbol_hint or ("TKN-" + addr[-4:])).upper()
        toks.append({"symbol": sym, "address": addr, "chain": chain, "specificChain": specific})
        _save_tokens(toks)
    except Exception:
        pass


# ---------- Trade History (server-side) ----------
TRADES_FILE = Path(__file__).resolve().parent / "trades.jsonl"

def _append_trade(entry: dict):
    try:
        entry = dict(entry)
        entry.setdefault("serverTime", datetime.now(timezone.utc).isoformat())
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort only
        pass

def _read_trades(limit: int = 100):
    items = []
    try:
        if not TRADES_FILE.exists():
            return []
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return list(reversed(items))[: max(0, int(limit or 0)) or 100]
    except Exception:
        return []

@app.get("/api/trades")
def api_trades(limit: int = 50):
    try:
        return JSONResponse({"trades": _read_trades(limit=limit)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/clear")
def api_trades_clear():
    try:
        if TRADES_FILE.exists():
            TRADES_FILE.unlink(missing_ok=True)  # type: ignore
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- PnL Ledger (server-side) ----------
LEDGER_FILE = Path(__file__).resolve().parent / "pnl_ledger.json"

def _load_ledger():
    base = {"positions": {}, "realized": [], "stats": {"realizedUsd": 0.0}}
    try:
        if LEDGER_FILE.exists():
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {**base, **data}
    except Exception:
        pass
    return base

def _save_ledger(ledger):
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

def _pos_key(addr: str, chain: str, specific: str) -> str:
    return f"{(chain or 'evm').lower()}:{(specific or 'eth').lower()}:{addr.lower()}"

def _symbol_of(addr: str) -> str:
    try:
        for t in _load_tokens():
            if str(t.get("address","")) .lower() == str(addr).lower():
                return str(t.get("symbol",""))
    except Exception:
        pass
    return ""

def _ledger_buy(addr: str, chain: str, specific: str, amount_to: float, cost_usd: float, symbol: str | None = None):
    if amount_to <= 0 or cost_usd <= 0:
        return
    ledger = _load_ledger()
    key = _pos_key(addr, chain, specific)
    pos = ledger["positions"].get(key) or {"symbol": symbol or _symbol_of(addr) or addr, "amount": 0.0, "investedUsd": 0.0}
    pos["amount"] = float(pos.get("amount", 0.0)) + float(amount_to)
    pos["investedUsd"] = float(pos.get("investedUsd", 0.0)) + float(cost_usd)
    ledger["positions"][key] = pos
    _save_ledger(ledger)

def _ledger_sell(addr: str, chain: str, specific: str, amount_from: float, proceeds_usd: float, symbol: str | None = None):
    if amount_from <= 0 or proceeds_usd < 0:
        return
    ledger = _load_ledger()
    key = _pos_key(addr, chain, specific)
    pos = ledger["positions"].get(key) or {"symbol": symbol or _symbol_of(addr) or addr, "amount": 0.0, "investedUsd": 0.0}
    cur_amt = float(pos.get("amount", 0.0))
    cur_inv = float(pos.get("investedUsd", 0.0))
    sell_amt = min(cur_amt, float(amount_from))
    avg_cost_per_unit = (cur_inv / cur_amt) if cur_amt > 0 else 0.0
    cost_usd = sell_amt * avg_cost_per_unit
    pnl = float(proceeds_usd) - cost_usd
    # update position
    pos["amount"] = cur_amt - sell_amt
    pos["investedUsd"] = max(0.0, cur_inv - cost_usd)
    ledger["positions"][key] = pos
    # realized log
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "token": symbol or pos.get("symbol") or addr,
        "address": addr,
        "chain": chain,
        "specificChain": specific,
        "side": "sell",
        "amount": sell_amt,
        "proceedsUsd": float(proceeds_usd),
        "costUsd": cost_usd,
        "pnlUsd": pnl,
    }
    ledger["realized"].append(event)
    ledger["stats"]["realizedUsd"] = float(ledger["stats"].get("realizedUsd", 0.0)) + pnl
    _save_ledger(ledger)

@app.get("/api/pnl")
def api_pnl():
    try:
        ledger = _load_ledger()
        positions = ledger.get("positions", {})
        enriched = {}
        total_mkt = 0.0
        total_inv = 0.0
        total_unrl = 0.0
        for key, pos in positions.items():
            try:
                _, specific, addr = key.split(":", 2)
            except Exception:
                specific, addr = "eth", key
            amt = float(pos.get("amount", 0.0))
            inv = float(pos.get("investedUsd", 0.0))
            price = 0.0
            if amt > 0:
                try:
                    price = _get_price(addr, chain="evm", specific_chain=specific)
                except Exception:
                    price = 0.0
            mkt = amt * (price or 0.0)
            unrl = mkt - inv
            total_mkt += mkt
            total_inv += inv
            total_unrl += unrl
            enriched[key] = {
                **pos,
                "avgCostPerUnitUsd": (inv / amt) if amt > 0 else 0.0,
                "marketPriceUsd": price,
                "marketValueUsd": mkt,
                "unrealizedUsd": unrl,
            }
        return JSONResponse({
            "positions": enriched,
            "realized": ledger.get("realized", []),
            "stats": {
                "realizedUsd": float(ledger.get("stats", {}).get("realizedUsd", 0.0)),
                "unrealizedUsd": total_unrl,
                "marketValueUsd": total_mkt,
                "investedUsd": total_inv,
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
@app.get("/api/tokens")
def api_tokens_list():
    try:
        # Deduplicate by address within same chain/specific, preserving first (canonical) occurrence
        raw = _load_tokens()
        seen = set()
        uniq = []
        for t in raw:
            key = (t.get("chain","evm"), t.get("specificChain","eth"), str(t.get("address","")) .lower())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        return JSONResponse({"tokens": uniq})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tokens")
def api_tokens_add(body: dict = Body(...)):
    try:
        symbol = str(body.get("symbol", "")).strip().upper()
        address = str(body.get("address", "")).strip()
        chain = body.get("chain", "evm")
        # default specific by chain; svm has none
        specific = body.get("specificChain")
        if not specific:
            if str(chain).lower() == "evm":
                specific = "eth"
            elif str(chain).lower() == "solana":
                specific = "sol"
            else:
                specific = ""
        if not symbol or not address:
            return JSONResponse({"error": "symbol dan address wajib diisi"}, status_code=400)
        # Address validation by chain
        if str(chain).lower() == "evm":
            if not address.lower().startswith("0x") or len(address) != 42:
                return JSONResponse({"error": "address tidak valid (EVM harus 0x.. 42 chars)"}, status_code=400)
        elif str(chain).lower() in {"solana", "svm"}:
            base58chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
            if any(c not in base58chars for c in address) or address.lower().startswith("0x") or len(address) < 32 or len(address) > 60:
                return JSONResponse({"error": "address tidak valid (base58 untuk Solana/SVM)"}, status_code=400)
        tokens = _load_tokens()
        # Reject duplicates by address (same chain/specific) to avoid wrong symbol mapping
        for t in tokens:
            if t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == address.lower():
                if t.get("symbol","" ).upper() != symbol:
                    return JSONResponse({"error": f"address sudah terdaftar untuk simbol {t.get('symbol')}"}, status_code=400)
                # if same symbol/address, allow idempotent update
                break
        # update if exists by symbol (same chain/specific), else append
        updated = False
        for t in tokens:
            if t.get("symbol", "").upper() == symbol and t.get("specificChain") == specific and t.get("chain") == chain:
                t["address"] = address
                updated = True
                break
        if not updated:
            tokens.append({"symbol": symbol, "address": address, "chain": chain, "specificChain": specific})
        _save_tokens(tokens)
        return JSONResponse({"ok": True, "tokens": tokens})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tokens/delete")
def api_tokens_delete(body: dict = Body(...)):
    try:
        address = str(body.get("address", "")).strip()
        symbol = str(body.get("symbol", "")).strip().upper()
        chain = body.get("chain", "evm")
        specific = body.get("specificChain")
        if not specific:
            if str(chain).lower() == "evm":
                specific = "eth"
            elif str(chain).lower() == "solana":
                specific = "sol"
            else:
                specific = ""
        tokens = _load_tokens()
        if address:
            addr_low = address.lower()
            tokens = [
                t for t in tokens
                if not (t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == addr_low)
            ]
        elif symbol:
            tokens = [
                t for t in tokens
                if not (t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("symbol","")) .upper() == symbol)
            ]
        else:
            return JSONResponse({"error": "wajib mengirim address atau symbol"}, status_code=400)
        _save_tokens(tokens)
        return JSONResponse({"ok": True, "tokens": tokens})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/rebalance")
def api_rebalance(body: dict = Body(...)):
    """
    body: {
      targetPct: float(0..100), maxTradeUsd: float, reserveUsd|usdcReserveUsd: float,
      targetToken?: str, cashToken?: str, chain?: "evm", specificChain?: "eth"
    }
    Rebalance target token ke target % memakai cash token.
    """
    try:
        target_pct     = float(body.get("targetPct", 10.0))/100.0
        max_trade_usd  = float(body.get("maxTradeUsd", 500.0))
        reserve_usd    = float(body.get("usdcReserveUsd", body.get("reserveUsd", 50.0)))
        chain          = body.get("chain", "evm")
        specific_chain = body.get("specificChain", "eth")
        # Default tokens only for EVM/ETH; otherwise require explicit tokens
        if str(chain).lower() == "evm" and str(specific_chain).lower() == "eth":
            target_token   = str(body.get("targetToken", WETH_ETH))
            cash_token     = str(body.get("cashToken", USDC_ETH))
        else:
            if not body.get("targetToken") or not body.get("cashToken"):
                return JSONResponse({"error": "targetToken dan cashToken wajib diisi untuk chain selain evm/eth"}, status_code=400)
            target_token = str(body.get("targetToken"))
            cash_token = str(body.get("cashToken"))

        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        b  = rb.get("balances", [])
        total = sum(x.get("value",0.0) for x in b)
        target_val = 0.0
        target_price = None
        target_amount = 0.0
        cash_amt = 0.0
        cash_price = 1.0
        for x in b:
            if x.get("chain") == chain and x.get("specificChain") == specific_chain:
                if x["tokenAddress"].lower()==target_token.lower():
                    target_val = x.get("value",0.0)
                    target_amount = x.get("amount",0.0)
                    if x.get("price") is not None:
                        target_price = x.get("price")
                if x["tokenAddress"].lower()==cash_token.lower():
                    cash_amt   = x.get("amount",0.0)
                    if x.get("price") is not None:
                        cash_price = x.get("price")

        target_val_goal = total * target_pct
        delta = target_val_goal - target_val
        allow_sell = bool(body.get("allowSell", False))
        if delta <= 0.0:
            if not allow_sell:
                return JSONResponse({"message":"Sudah >= target, tidak perlu beli.", "deltaUsd": delta})
            # Sell-down mode: kurangi ke target dengan menjual target token ke cash token
            overweight_usd = min(abs(delta), max_trade_usd, target_val)
            if overweight_usd <= 0.0:
                return JSONResponse({"message":"Tidak ada nilai untuk dijual.", "deltaUsd": delta})
            # Pastikan harga target tersedia untuk konversi unit
            if target_price in (None, 0):
                try:
                    target_price = _get_price(target_token, chain, specific_chain)
                except Exception:
                    target_price = None
            if not target_price:
                return JSONResponse({"error":"Harga target token tidak tersedia untuk jual"}, status_code=500)
            amount_target = overweight_usd / float(target_price)
            payload = {
                "fromToken": target_token, "toToken": cash_token,
                "amount": str(round(amount_target, 8)),
                "reason": f"rebalance sell-down from {int(target_pct*100)}% target",
                "slippageTolerance": "0.5",
                "fromChain": chain, "fromSpecificChain": specific_chain,
                "toChain":   chain, "toSpecificChain":   specific_chain,
            }
            r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
            r.raise_for_status()
            result = {
                "action": "sell",
                "targetPct": target_pct,
                "deltaUsd": delta,
                "tradeUsd": overweight_usd,
                "amountFrom": amount_target,
                "fromToken": target_token,
                "toToken": cash_token,
                "tx": r.json(),
            }
            try:
                _append_trade({
                    "type": "rebalance",
                    "action": "sell",
                    "fromToken": target_token,
                    "toToken": cash_token,
                    "amountFrom": amount_target,
                    "tradeUsd": overweight_usd,
                    "payload": payload,
                    "response": result.get("tx"),
                    "status": "ok",
                })
                # PnL: realized on sell
                _ledger_sell(target_token, chain, specific_chain, amount_from=float(amount_target), proceeds_usd=float(overweight_usd), symbol=None)
            except Exception:
                pass
            return JSONResponse(result)

        available = max(0.0, cash_amt*cash_price - reserve_usd)
        trade_usd = min(delta, max_trade_usd, available)
        if trade_usd <= 0.0:
            return JSONResponse({"message":"Cash token tidak cukup setelah cadangan.", "deltaUsd": delta})

        # convert USD to from-token units via price
        if cash_price in (None, 0):
            try:
                cash_price = _get_price(cash_token, chain, specific_chain)
            except Exception:
                pass
        amount_from = trade_usd / (cash_price or 1.0)
        payload = {
            "fromToken": cash_token, "toToken": target_token,
            "amount": str(round(amount_from, 8)),
            "reason": f"rebalance to {int(target_pct*100)}% target",
            "slippageTolerance": "0.5",
            "fromChain": chain, "fromSpecificChain": specific_chain,
            "toChain":   chain, "toSpecificChain":   specific_chain,
        }
        r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
        r.raise_for_status()
        result = {
            "action": "buy",
            "targetPct": target_pct,
            "deltaUsd": delta,
            "tradeUsd": trade_usd,
            "amountFrom": amount_from,
            "fromToken": cash_token,
            "toToken": target_token,
            "tx": r.json(),
        }
        try:
            _append_trade({
                "type": "rebalance",
                "fromToken": cash_token,
                "toToken": target_token,
                "amountFrom": amount_from,
                "tradeUsd": trade_usd,
                "payload": payload,
                "response": result.get("tx"),
                "status": "ok",
            })
            # PnL ledger (buy target token approx using price snapshot)
            try:
                price_to = _get_price(target_token, chain, specific_chain)
            except Exception:
                price_to = None
            if price_to:
                amount_to = float(trade_usd) / float(price_to)
                _ledger_buy(target_token, chain, specific_chain, amount_to=amount_to, cost_usd=float(trade_usd), symbol=None)
        except Exception:
            pass
        return JSONResponse(result)
    except Exception as e:
        try:
            _append_trade({
                "type": "rebalance",
                "error": str(e),
                "status": "error",
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/batch-trade")
def api_batch_trade(body: dict = Body(...)):
    try:
        side = str(body.get("side", "buy")).lower()
        if side not in {"buy", "sell"}:
            return JSONResponse({"error": "side harus buy atau sell"}, status_code=400)
        chain = body.get("chain", "evm")
        specific_chain = body.get("specificChain")
        if specific_chain is not None:
            specific_chain = str(specific_chain).strip() or None
        if specific_chain is None:
            specific_chain = _default_specific_for_chain(chain)
        to_chain = body.get("toChain", chain)
        to_specific_chain = body.get("toSpecificChain")
        if to_specific_chain is not None:
            to_specific_chain = str(to_specific_chain).strip() or None
        if to_specific_chain is None:
            to_specific_chain = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        total_usd = float(body.get("totalUsd") or 0)
        chunk_usd = float(body.get("chunkUsd") or 0)
        reason_base = str(body.get("reason", f"batch {side}") or f"batch {side}").strip() or f"batch {side}"
        if not from_token or not to_token:
            return JSONResponse({"error": "fromToken dan toToken wajib"}, status_code=400)
        if total_usd <= 0:
            return JSONResponse({"error": "totalUsd harus > 0"}, status_code=400)
        if chunk_usd <= 0:
            return JSONResponse({"error": "chunkUsd harus > 0"}, status_code=400)
        spent = 0.0
        iteration = 0
        chunks = []

        while spent + 1e-9 < total_usd:
            remaining = total_usd - spent
            usd_chunk = min(chunk_usd, remaining)
            try:
                price_from = _get_price(from_token, chain, specific_chain)
            except Exception as exc:
                return JSONResponse({"error": f"gagal ambil harga: {exc}"}, status_code=500)
            if not price_from:
                return JSONResponse({"error": "Harga fromToken tidak tersedia"}, status_code=500)
            amount_human = float(usd_chunk) / float(price_from)
            iteration += 1
            reason = f"{reason_base} #{iteration}"
            payload = {
                "fromToken": from_token,
                "toToken": to_token,
                "amount": str(round(float(amount_human), 8)),
                "reason": reason,
                "slippageTolerance": "0.5",
                "fromChain": chain,
                "toChain": to_chain,
            }
            if specific_chain:
                payload["fromSpecificChain"] = specific_chain
            if to_specific_chain:
                payload["toSpecificChain"] = to_specific_chain

            upstream_status = None
            upstream_text = None
            try:
                resp, upstream_status, upstream_text = _execute_trade_payload(payload)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                body_txt = getattr(getattr(exc, "response", None), "text", None)
                try:
                    _append_trade({
                        "type": "batch",
                        "side": side,
                        "status": "error",
                        "iteration": iteration,
                        "error": str(exc),
                        "payload": payload,
                        "upstream_status": status if status is not None else upstream_status,
                        "upstream_body": body_txt if body_txt is not None else upstream_text,
                    })
                except Exception:
                    pass
                return JSONResponse({
                    "error": str(exc),
                    "iteration": iteration,
                    "spent": spent,
                    "chunks": chunks,
                    "side": side,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                }, status_code=status if isinstance(status, int) else 500)

            chunk_entry = {
                "iteration": iteration,
                "usd": usd_chunk,
                "amountHuman": amount_human,
                "response": resp,
                "side": side,
            }
            chunks.append(chunk_entry)
            spent += usd_chunk
            try:
                log = {
                    "type": "batch",
                    "side": side,
                    "status": "ok",
                    "iteration": iteration,
                    "usd": usd_chunk,
                    "amountHuman": amount_human,
                    "payload": payload,
                    "response": resp,
                }
                _append_trade(log)
                if side == "buy":
                    _ensure_token_known(to_token, None, to_chain, to_specific_chain or "")
                else:
                    _ensure_token_known(from_token, None, chain, specific_chain or "")
                if side == "buy":
                    try:
                        price_to = _get_price(to_token, to_chain, to_specific_chain)
                    except Exception:
                        price_to = None
                    cost_usd = float(usd_chunk)
                    if price_to:
                        amount_to = cost_usd / float(price_to)
                        _ledger_buy(to_token, to_chain, to_specific_chain or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
                else:
                    _ledger_sell(from_token, chain, specific_chain or "", amount_from=float(amount_human), proceeds_usd=float(usd_chunk), symbol=None)
            except Exception:
                pass

        return JSONResponse({
            "status": "ok",
            "side": side,
            "totalUsd": spent,
            "chunks": chunks,
            "iterations": iteration,
        })
    except Exception as e:
        try:
            _append_trade({
                "type": "batch",
                "side": str(body.get("side")),
                "status": "error",
                "error": str(e),
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/api/manual-trade")
def api_manual_trade(body: dict = Body(...)):
    """
    body: {
      side: "buy"|"sell",
      amountUsd?: float, amountHuman?: float, reason?: str,
      fromToken?: str, toToken?: str, chain?: str, specificChain?: str
    }
    """
    try:
        side = body.get("side", "buy")
        reason = body.get("reason", "manual trade")
        chain = body.get("chain", "evm")
        specific_chain = body.get("specificChain")
        if specific_chain is not None:
            specific_chain = str(specific_chain).strip() or None
        if specific_chain is None:
            specific_chain = _default_specific_for_chain(chain)
        to_chain = body.get("toChain", chain)
        to_specific_chain = body.get("toSpecificChain")
        if to_specific_chain is not None:
            to_specific_chain = str(to_specific_chain).strip() or None
        if to_specific_chain is None:
            to_specific_chain = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        amount_human = body.get("amountHuman")
        amount_usd = body.get("amountUsd")

        if not from_token or not to_token:
            if chain == "evm":
                if side == "buy":
                    from_token, to_token = USDC_ETH, WETH_ETH
                else:
                    from_token, to_token = WETH_ETH, USDC_ETH
            else:
                return JSONResponse({"error": "fromToken/toToken wajib diisi untuk non-EVM"}, status_code=400)

        if amount_human is None:
            amt = float(amount_usd or 0)
            if amt <= 0:
                return JSONResponse({"error": "amountUsd must be > 0 or provide amountHuman"}, status_code=400)
            px = _get_price(from_token, chain, specific_chain)
            amount_human = amt / px if px else amt

        payload = {
            "fromToken": from_token,
            "toToken": to_token,
            "amount": str(round(float(amount_human), 8)),
            "reason": reason,
            "slippageTolerance": "0.5",
            "fromChain": chain,
            "toChain": to_chain,
        }
        if specific_chain:
            payload["fromSpecificChain"] = specific_chain
        if to_specific_chain:
            payload["toSpecificChain"] = to_specific_chain

        upstream_status = None
        upstream_text = None
        try:
            resp, upstream_status, upstream_text = _execute_trade_payload(payload)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            body_txt = getattr(getattr(exc, "response", None), "text", None)
            try:
                _append_trade({
                    "type": "manual",
                    "side": side,
                    "error": str(exc),
                    "status": "error",
                    "body": body,
                    "payload": payload,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                })
            except Exception:
                pass
            return JSONResponse({
                "error": str(exc),
                "upstream_status": status if status is not None else upstream_status,
                "upstream_body": body_txt if body_txt is not None else upstream_text,
            }, status_code=status if isinstance(status, int) else 500)

        try:
            _append_trade({
                "type": "manual",
                "side": side,
                "fromToken": from_token,
                "toToken": to_token,
                "amountHuman": float(amount_human),
                "amountUsd": float(amount_usd) if amount_usd is not None else None,
                "reason": reason,
                "payload": payload,
                "response": resp,
                "status": "ok",
            })
            if side == "buy":
                _ensure_token_known(to_token, None, to_chain, to_specific_chain or "")
            else:
                _ensure_token_known(from_token, None, chain, specific_chain or "")
            try:
                price_from = _get_price(from_token, chain, specific_chain)
            except Exception:
                price_from = None
            try:
                price_to = _get_price(to_token, to_chain, to_specific_chain)
            except Exception:
                price_to = None
            if side == "buy":
                cost_usd = float(amount_usd) if amount_usd is not None else (float(amount_human) * float(price_from or 0))
                if cost_usd and price_to:
                    amount_to = cost_usd / float(price_to)
                    _ledger_buy(to_token, to_chain, to_specific_chain or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
            else:
                amt_from = float(amount_human)
                proceeds_usd = float(amount_usd) if amount_usd is not None else (amt_from * float(price_from or 0))
                if amt_from and proceeds_usd:
                    _ledger_sell(from_token, chain, specific_chain or "", amount_from=amt_from, proceeds_usd=proceeds_usd, symbol=None)
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        try:
            _append_trade({
                "type": "manual",
                "error": str(e),
                "status": "error",
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/api/bridge")
def api_bridge(body: dict = Body(...)):
    try:
        from_chain = body.get("fromChain", "evm")
        from_specific = body.get("fromSpecificChain")
        if from_specific is not None:
            from_specific = str(from_specific).strip() or None
        if from_specific is None:
            from_specific = _default_specific_for_chain(from_chain)
        to_chain = body.get("toChain") or body.get("targetChain")
        if not to_chain:
            to_chain = from_chain
        to_specific = body.get("toSpecificChain")
        if to_specific is not None:
            to_specific = str(to_specific).strip() or None
        if to_specific is None:
            to_specific = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        amount_human = body.get("amountHuman")
        amount_usd = body.get("amountUsd")
        reason = body.get("reason", "bridge")

        if not from_token or not to_token:
            return JSONResponse({"error": "fromToken dan toToken wajib diisi"}, status_code=400)

        if amount_human is None:
            amt = float(amount_usd or 0)
            if amt <= 0:
                return JSONResponse({"error": "amountUsd harus > 0 atau sertakan amountHuman"}, status_code=400)
            price_from = _get_price(from_token, from_chain, from_specific)
            if not price_from:
                return JSONResponse({"error": "Harga fromToken tidak tersedia"}, status_code=500)
            amount_human = amt / float(price_from)
        amount_human = float(amount_human)

        payload = {
            "fromToken": from_token,
            "toToken": to_token,
            "amount": str(round(amount_human, 8)),
            "reason": reason,
            "slippageTolerance": "0.5",
            "fromChain": from_chain,
            "toChain": to_chain,
        }
        if from_specific:
            payload["fromSpecificChain"] = from_specific
        if to_specific:
            payload["toSpecificChain"] = to_specific

        upstream_status = None
        upstream_text = None
        try:
            resp, upstream_status, upstream_text = _execute_trade_payload(payload)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            body_txt = getattr(getattr(exc, "response", None), "text", None)
            try:
                _append_trade({
                    "type": "bridge",
                    "error": str(exc),
                    "status": "error",
                    "body": body,
                    "payload": payload,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                })
            except Exception:
                pass
            return JSONResponse({
                "error": str(exc),
                "upstream_status": status if status is not None else upstream_status,
                "upstream_body": body_txt if body_txt is not None else upstream_text,
            }, status_code=status if isinstance(status, int) else 500)

        try:
            _append_trade({
                "type": "bridge",
                "status": "ok",
                "fromChain": from_chain,
                "toChain": to_chain,
                "fromToken": from_token,
                "toToken": to_token,
                "amountHuman": amount_human,
                "amountUsd": float(amount_usd) if amount_usd is not None else None,
                "reason": reason,
                "payload": payload,
                "response": resp,
            })
            _ensure_token_known(from_token, None, from_chain, from_specific or "")
            _ensure_token_known(to_token, None, to_chain, to_specific or "")
            try:
                price_from = _get_price(from_token, from_chain, from_specific)
            except Exception:
                price_from = None
            cost_usd = float(amount_usd) if amount_usd is not None else (float(amount_human) * float(price_from or 0))
            if cost_usd:
                _ledger_sell(from_token, from_chain, from_specific or "", amount_from=float(amount_human), proceeds_usd=cost_usd, symbol=None)
            try:
                price_to = _get_price(to_token, to_chain, to_specific)
            except Exception:
                price_to = None
            if cost_usd and price_to:
                amount_to = cost_usd / float(price_to)
                _ledger_buy(to_token, to_chain, to_specific or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        try:
            _append_trade({
                "type": "bridge",
                "status": "error",
                "error": str(e),
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Auto Trade (scheduler, disabled by default) ----------
AUTO_ENABLED  = str(os.getenv("AUTO_TRADE_ENABLED", "0")).lower() in {"1","true","yes","on"}
AUTO_INTERVAL = int(os.getenv("AUTO_TRADE_INTERVAL", "120"))  # seconds
AUTO_TARGET_PCT = float(os.getenv("AUTO_TRADE_TARGET_PCT", "10"))
AUTO_MAX_TRADE = float(os.getenv("AUTO_TRADE_MAX_TRADE_USD", "250"))
AUTO_RESERVE   = float(os.getenv("AUTO_TRADE_RESERVE_USD", "50"))
AUTO_MIN_DELTA = float(os.getenv("AUTO_TRADE_MIN_DELTA_USD", "10"))  # minimal delta/trade USD to trigger
AUTO_TARGET_TOKEN = os.getenv("AUTO_TRADE_TARGET_TOKEN", WETH_ETH)
AUTO_CASH_TOKEN   = os.getenv("AUTO_TRADE_CASH_TOKEN", USDC_ETH)
AUTO_DRY_RUN = str(os.getenv("AUTO_TRADE_DRY_RUN", "1")).lower() in {"1","true","yes","on"}

_auto_state = {
    "enabled": AUTO_ENABLED,
    "interval": AUTO_INTERVAL,
    "lastRun": None,
    "lastAction": None,
    "dryRun": AUTO_DRY_RUN,
}

def _auto_cfg_from_env():
    return {
        "targetPct": AUTO_TARGET_PCT,
        "maxTradeUsd": AUTO_MAX_TRADE,
        "reserveUsd": AUTO_RESERVE,
        "minDeltaUsd": AUTO_MIN_DELTA,
        "targetToken": AUTO_TARGET_TOKEN,
        "cashToken": AUTO_CASH_TOKEN,
    }

def _autotrade_once():
    try:
        cfg = _auto_cfg_from_env()
        # Reuse logic like api_rebalance, but with min-delta and dry-run gate
        target_pct     = float(cfg.get("targetPct", 10.0)) / 100.0
        max_trade_usd  = float(cfg.get("maxTradeUsd", 250.0))
        reserve_usd    = float(cfg.get("reserveUsd", 50.0))
        min_delta_usd  = float(cfg.get("minDeltaUsd", 10.0))
        chain          = "evm"
        specific_chain = "eth"
        target_token   = str(cfg.get("targetToken", WETH_ETH))
        cash_token     = str(cfg.get("cashToken", USDC_ETH))

        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        b  = rb.get("balances", [])
        total = sum(x.get("value",0.0) for x in b)
        target_val = 0.0
        cash_amt = 0.0
        cash_price = 1.0
        for x in b:
            if x.get("chain") == chain and x.get("specificChain") == specific_chain:
                if x.get("tokenAddress"," ").lower()==target_token.lower():
                    target_val = x.get("value",0.0)
                if x.get("tokenAddress"," ").lower()==cash_token.lower():
                    cash_amt   = x.get("amount",0.0)
                    if x.get("price") is not None:
                        cash_price = x.get("price")

        desired_val = total * target_pct
        delta = desired_val - target_val
        _auto_state["lastRun"] = datetime.now(timezone.utc).isoformat()

        if delta <= 0.0 or delta < min_delta_usd:
            _auto_state["lastAction"] = {"message": "No action: delta small or negative", "deltaUsd": float(delta)}
            return {"skipped": True, "reason": "small_or_negative_delta", "deltaUsd": float(delta)}

        available = max(0.0, cash_amt*cash_price - reserve_usd)
        trade_usd = min(delta, max_trade_usd, available)
        if trade_usd < min_delta_usd:
            _auto_state["lastAction"] = {"message": "No action: insufficient cash", "deltaUsd": float(delta), "available": float(available)}
            return {"skipped": True, "reason": "insufficient_cash", "deltaUsd": float(delta), "available": float(available)}

        if cash_price in (None, 0):
            try:
                cash_price = _get_price(cash_token, chain, specific_chain)
            except Exception:
                cash_price = 1.0
        amount_from = trade_usd / (cash_price or 1.0)
        payload = {
            "fromToken": cash_token, "toToken": target_token,
            "amount": str(round(amount_from, 8)),
            "reason": f"auto-rebalance to {int(target_pct*100)}% target",
            "slippageTolerance": "0.5",
            "fromChain": chain, "fromSpecificChain": specific_chain,
            "toChain":   chain, "toSpecificChain":   specific_chain,
        }

        if _auto_state.get("dryRun", True):
            log = {"type": "auto", "mode": "dry", "fromToken": cash_token, "toToken": target_token, "amountFrom": amount_from, "tradeUsd": trade_usd, "payload": payload, "status": "dry"}
            _append_trade(log)
            _auto_state["lastAction"] = log
            return {"dryRun": True, **log}

        r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
        r.raise_for_status()
        resp = r.json()
        log = {"type": "auto", "mode": "live", "fromToken": cash_token, "toToken": target_token, "amountFrom": amount_from, "tradeUsd": trade_usd, "payload": payload, "response": resp, "status": "ok"}
        _append_trade(log)
        _auto_state["lastAction"] = log
        return log
    except Exception as e:
        err = {"error": str(e)}
        try:
            _append_trade({"type": "auto", "status": "error", **err})
        except Exception:
            pass
        _auto_state["lastAction"] = err
        return err


@app.get("/api/auto-trade/status")
def api_auto_status():
    cfg = _auto_cfg_from_env()
    return JSONResponse({"state": _auto_state, "config": cfg})


@app.post("/api/auto-trade/toggle")
def api_auto_toggle(body: dict = Body(...)):
    try:
        enabled = bool(body.get("enabled", False))
        dry = body.get("dryRun")
        _auto_state["enabled"] = enabled
        if dry is not None:
            _auto_state["dryRun"] = bool(dry)
        return JSONResponse({"ok": True, "state": _auto_state})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/auto-trade/run-once")
def api_auto_run_once():
    try:
        out = _autotrade_once()
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.on_event("startup")
async def _start_autotrader():
    async def loop():
        while True:
            try:
                if _auto_state.get("enabled", False):
                    _autotrade_once()
            except Exception:
                pass
            await asyncio.sleep(max(5, int(_auto_state.get("interval", AUTO_INTERVAL) or 60)))
    try:
        asyncio.create_task(loop())
    except Exception:
        pass
=======
import os
import asyncio
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Request, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, timezone
from datetime import datetime, timezone

# === Load env ===
load_dotenv()
API_KEY = os.getenv("RECALL_API_KEY")
BASE    = os.getenv("RECALL_API_URL", "https://api.sandbox.competitions.recall.network")
if not API_KEY:
    raise SystemExit("RECALL_API_KEY belum ada di .env")

USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_ETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"


def hdrs():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _default_specific_for_chain(chain: str | None):
    if not chain:
        return None
    c = str(chain).lower()
    if c == "evm":
        return "eth"
    if c == "solana":
        return "sol"
    if c == "svm":
        return "svm"
    return None


def _execute_trade_payload(payload: dict):
    upstream_text = None
    upstream_status = None

    def _execute(p: dict):
        return requests.post(f"{BASE}/api/trade/execute", json=p, headers=hdrs(), timeout=30)

    r = _execute(payload)
    upstream_text = r.text
    upstream_status = r.status_code
    if r.status_code in (400, 404) and (str(payload.get("fromChain")) == "svm" or str(payload.get("toChain")) == "svm"):
        fallback = dict(payload)
        changed = False
        if str(fallback.get("fromChain")) == "svm":
            fallback["fromChain"] = "solana"
            if not fallback.get("fromSpecificChain"):
                fallback["fromSpecificChain"] = "svm"
            changed = True
        if str(fallback.get("toChain")) == "svm":
            fallback["toChain"] = "solana"
            if not fallback.get("toSpecificChain"):
                fallback["toSpecificChain"] = "svm"
            changed = True
        if changed:
            r2 = _execute(fallback)
            upstream_text = r2.text
            upstream_status = r2.status_code
            r2.raise_for_status()
            return r2.json(), upstream_status, upstream_text
        r.raise_for_status()
        return r.json(), upstream_status, upstream_text
    r.raise_for_status()
    return r.json(), upstream_status, upstream_text




def _get_price(token: str, chain: str = "evm", specific_chain: str | None = "eth") -> float:
    params = {"token": token, "chain": chain}
    if specific_chain:
        params["specificChain"] = specific_chain
    r = requests.get(
        f"{BASE}/api/price",
        params=params,
        headers=hdrs(), timeout=20
    )
    r.raise_for_status()
    data = r.json()
    return float(data.get("price", 0))


# === FastAPI ===
app = FastAPI(title="Recall Agent Dashboard")
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str((BASE_DIR / "static").resolve())), name="static")
templates = Jinja2Templates(directory=str((BASE_DIR / "templates").resolve()))

# Optional LLM module (robust local import)
llm_mod = None
try:
    # Try regular import if running inside the folder
    import llm as _llm
    llm_mod = _llm
except Exception:
    try:
        import importlib.util, pathlib, sys
        here = pathlib.Path(__file__).resolve().parent
        llm_path = here / "llm.py"
        if llm_path.exists():
            spec = importlib.util.spec_from_file_location("llm", str(llm_path))
            _mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(_mod)  # type: ignore
            llm_mod = _mod
    except Exception:
        llm_mod = None


# ---------- Pages ----------
@app.get("/")
def home(request: Request):
    # Info dasar tidak mengandung secret (API key hanya di server)
    ctx = {"request": request, "base_url": BASE, "hide_small_default": float(os.getenv("BALANCE_HIDE_USD_THRESHOLD", "1"))}
    return templates.TemplateResponse("index.html", ctx)


# ---------- API Proxies ----------
@app.get("/api/balances")
def api_balances(minUsd: float | None = Query(None)):
    try:
        upstream_text = None
        upstream_status = None
        r = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20)
        upstream_text = r.text
        upstream_status = r.status_code
        r.raise_for_status()
        resp = r.json()
        try:
            if minUsd is None:
                threshold = float(os.getenv("BALANCE_HIDE_USD_THRESHOLD", "1"))
            else:
                threshold = max(0.0, float(minUsd))
            items = []
            for it in resp.get("balances", []):
                val = float(it.get("value", 0) or 0)
                token_addr = str(it.get("tokenAddress", ""))
                kind = str(it.get("kind", "")).lower()
                is_token = bool(token_addr) or kind == "token"
                if is_token and val < threshold:
                    continue
                items.append(it)
            resp["balances"] = items
            resp["minUsd"] = threshold
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "upstream_status": upstream_status,
            "upstream_body": upstream_text,
        }, status_code=500)


@app.get("/api/price")
def api_price(token: str = Query(...), chain: str = "evm", specificChain: str = "eth"):
    try:
        upstream_text = None
        upstream_status = None
        r = requests.get(
            f"{BASE}/api/price",
            params={"token": token, "chain": chain, "specificChain": specificChain},
            headers=hdrs(), timeout=20,
        )
        upstream_text = r.text
        upstream_status = r.status_code
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "upstream_status": upstream_status,
            "upstream_body": upstream_text,
        }, status_code=500)


@app.get("/api/ai/suggest-rebalance")
def api_ai_suggest_rebalance():
    if not llm_mod:
        return JSONResponse({"error": "LLM module not available. Install 'openai' and ensure llm.py present."}, status_code=500)
    try:
        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        balances = rb.get("balances", [])
        data = llm_mod.suggest_rebalance_with_llm(balances, default_target=WETH_ETH, default_cash=USDC_ETH)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ai/status")
def api_ai_status(test: int = 0):
    """Return whether LLM is available and API key present. If test=1, try init client."""
    info = {"available": False, "hasKey": False, "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
    if not llm_mod:
        return JSONResponse(info)
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    info["hasKey"] = has_key
    if not has_key:
        return JSONResponse(info)
    if test:
        try:
            # Only try to construct client; avoid actual network call for speed
            _ = llm_mod._get_openai_client()
            info["available"] = True
        except Exception as e:
            return JSONResponse({**info, "error": str(e)})
        return JSONResponse(info)
    # Without test flag, report optimistic availability if key exists and module present
    info["available"] = True
    return JSONResponse(info)

# ---------- Token Registry (server-side) ----------
import json
from pathlib import Path

TOKENS_FILE = Path(__file__).resolve().parent / "tokens.json"

DEFAULT_TOKENS = [
    {"symbol": "USDC", "address": USDC_ETH, "chain": "evm", "specificChain": "eth"},
    {"symbol": "WETH", "address": WETH_ETH, "chain": "evm", "specificChain": "eth"},
    {"symbol": "USDT", "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "chain": "evm", "specificChain": "eth"},
    {"symbol": "DAI",  "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "chain": "evm", "specificChain": "eth"},
    {"symbol": "UNI",  "address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "chain": "evm", "specificChain": "eth"},
    # EVM: Polygon
    {"symbol": "USDC", "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "WETH", "address": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "USDT", "address": "0xC2132D05D31c914a87C6611C10748AEB04B58e8F", "chain": "evm", "specificChain": "polygon"},
    {"symbol": "DAI",  "address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063", "chain": "evm", "specificChain": "polygon"},
    # EVM: Base (partial known set)
    {"symbol": "WETH", "address": "0x4200000000000000000000000000000000000006", "chain": "evm", "specificChain": "base"},
    # Solana (SPL mints)
    {"symbol": "wSOL", "address": "So11111111111111111111111111111111111111112", "chain": "solana", "specificChain": "sol"},
    {"symbol": "USDC", "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "chain": "solana", "specificChain": "sol"},
]

def _load_tokens():
    if TOKENS_FILE.exists():
        try:
            with open(TOKENS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Merge defaults that are missing by (chain, specific, address)
                    base = list(data)
                    seen = set((str(t.get("chain","")), str(t.get("specificChain","")), str(t.get("address","")) .lower()) for t in base)
                    for d in DEFAULT_TOKENS:
                        key = (str(d.get("chain","")), str(d.get("specificChain","")), str(d.get("address","")) .lower())
                        if key not in seen:
                            base.append(d)
                            seen.add(key)
                    return base
        except Exception:
            pass
    return DEFAULT_TOKENS.copy()

def _save_tokens(tokens):
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

def _ensure_token_known(address: str, symbol_hint: str | None = None, chain: str = "evm", specific: str = "eth"):
    try:
        addr = str(address or "").strip()
        if not addr.lower().startswith("0x") or len(addr) != 42:
            return
        toks = _load_tokens()
        key = (chain, specific, addr.lower())
        for t in toks:
            if t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == addr.lower():
                return  # already known
        # Add with placeholder symbol if hint missing
        sym = (symbol_hint or ("TKN-" + addr[-4:])).upper()
        toks.append({"symbol": sym, "address": addr, "chain": chain, "specificChain": specific})
        _save_tokens(toks)
    except Exception:
        pass


# ---------- Trade History (server-side) ----------
TRADES_FILE = Path(__file__).resolve().parent / "trades.jsonl"

def _append_trade(entry: dict):
    try:
        entry = dict(entry)
        entry.setdefault("serverTime", datetime.now(timezone.utc).isoformat())
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort only
        pass

def _read_trades(limit: int = 100):
    items = []
    try:
        if not TRADES_FILE.exists():
            return []
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return list(reversed(items))[: max(0, int(limit or 0)) or 100]
    except Exception:
        return []

@app.get("/api/trades")
def api_trades(limit: int = 50):
    try:
        return JSONResponse({"trades": _read_trades(limit=limit)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trades/clear")
def api_trades_clear():
    try:
        if TRADES_FILE.exists():
            TRADES_FILE.unlink(missing_ok=True)  # type: ignore
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- PnL Ledger (server-side) ----------
LEDGER_FILE = Path(__file__).resolve().parent / "pnl_ledger.json"

def _load_ledger():
    base = {"positions": {}, "realized": [], "stats": {"realizedUsd": 0.0}}
    try:
        if LEDGER_FILE.exists():
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {**base, **data}
    except Exception:
        pass
    return base

def _save_ledger(ledger):
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

def _pos_key(addr: str, chain: str, specific: str) -> str:
    return f"{(chain or 'evm').lower()}:{(specific or 'eth').lower()}:{addr.lower()}"

def _symbol_of(addr: str) -> str:
    try:
        for t in _load_tokens():
            if str(t.get("address","")) .lower() == str(addr).lower():
                return str(t.get("symbol",""))
    except Exception:
        pass
    return ""

def _ledger_buy(addr: str, chain: str, specific: str, amount_to: float, cost_usd: float, symbol: str | None = None):
    if amount_to <= 0 or cost_usd <= 0:
        return
    ledger = _load_ledger()
    key = _pos_key(addr, chain, specific)
    pos = ledger["positions"].get(key) or {"symbol": symbol or _symbol_of(addr) or addr, "amount": 0.0, "investedUsd": 0.0}
    pos["amount"] = float(pos.get("amount", 0.0)) + float(amount_to)
    pos["investedUsd"] = float(pos.get("investedUsd", 0.0)) + float(cost_usd)
    ledger["positions"][key] = pos
    _save_ledger(ledger)

def _ledger_sell(addr: str, chain: str, specific: str, amount_from: float, proceeds_usd: float, symbol: str | None = None):
    if amount_from <= 0 or proceeds_usd < 0:
        return
    ledger = _load_ledger()
    key = _pos_key(addr, chain, specific)
    pos = ledger["positions"].get(key) or {"symbol": symbol or _symbol_of(addr) or addr, "amount": 0.0, "investedUsd": 0.0}
    cur_amt = float(pos.get("amount", 0.0))
    cur_inv = float(pos.get("investedUsd", 0.0))
    sell_amt = min(cur_amt, float(amount_from))
    avg_cost_per_unit = (cur_inv / cur_amt) if cur_amt > 0 else 0.0
    cost_usd = sell_amt * avg_cost_per_unit
    pnl = float(proceeds_usd) - cost_usd
    # update position
    pos["amount"] = cur_amt - sell_amt
    pos["investedUsd"] = max(0.0, cur_inv - cost_usd)
    ledger["positions"][key] = pos
    # realized log
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "token": symbol or pos.get("symbol") or addr,
        "address": addr,
        "chain": chain,
        "specificChain": specific,
        "side": "sell",
        "amount": sell_amt,
        "proceedsUsd": float(proceeds_usd),
        "costUsd": cost_usd,
        "pnlUsd": pnl,
    }
    ledger["realized"].append(event)
    ledger["stats"]["realizedUsd"] = float(ledger["stats"].get("realizedUsd", 0.0)) + pnl
    _save_ledger(ledger)

@app.get("/api/pnl")
def api_pnl():
    try:
        ledger = _load_ledger()
        positions = ledger.get("positions", {})
        enriched = {}
        total_mkt = 0.0
        total_inv = 0.0
        total_unrl = 0.0
        for key, pos in positions.items():
            try:
                _, specific, addr = key.split(":", 2)
            except Exception:
                specific, addr = "eth", key
            amt = float(pos.get("amount", 0.0))
            inv = float(pos.get("investedUsd", 0.0))
            price = 0.0
            if amt > 0:
                try:
                    price = _get_price(addr, chain="evm", specific_chain=specific)
                except Exception:
                    price = 0.0
            mkt = amt * (price or 0.0)
            unrl = mkt - inv
            total_mkt += mkt
            total_inv += inv
            total_unrl += unrl
            enriched[key] = {
                **pos,
                "avgCostPerUnitUsd": (inv / amt) if amt > 0 else 0.0,
                "marketPriceUsd": price,
                "marketValueUsd": mkt,
                "unrealizedUsd": unrl,
            }
        return JSONResponse({
            "positions": enriched,
            "realized": ledger.get("realized", []),
            "stats": {
                "realizedUsd": float(ledger.get("stats", {}).get("realizedUsd", 0.0)),
                "unrealizedUsd": total_unrl,
                "marketValueUsd": total_mkt,
                "investedUsd": total_inv,
            },
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
@app.get("/api/tokens")
def api_tokens_list():
    try:
        # Deduplicate by address within same chain/specific, preserving first (canonical) occurrence
        raw = _load_tokens()
        seen = set()
        uniq = []
        for t in raw:
            key = (t.get("chain","evm"), t.get("specificChain","eth"), str(t.get("address","")) .lower())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        return JSONResponse({"tokens": uniq})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tokens")
def api_tokens_add(body: dict = Body(...)):
    try:
        symbol = str(body.get("symbol", "")).strip().upper()
        address = str(body.get("address", "")).strip()
        chain = body.get("chain", "evm")
        # default specific by chain; svm has none
        specific = body.get("specificChain")
        if not specific:
            if str(chain).lower() == "evm":
                specific = "eth"
            elif str(chain).lower() == "solana":
                specific = "sol"
            else:
                specific = ""
        if not symbol or not address:
            return JSONResponse({"error": "symbol dan address wajib diisi"}, status_code=400)
        # Address validation by chain
        if str(chain).lower() == "evm":
            if not address.lower().startswith("0x") or len(address) != 42:
                return JSONResponse({"error": "address tidak valid (EVM harus 0x.. 42 chars)"}, status_code=400)
        elif str(chain).lower() in {"solana", "svm"}:
            base58chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
            if any(c not in base58chars for c in address) or address.lower().startswith("0x") or len(address) < 32 or len(address) > 60:
                return JSONResponse({"error": "address tidak valid (base58 untuk Solana/SVM)"}, status_code=400)
        tokens = _load_tokens()
        # Reject duplicates by address (same chain/specific) to avoid wrong symbol mapping
        for t in tokens:
            if t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == address.lower():
                if t.get("symbol","" ).upper() != symbol:
                    return JSONResponse({"error": f"address sudah terdaftar untuk simbol {t.get('symbol')}"}, status_code=400)
                # if same symbol/address, allow idempotent update
                break
        # update if exists by symbol (same chain/specific), else append
        updated = False
        for t in tokens:
            if t.get("symbol", "").upper() == symbol and t.get("specificChain") == specific and t.get("chain") == chain:
                t["address"] = address
                updated = True
                break
        if not updated:
            tokens.append({"symbol": symbol, "address": address, "chain": chain, "specificChain": specific})
        _save_tokens(tokens)
        return JSONResponse({"ok": True, "tokens": tokens})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tokens/delete")
def api_tokens_delete(body: dict = Body(...)):
    try:
        address = str(body.get("address", "")).strip()
        symbol = str(body.get("symbol", "")).strip().upper()
        chain = body.get("chain", "evm")
        specific = body.get("specificChain")
        if not specific:
            if str(chain).lower() == "evm":
                specific = "eth"
            elif str(chain).lower() == "solana":
                specific = "sol"
            else:
                specific = ""
        tokens = _load_tokens()
        if address:
            addr_low = address.lower()
            tokens = [
                t for t in tokens
                if not (t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("address","")) .lower() == addr_low)
            ]
        elif symbol:
            tokens = [
                t for t in tokens
                if not (t.get("chain") == chain and t.get("specificChain") == specific and str(t.get("symbol","")) .upper() == symbol)
            ]
        else:
            return JSONResponse({"error": "wajib mengirim address atau symbol"}, status_code=400)
        _save_tokens(tokens)
        return JSONResponse({"ok": True, "tokens": tokens})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/rebalance")
def api_rebalance(body: dict = Body(...)):
    """
    body: {
      targetPct: float(0..100), maxTradeUsd: float, reserveUsd|usdcReserveUsd: float,
      targetToken?: str, cashToken?: str, chain?: "evm", specificChain?: "eth"
    }
    Rebalance target token ke target % memakai cash token.
    """
    try:
        target_pct     = float(body.get("targetPct", 10.0))/100.0
        max_trade_usd  = float(body.get("maxTradeUsd", 500.0))
        reserve_usd    = float(body.get("usdcReserveUsd", body.get("reserveUsd", 50.0)))
        chain          = body.get("chain", "evm")
        specific_chain = body.get("specificChain", "eth")
        # Default tokens only for EVM/ETH; otherwise require explicit tokens
        if str(chain).lower() == "evm" and str(specific_chain).lower() == "eth":
            target_token   = str(body.get("targetToken", WETH_ETH))
            cash_token     = str(body.get("cashToken", USDC_ETH))
        else:
            if not body.get("targetToken") or not body.get("cashToken"):
                return JSONResponse({"error": "targetToken dan cashToken wajib diisi untuk chain selain evm/eth"}, status_code=400)
            target_token = str(body.get("targetToken"))
            cash_token = str(body.get("cashToken"))

        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        b  = rb.get("balances", [])
        total = sum(x.get("value",0.0) for x in b)
        target_val = 0.0
        target_price = None
        target_amount = 0.0
        cash_amt = 0.0
        cash_price = 1.0
        for x in b:
            if x.get("chain") == chain and x.get("specificChain") == specific_chain:
                if x["tokenAddress"].lower()==target_token.lower():
                    target_val = x.get("value",0.0)
                    target_amount = x.get("amount",0.0)
                    if x.get("price") is not None:
                        target_price = x.get("price")
                if x["tokenAddress"].lower()==cash_token.lower():
                    cash_amt   = x.get("amount",0.0)
                    if x.get("price") is not None:
                        cash_price = x.get("price")

        target_val_goal = total * target_pct
        delta = target_val_goal - target_val
        allow_sell = bool(body.get("allowSell", False))
        if delta <= 0.0:
            if not allow_sell:
                return JSONResponse({"message":"Sudah >= target, tidak perlu beli.", "deltaUsd": delta})
            # Sell-down mode: kurangi ke target dengan menjual target token ke cash token
            overweight_usd = min(abs(delta), max_trade_usd, target_val)
            if overweight_usd <= 0.0:
                return JSONResponse({"message":"Tidak ada nilai untuk dijual.", "deltaUsd": delta})
            # Pastikan harga target tersedia untuk konversi unit
            if target_price in (None, 0):
                try:
                    target_price = _get_price(target_token, chain, specific_chain)
                except Exception:
                    target_price = None
            if not target_price:
                return JSONResponse({"error":"Harga target token tidak tersedia untuk jual"}, status_code=500)
            amount_target = overweight_usd / float(target_price)
            payload = {
                "fromToken": target_token, "toToken": cash_token,
                "amount": str(round(amount_target, 8)),
                "reason": f"rebalance sell-down from {int(target_pct*100)}% target",
                "slippageTolerance": "0.5",
                "fromChain": chain, "fromSpecificChain": specific_chain,
                "toChain":   chain, "toSpecificChain":   specific_chain,
            }
            r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
            r.raise_for_status()
            result = {
                "action": "sell",
                "targetPct": target_pct,
                "deltaUsd": delta,
                "tradeUsd": overweight_usd,
                "amountFrom": amount_target,
                "fromToken": target_token,
                "toToken": cash_token,
                "tx": r.json(),
            }
            try:
                _append_trade({
                    "type": "rebalance",
                    "action": "sell",
                    "fromToken": target_token,
                    "toToken": cash_token,
                    "amountFrom": amount_target,
                    "tradeUsd": overweight_usd,
                    "payload": payload,
                    "response": result.get("tx"),
                    "status": "ok",
                })
                # PnL: realized on sell
                _ledger_sell(target_token, chain, specific_chain, amount_from=float(amount_target), proceeds_usd=float(overweight_usd), symbol=None)
            except Exception:
                pass
            return JSONResponse(result)

        available = max(0.0, cash_amt*cash_price - reserve_usd)
        trade_usd = min(delta, max_trade_usd, available)
        if trade_usd <= 0.0:
            return JSONResponse({"message":"Cash token tidak cukup setelah cadangan.", "deltaUsd": delta})

        # convert USD to from-token units via price
        if cash_price in (None, 0):
            try:
                cash_price = _get_price(cash_token, chain, specific_chain)
            except Exception:
                pass
        amount_from = trade_usd / (cash_price or 1.0)
        payload = {
            "fromToken": cash_token, "toToken": target_token,
            "amount": str(round(amount_from, 8)),
            "reason": f"rebalance to {int(target_pct*100)}% target",
            "slippageTolerance": "0.5",
            "fromChain": chain, "fromSpecificChain": specific_chain,
            "toChain":   chain, "toSpecificChain":   specific_chain,
        }
        r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
        r.raise_for_status()
        result = {
            "action": "buy",
            "targetPct": target_pct,
            "deltaUsd": delta,
            "tradeUsd": trade_usd,
            "amountFrom": amount_from,
            "fromToken": cash_token,
            "toToken": target_token,
            "tx": r.json(),
        }
        try:
            _append_trade({
                "type": "rebalance",
                "fromToken": cash_token,
                "toToken": target_token,
                "amountFrom": amount_from,
                "tradeUsd": trade_usd,
                "payload": payload,
                "response": result.get("tx"),
                "status": "ok",
            })
            # PnL ledger (buy target token approx using price snapshot)
            try:
                price_to = _get_price(target_token, chain, specific_chain)
            except Exception:
                price_to = None
            if price_to:
                amount_to = float(trade_usd) / float(price_to)
                _ledger_buy(target_token, chain, specific_chain, amount_to=amount_to, cost_usd=float(trade_usd), symbol=None)
        except Exception:
            pass
        return JSONResponse(result)
    except Exception as e:
        try:
            _append_trade({
                "type": "rebalance",
                "error": str(e),
                "status": "error",
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/batch-trade")
def api_batch_trade(body: dict = Body(...)):
    try:
        side = str(body.get("side", "buy")).lower()
        if side not in {"buy", "sell"}:
            return JSONResponse({"error": "side harus buy atau sell"}, status_code=400)
        chain = body.get("chain", "evm")
        specific_chain = body.get("specificChain")
        if specific_chain is not None:
            specific_chain = str(specific_chain).strip() or None
        if specific_chain is None:
            specific_chain = _default_specific_for_chain(chain)
        to_chain = body.get("toChain", chain)
        to_specific_chain = body.get("toSpecificChain")
        if to_specific_chain is not None:
            to_specific_chain = str(to_specific_chain).strip() or None
        if to_specific_chain is None:
            to_specific_chain = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        total_usd = float(body.get("totalUsd") or 0)
        chunk_usd = float(body.get("chunkUsd") or 0)
        reason_base = str(body.get("reason", f"batch {side}") or f"batch {side}").strip() or f"batch {side}"
        if not from_token or not to_token:
            return JSONResponse({"error": "fromToken dan toToken wajib"}, status_code=400)
        if total_usd <= 0:
            return JSONResponse({"error": "totalUsd harus > 0"}, status_code=400)
        if chunk_usd <= 0:
            return JSONResponse({"error": "chunkUsd harus > 0"}, status_code=400)
        spent = 0.0
        iteration = 0
        chunks = []

        while spent + 1e-9 < total_usd:
            remaining = total_usd - spent
            usd_chunk = min(chunk_usd, remaining)
            try:
                price_from = _get_price(from_token, chain, specific_chain)
            except Exception as exc:
                return JSONResponse({"error": f"gagal ambil harga: {exc}"}, status_code=500)
            if not price_from:
                return JSONResponse({"error": "Harga fromToken tidak tersedia"}, status_code=500)
            amount_human = float(usd_chunk) / float(price_from)
            iteration += 1
            reason = f"{reason_base} #{iteration}"
            payload = {
                "fromToken": from_token,
                "toToken": to_token,
                "amount": str(round(float(amount_human), 8)),
                "reason": reason,
                "slippageTolerance": "0.5",
                "fromChain": chain,
                "toChain": to_chain,
            }
            if specific_chain:
                payload["fromSpecificChain"] = specific_chain
            if to_specific_chain:
                payload["toSpecificChain"] = to_specific_chain

            upstream_status = None
            upstream_text = None
            try:
                resp, upstream_status, upstream_text = _execute_trade_payload(payload)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                body_txt = getattr(getattr(exc, "response", None), "text", None)
                try:
                    _append_trade({
                        "type": "batch",
                        "side": side,
                        "status": "error",
                        "iteration": iteration,
                        "error": str(exc),
                        "payload": payload,
                        "upstream_status": status if status is not None else upstream_status,
                        "upstream_body": body_txt if body_txt is not None else upstream_text,
                    })
                except Exception:
                    pass
                return JSONResponse({
                    "error": str(exc),
                    "iteration": iteration,
                    "spent": spent,
                    "chunks": chunks,
                    "side": side,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                }, status_code=status if isinstance(status, int) else 500)

            chunk_entry = {
                "iteration": iteration,
                "usd": usd_chunk,
                "amountHuman": amount_human,
                "response": resp,
                "side": side,
            }
            chunks.append(chunk_entry)
            spent += usd_chunk
            try:
                log = {
                    "type": "batch",
                    "side": side,
                    "status": "ok",
                    "iteration": iteration,
                    "usd": usd_chunk,
                    "amountHuman": amount_human,
                    "payload": payload,
                    "response": resp,
                }
                _append_trade(log)
                if side == "buy":
                    _ensure_token_known(to_token, None, to_chain, to_specific_chain or "")
                else:
                    _ensure_token_known(from_token, None, chain, specific_chain or "")
                if side == "buy":
                    try:
                        price_to = _get_price(to_token, to_chain, to_specific_chain)
                    except Exception:
                        price_to = None
                    cost_usd = float(usd_chunk)
                    if price_to:
                        amount_to = cost_usd / float(price_to)
                        _ledger_buy(to_token, to_chain, to_specific_chain or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
                else:
                    _ledger_sell(from_token, chain, specific_chain or "", amount_from=float(amount_human), proceeds_usd=float(usd_chunk), symbol=None)
            except Exception:
                pass

        return JSONResponse({
            "status": "ok",
            "side": side,
            "totalUsd": spent,
            "chunks": chunks,
            "iterations": iteration,
        })
    except Exception as e:
        try:
            _append_trade({
                "type": "batch",
                "side": str(body.get("side")),
                "status": "error",
                "error": str(e),
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/api/manual-trade")
def api_manual_trade(body: dict = Body(...)):
    """
    body: {
      side: "buy"|"sell",
      amountUsd?: float, amountHuman?: float, reason?: str,
      fromToken?: str, toToken?: str, chain?: str, specificChain?: str
    }
    """
    try:
        side = body.get("side", "buy")
        reason = body.get("reason", "manual trade")
        chain = body.get("chain", "evm")
        specific_chain = body.get("specificChain")
        if specific_chain is not None:
            specific_chain = str(specific_chain).strip() or None
        if specific_chain is None:
            specific_chain = _default_specific_for_chain(chain)
        to_chain = body.get("toChain", chain)
        to_specific_chain = body.get("toSpecificChain")
        if to_specific_chain is not None:
            to_specific_chain = str(to_specific_chain).strip() or None
        if to_specific_chain is None:
            to_specific_chain = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        amount_human = body.get("amountHuman")
        amount_usd = body.get("amountUsd")

        if not from_token or not to_token:
            if chain == "evm":
                if side == "buy":
                    from_token, to_token = USDC_ETH, WETH_ETH
                else:
                    from_token, to_token = WETH_ETH, USDC_ETH
            else:
                return JSONResponse({"error": "fromToken/toToken wajib diisi untuk non-EVM"}, status_code=400)

        if amount_human is None:
            amt = float(amount_usd or 0)
            if amt <= 0:
                return JSONResponse({"error": "amountUsd must be > 0 or provide amountHuman"}, status_code=400)
            px = _get_price(from_token, chain, specific_chain)
            amount_human = amt / px if px else amt

        payload = {
            "fromToken": from_token,
            "toToken": to_token,
            "amount": str(round(float(amount_human), 8)),
            "reason": reason,
            "slippageTolerance": "0.5",
            "fromChain": chain,
            "toChain": to_chain,
        }
        if specific_chain:
            payload["fromSpecificChain"] = specific_chain
        if to_specific_chain:
            payload["toSpecificChain"] = to_specific_chain

        upstream_status = None
        upstream_text = None
        try:
            resp, upstream_status, upstream_text = _execute_trade_payload(payload)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            body_txt = getattr(getattr(exc, "response", None), "text", None)
            try:
                _append_trade({
                    "type": "manual",
                    "side": side,
                    "error": str(exc),
                    "status": "error",
                    "body": body,
                    "payload": payload,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                })
            except Exception:
                pass
            return JSONResponse({
                "error": str(exc),
                "upstream_status": status if status is not None else upstream_status,
                "upstream_body": body_txt if body_txt is not None else upstream_text,
            }, status_code=status if isinstance(status, int) else 500)

        try:
            _append_trade({
                "type": "manual",
                "side": side,
                "fromToken": from_token,
                "toToken": to_token,
                "amountHuman": float(amount_human),
                "amountUsd": float(amount_usd) if amount_usd is not None else None,
                "reason": reason,
                "payload": payload,
                "response": resp,
                "status": "ok",
            })
            if side == "buy":
                _ensure_token_known(to_token, None, to_chain, to_specific_chain or "")
            else:
                _ensure_token_known(from_token, None, chain, specific_chain or "")
            try:
                price_from = _get_price(from_token, chain, specific_chain)
            except Exception:
                price_from = None
            try:
                price_to = _get_price(to_token, to_chain, to_specific_chain)
            except Exception:
                price_to = None
            if side == "buy":
                cost_usd = float(amount_usd) if amount_usd is not None else (float(amount_human) * float(price_from or 0))
                if cost_usd and price_to:
                    amount_to = cost_usd / float(price_to)
                    _ledger_buy(to_token, to_chain, to_specific_chain or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
            else:
                amt_from = float(amount_human)
                proceeds_usd = float(amount_usd) if amount_usd is not None else (amt_from * float(price_from or 0))
                if amt_from and proceeds_usd:
                    _ledger_sell(from_token, chain, specific_chain or "", amount_from=amt_from, proceeds_usd=proceeds_usd, symbol=None)
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        try:
            _append_trade({
                "type": "manual",
                "error": str(e),
                "status": "error",
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/api/bridge")
def api_bridge(body: dict = Body(...)):
    try:
        from_chain = body.get("fromChain", "evm")
        from_specific = body.get("fromSpecificChain")
        if from_specific is not None:
            from_specific = str(from_specific).strip() or None
        if from_specific is None:
            from_specific = _default_specific_for_chain(from_chain)
        to_chain = body.get("toChain") or body.get("targetChain")
        if not to_chain:
            to_chain = from_chain
        to_specific = body.get("toSpecificChain")
        if to_specific is not None:
            to_specific = str(to_specific).strip() or None
        if to_specific is None:
            to_specific = _default_specific_for_chain(to_chain)
        from_token = body.get("fromToken")
        to_token = body.get("toToken")
        amount_human = body.get("amountHuman")
        amount_usd = body.get("amountUsd")
        reason = body.get("reason", "bridge")

        if not from_token or not to_token:
            return JSONResponse({"error": "fromToken dan toToken wajib diisi"}, status_code=400)

        if amount_human is None:
            amt = float(amount_usd or 0)
            if amt <= 0:
                return JSONResponse({"error": "amountUsd harus > 0 atau sertakan amountHuman"}, status_code=400)
            price_from = _get_price(from_token, from_chain, from_specific)
            if not price_from:
                return JSONResponse({"error": "Harga fromToken tidak tersedia"}, status_code=500)
            amount_human = amt / float(price_from)
        amount_human = float(amount_human)

        payload = {
            "fromToken": from_token,
            "toToken": to_token,
            "amount": str(round(amount_human, 8)),
            "reason": reason,
            "slippageTolerance": "0.5",
            "fromChain": from_chain,
            "toChain": to_chain,
        }
        if from_specific:
            payload["fromSpecificChain"] = from_specific
        if to_specific:
            payload["toSpecificChain"] = to_specific

        upstream_status = None
        upstream_text = None
        try:
            resp, upstream_status, upstream_text = _execute_trade_payload(payload)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            body_txt = getattr(getattr(exc, "response", None), "text", None)
            try:
                _append_trade({
                    "type": "bridge",
                    "error": str(exc),
                    "status": "error",
                    "body": body,
                    "payload": payload,
                    "upstream_status": status if status is not None else upstream_status,
                    "upstream_body": body_txt if body_txt is not None else upstream_text,
                })
            except Exception:
                pass
            return JSONResponse({
                "error": str(exc),
                "upstream_status": status if status is not None else upstream_status,
                "upstream_body": body_txt if body_txt is not None else upstream_text,
            }, status_code=status if isinstance(status, int) else 500)

        try:
            _append_trade({
                "type": "bridge",
                "status": "ok",
                "fromChain": from_chain,
                "toChain": to_chain,
                "fromToken": from_token,
                "toToken": to_token,
                "amountHuman": amount_human,
                "amountUsd": float(amount_usd) if amount_usd is not None else None,
                "reason": reason,
                "payload": payload,
                "response": resp,
            })
            _ensure_token_known(from_token, None, from_chain, from_specific or "")
            _ensure_token_known(to_token, None, to_chain, to_specific or "")
            try:
                price_from = _get_price(from_token, from_chain, from_specific)
            except Exception:
                price_from = None
            cost_usd = float(amount_usd) if amount_usd is not None else (float(amount_human) * float(price_from or 0))
            if cost_usd:
                _ledger_sell(from_token, from_chain, from_specific or "", amount_from=float(amount_human), proceeds_usd=cost_usd, symbol=None)
            try:
                price_to = _get_price(to_token, to_chain, to_specific)
            except Exception:
                price_to = None
            if cost_usd and price_to:
                amount_to = cost_usd / float(price_to)
                _ledger_buy(to_token, to_chain, to_specific or "", amount_to=amount_to, cost_usd=cost_usd, symbol=None)
        except Exception:
            pass
        return JSONResponse(resp)
    except Exception as e:
        try:
            _append_trade({
                "type": "bridge",
                "status": "error",
                "error": str(e),
                "body": body,
            })
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Auto Trade (scheduler, disabled by default) ----------
AUTO_ENABLED  = str(os.getenv("AUTO_TRADE_ENABLED", "0")).lower() in {"1","true","yes","on"}
AUTO_INTERVAL = int(os.getenv("AUTO_TRADE_INTERVAL", "120"))  # seconds
AUTO_TARGET_PCT = float(os.getenv("AUTO_TRADE_TARGET_PCT", "10"))
AUTO_MAX_TRADE = float(os.getenv("AUTO_TRADE_MAX_TRADE_USD", "250"))
AUTO_RESERVE   = float(os.getenv("AUTO_TRADE_RESERVE_USD", "50"))
AUTO_MIN_DELTA = float(os.getenv("AUTO_TRADE_MIN_DELTA_USD", "10"))  # minimal delta/trade USD to trigger
AUTO_TARGET_TOKEN = os.getenv("AUTO_TRADE_TARGET_TOKEN", WETH_ETH)
AUTO_CASH_TOKEN   = os.getenv("AUTO_TRADE_CASH_TOKEN", USDC_ETH)
AUTO_DRY_RUN = str(os.getenv("AUTO_TRADE_DRY_RUN", "1")).lower() in {"1","true","yes","on"}

_auto_state = {
    "enabled": AUTO_ENABLED,
    "interval": AUTO_INTERVAL,
    "lastRun": None,
    "lastAction": None,
    "dryRun": AUTO_DRY_RUN,
}

def _auto_cfg_from_env():
    return {
        "targetPct": AUTO_TARGET_PCT,
        "maxTradeUsd": AUTO_MAX_TRADE,
        "reserveUsd": AUTO_RESERVE,
        "minDeltaUsd": AUTO_MIN_DELTA,
        "targetToken": AUTO_TARGET_TOKEN,
        "cashToken": AUTO_CASH_TOKEN,
    }

def _autotrade_once():
    try:
        cfg = _auto_cfg_from_env()
        # Reuse logic like api_rebalance, but with min-delta and dry-run gate
        target_pct     = float(cfg.get("targetPct", 10.0)) / 100.0
        max_trade_usd  = float(cfg.get("maxTradeUsd", 250.0))
        reserve_usd    = float(cfg.get("reserveUsd", 50.0))
        min_delta_usd  = float(cfg.get("minDeltaUsd", 10.0))
        chain          = "evm"
        specific_chain = "eth"
        target_token   = str(cfg.get("targetToken", WETH_ETH))
        cash_token     = str(cfg.get("cashToken", USDC_ETH))

        rb = requests.get(f"{BASE}/api/agent/balances", headers=hdrs(), timeout=20).json()
        b  = rb.get("balances", [])
        total = sum(x.get("value",0.0) for x in b)
        target_val = 0.0
        cash_amt = 0.0
        cash_price = 1.0
        for x in b:
            if x.get("chain") == chain and x.get("specificChain") == specific_chain:
                if x.get("tokenAddress"," ").lower()==target_token.lower():
                    target_val = x.get("value",0.0)
                if x.get("tokenAddress"," ").lower()==cash_token.lower():
                    cash_amt   = x.get("amount",0.0)
                    if x.get("price") is not None:
                        cash_price = x.get("price")

        desired_val = total * target_pct
        delta = desired_val - target_val
        _auto_state["lastRun"] = datetime.now(timezone.utc).isoformat()

        if delta <= 0.0 or delta < min_delta_usd:
            _auto_state["lastAction"] = {"message": "No action: delta small or negative", "deltaUsd": float(delta)}
            return {"skipped": True, "reason": "small_or_negative_delta", "deltaUsd": float(delta)}

        available = max(0.0, cash_amt*cash_price - reserve_usd)
        trade_usd = min(delta, max_trade_usd, available)
        if trade_usd < min_delta_usd:
            _auto_state["lastAction"] = {"message": "No action: insufficient cash", "deltaUsd": float(delta), "available": float(available)}
            return {"skipped": True, "reason": "insufficient_cash", "deltaUsd": float(delta), "available": float(available)}

        if cash_price in (None, 0):
            try:
                cash_price = _get_price(cash_token, chain, specific_chain)
            except Exception:
                cash_price = 1.0
        amount_from = trade_usd / (cash_price or 1.0)
        payload = {
            "fromToken": cash_token, "toToken": target_token,
            "amount": str(round(amount_from, 8)),
            "reason": f"auto-rebalance to {int(target_pct*100)}% target",
            "slippageTolerance": "0.5",
            "fromChain": chain, "fromSpecificChain": specific_chain,
            "toChain":   chain, "toSpecificChain":   specific_chain,
        }

        if _auto_state.get("dryRun", True):
            log = {"type": "auto", "mode": "dry", "fromToken": cash_token, "toToken": target_token, "amountFrom": amount_from, "tradeUsd": trade_usd, "payload": payload, "status": "dry"}
            _append_trade(log)
            _auto_state["lastAction"] = log
            return {"dryRun": True, **log}

        r = requests.post(f"{BASE}/api/trade/execute", json=payload, headers=hdrs(), timeout=30)
        r.raise_for_status()
        resp = r.json()
        log = {"type": "auto", "mode": "live", "fromToken": cash_token, "toToken": target_token, "amountFrom": amount_from, "tradeUsd": trade_usd, "payload": payload, "response": resp, "status": "ok"}
        _append_trade(log)
        _auto_state["lastAction"] = log
        return log
    except Exception as e:
        err = {"error": str(e)}
        try:
            _append_trade({"type": "auto", "status": "error", **err})
        except Exception:
            pass
        _auto_state["lastAction"] = err
        return err


@app.get("/api/auto-trade/status")
def api_auto_status():
    cfg = _auto_cfg_from_env()
    return JSONResponse({"state": _auto_state, "config": cfg})


@app.post("/api/auto-trade/toggle")
def api_auto_toggle(body: dict = Body(...)):
    try:
        enabled = bool(body.get("enabled", False))
        dry = body.get("dryRun")
        _auto_state["enabled"] = enabled
        if dry is not None:
            _auto_state["dryRun"] = bool(dry)
        return JSONResponse({"ok": True, "state": _auto_state})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/auto-trade/run-once")
def api_auto_run_once():
    try:
        out = _autotrade_once()
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.on_event("startup")
async def _start_autotrader():
    async def loop():
        while True:
            try:
                if _auto_state.get("enabled", False):
                    _autotrade_once()
            except Exception:
                pass
            await asyncio.sleep(max(5, int(_auto_state.get("interval", AUTO_INTERVAL) or 60)))
    try:
        asyncio.create_task(loop())
    except Exception:
        pass
>>>>>>> a51f7a7 (push)
