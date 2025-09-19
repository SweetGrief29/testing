import os
import argparse
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Optional LLM module (robust local import)
llm_mod = None
try:
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

# ===== Konstanta token (ETH mainnet) =====
USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_ETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

# Token registry (ETH mainnet)
TOKEN_REGISTRY = {
    "ETH:USDC": USDC_ETH,
    "ETH:WETH": WETH_ETH,
    "ETH:USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "ETH:DAI":  "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "ETH:UNI":  "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
}

# ===== Parameter strategi =====
TARGET_WETH_PCT = 0.10  # target 10% dari total nilai portofolio
MIN_TRADE_USD = 5.0  # jangan eksekusi < $5
MAX_TRADE_USD = 500.0  # batasi ukuran trade per eksekusi
USDC_RESERVE_USD = 50.0  # sisakan cadangan USDC di ETH
SLIPPAGE_TOLERANCE_P = "0.5"  # 0.5%


def make_headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def get_balances(base: str, hdrs: Dict[str, str]) -> List[dict]:
    url = f"{base}/api/agent/balances"
    r = requests.get(url, headers=hdrs, timeout=20)
    r.raise_for_status()
    return r.json().get("balances", [])


def get_price(base: str, hdrs: Dict[str, str], token: str, chain: str = "evm", specific: str = "eth") -> float:
    url = f"{base}/api/price"
    r = requests.get(
        url,
        params={"token": token, "chain": chain, "specificChain": specific},
        headers=hdrs,
        timeout=20,
    )
    r.raise_for_status()
    return float(r.json().get("price", 0))


def resolve_token(token_or_symbol: str, specific: str = "eth") -> str:
    """Resolve ERC-20 address from symbol or return the address if already 0x..."""
    if not token_or_symbol:
        raise ValueError("Token kosong")
    t = token_or_symbol.strip()
    if t.lower().startswith("0x") and len(t) == 42:
        return t
    key = f"{specific.upper()}:{t.upper()}"
    if key in TOKEN_REGISTRY:
        return TOKEN_REGISTRY[key]
    # Allow common without specific prefix (assume ETH)
    key2 = f"ETH:{t.upper()}"
    if key2 in TOKEN_REGISTRY:
        return TOKEN_REGISTRY[key2]
    raise ValueError(f"Tidak menemukan alamat token untuk simbol: {t}")


def execute_trade(base: str, hdrs: Dict[str, str], from_token: str, to_token: str,
                  amount_from_human: float, reason: Optional[str] = None,
                  chain: str = "evm", specific: str = "eth") -> dict:
    url = f"{base}/api/trade/execute"
    payload = {
        "fromToken": from_token,
        "toToken": to_token,
        "amount": str(round(amount_from_human, 8)),
        "reason": reason or "manual trade",
        "slippageTolerance": SLIPPAGE_TOLERANCE_P,
        "fromChain": chain,
        "fromSpecificChain": specific,
        "toChain": chain,
        "toSpecificChain": specific,
    }
    r = requests.post(url, json=payload, headers=hdrs, timeout=30)
    print("Trade response status:", r.status_code)
    print(r.text)
    r.raise_for_status()
    return r.json()


def trade_value_to_amount_usd(value_usd: float, token_price: float) -> float:
    if token_price is None or token_price <= 0:
        raise ValueError("Harga token tidak valid untuk konversi")
    return value_usd / token_price


def run_rebalance(BASE: str, hdrs: Dict[str, str], target_pct: float = TARGET_WETH_PCT,
                  max_trade_usd: float = MAX_TRADE_USD, reserve_usd: float = USDC_RESERVE_USD,
                  min_trade_usd: float = MIN_TRADE_USD,
                  target_token: str = WETH_ETH, cash_token: str = USDC_ETH,
                  chain: str = "evm", specific: str = "eth") -> None:
    target_token = resolve_token(target_token, specific)
    cash_token = resolve_token(cash_token, specific)

    balances = get_balances(BASE, hdrs)
    total_value = sum(float(b.get("value", 0.0)) for b in balances)
    target_val = 0.0
    target_amount = 0.0
    target_price = None
    cash_amount = 0.0
    cash_price = 1.0

    for x in balances:
        try:
            token_addr = str(x.get("tokenAddress", "")).lower()
            if x.get("chain") == chain and x.get("specificChain") == specific:
                if token_addr == target_token.lower():
                    target_val = float(x.get("value", 0.0))
                    target_amount = float(x.get("amount", 0.0))
                    if x.get("price") is not None:
                        target_price = float(x.get("price"))
                if token_addr == cash_token.lower():
                    cash_amount = float(x.get("amount", 0.0))
                    if x.get("price") is not None:
                        cash_price = float(x.get("price"))
        except Exception:
            continue

    if target_price in (None, 0):
        try:
            target_price = get_price(BASE, hdrs, target_token, chain=chain, specific=specific)
        except Exception:
            pass

    print(f"Total Value: ${total_value:,.2f}")
    print(f"Target token value: ${target_val:,.2f}")
    print(f"Cash token amount: {cash_amount:,.6f} @ ${cash_price:.4f}")

    desired_val = total_value * target_pct
    delta = desired_val - target_val
    print(f"Target value: ${desired_val:,.2f} | Delta: ${delta:,.2f}")

    if abs(delta) <= min_trade_usd:
        print("Sudah dekat target (delta kecil). Tidak ada trade.")
        return

    if delta > 0:
        # Need to BUY target token using cash token
        available_cash_usd = max(0.0, cash_amount * cash_price - reserve_usd)
        trade_usd = min(delta, max_trade_usd, available_cash_usd)
        if trade_usd < min_trade_usd:
            print("Dana cash tidak cukup setelah cadangan. Skip trade.")
            return
        amount_from = trade_value_to_amount_usd(trade_usd, cash_price)
        print(f"BUY {amount_from:.6f} (from cash) ≈ ${trade_usd:,.2f}")
        execute_trade(BASE, hdrs, cash_token, target_token, amount_from,
                      reason=f"rebalance buy to {int(target_pct*100)}%", chain=chain, specific=specific)
    else:
        # Need to SELL target token
        if not target_price or target_price <= 0:
            raise SystemExit("Harga target token tidak tersedia untuk perhitungan jual.")
        overweight_usd = min(abs(delta), max_trade_usd, target_val)
        if overweight_usd < min_trade_usd:
            print("Delta jual kecil atau tidak ada nilai yang bisa dijual.")
            return
        amount_target = trade_value_to_amount_usd(overweight_usd, target_price)
        amount_target = min(amount_target, target_amount)
        if amount_target * target_price < min_trade_usd:
            print("Jumlah token untuk dijual terlalu kecil.")
            return
        print(f"SELL {amount_target:.6f} (target) ≈ ${overweight_usd:,.2f}")
        execute_trade(BASE, hdrs, target_token, cash_token, amount_target,
                      reason=f"rebalance sell from {int(target_pct*100)}%", chain=chain, specific=specific)


def run_manual(BASE: str, hdrs: Dict[str, str], side: str,
               from_token: str, to_token: str,
               amount_human: Optional[float], amount_usd: Optional[float],
               reason: Optional[str], chain: str = "evm", specific: str = "eth") -> None:
    if side not in {"buy", "sell"}:
        raise SystemExit("Side harus 'buy' atau 'sell'")
    from_token = resolve_token(from_token, specific)
    to_token = resolve_token(to_token, specific)
    if amount_human is None and (amount_usd is None or amount_usd <= 0):
        raise SystemExit("Wajib set --amount (human units) atau --amount-usd (>0)")
    if amount_human is None:
        # convert USD to human units using from-token price
        px = get_price(BASE, hdrs, from_token, chain=chain, specific=specific)
        amount_human = trade_value_to_amount_usd(amount_usd, px)
    print(f"Manual {side.upper()}: {amount_human:.6f} from {from_token} → {to_token}")
    execute_trade(BASE, hdrs, from_token, to_token, amount_human, reason=reason, chain=chain, specific=specific)


def run_batch_buy(
    BASE: str,
    hdrs: Dict[str, str],
    total_usd: float,
    chunk_usd: float = 1.0,
    from_token: str = 'USDC',
    to_token: str = 'WETH',
    chain: str = 'evm',
    specific: str = 'eth',
    reason: Optional[str] = None,
) -> None:
    """Execute repeated buy trades until total_usd is spent."""
    if total_usd <= 0:
        raise SystemExit('total_usd harus > 0')
    if chunk_usd <= 0:
        raise SystemExit('chunk_usd harus > 0')

    from_token_addr = resolve_token(from_token, specific)
    to_token_addr = resolve_token(to_token, specific)
    spent = 0.0
    iteration = 0
    base_reason = (reason or 'batch buy').strip() or 'batch buy'

    while spent + 1e-9 < total_usd:
        remaining = total_usd - spent
        usd_chunk = min(chunk_usd, remaining)
        price_from = get_price(BASE, hdrs, from_token_addr, chain=chain, specific=specific)
        amount_from = trade_value_to_amount_usd(usd_chunk, price_from)
        iteration += 1
        print(f"Chunk {iteration}: buy ${usd_chunk:.2f} -> {amount_from:.6f} units")
        print(f"Progress: ${spent + usd_chunk:.2f} / ${total_usd:.2f}")
        execute_trade(
            BASE,
            hdrs,
            from_token_addr,
            to_token_addr,
            amount_from,
            reason=f"{base_reason} #{iteration}",
            chain=chain,
            specific=specific,
        )
        spent += usd_chunk

    print(f"Selesai batch buy: total ${spent:.2f} dalam {iteration} langkah.")


def run_batch_sell(
    BASE: str,
    hdrs: Dict[str, str],
    total_usd: float,
    chunk_usd: float = 1.0,
    from_token: str = 'WETH',
    to_token: str = 'USDC',
    chain: str = 'evm',
    specific: str = 'eth',
    reason: Optional[str] = None,
) -> None:
    """Execute repeated sell trades until total_usd worth is sold."""
    if total_usd <= 0:
        raise SystemExit('total_usd harus > 0')
    if chunk_usd <= 0:
        raise SystemExit('chunk_usd harus > 0')

    from_token_addr = resolve_token(from_token, specific)
    to_token_addr = resolve_token(to_token, specific)
    sold = 0.0
    iteration = 0
    base_reason = (reason or 'batch sell').strip() or 'batch sell'

    while sold + 1e-9 < total_usd:
        remaining = total_usd - sold
        usd_chunk = min(chunk_usd, remaining)
        price_from = get_price(BASE, hdrs, from_token_addr, chain=chain, specific=specific)
        amount_from = trade_value_to_amount_usd(usd_chunk, price_from)
        iteration += 1
        print(f"Chunk {iteration}: sell ${usd_chunk:.2f} -> {amount_from:.6f} units")
        print(f"Progress: ${sold + usd_chunk:.2f} / ${total_usd:.2f}")
        execute_trade(
            BASE,
            hdrs,
            from_token_addr,
            to_token_addr,
            amount_from,
            reason=f"{base_reason} #{iteration}",
            chain=chain,
            specific=specific,
        )
        sold += usd_chunk

    print(f"Selesai batch sell: total ${sold:.2f} dalam {iteration} langkah.")




if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("RECALL_API_KEY")
    BASE = os.getenv("RECALL_API_URL", "https://api.sandbox.competitions.recall.network")
    if not API_KEY:
        raise SystemExit("RECALL_API_KEY belum ada di .env")

    parser = argparse.ArgumentParser(description="Trading Agent CLI (DEX multi-token)")
    parser.add_argument("mode", nargs="?", default="rebalance", choices=["rebalance", "manual", "batch-buy", "batch-sell"], help="Mode eksekusi")
    # rebalance params
    parser.add_argument("--target-pct", type=float, default=TARGET_WETH_PCT * 100, help="Target persentase dari total untuk target token")
    parser.add_argument("--max-trade-usd", type=float, default=MAX_TRADE_USD, help="Maks USD per trade")
    parser.add_argument("--reserve-usd", type=float, default=USDC_RESERVE_USD, help="Cadangan USD pada cash token")
    parser.add_argument("--min-trade-usd", type=float, default=MIN_TRADE_USD, help="Ambang minimal trade USD")
    parser.add_argument("--target-token", type=str, default="WETH", help="Symbol atau alamat target token (default WETH)")
    parser.add_argument("--cash-token", type=str, default="USDC", help="Symbol atau alamat cash token (default USDC)")
    parser.add_argument("--chain", type=str, default="evm", help="Chain (default evm)")
    parser.add_argument("--specific-chain", type=str, default="eth", help="Specific chain (default eth)")
    parser.add_argument("--use-llm", action="store_true", help="Gunakan OpenAI untuk menyarankan parameter rebalance")
    # manual params
    parser.add_argument("--side", choices=["buy", "sell"], help="Side manual trade")
    parser.add_argument("--from-token", type=str, help="Dari token (symbol/alamat)")
    parser.add_argument("--to-token", type=str, help="Ke token (symbol/alamat)")
    parser.add_argument("--amount", type=float, help="Jumlah dalam human units dari from-token")
    parser.add_argument("--amount-usd", type=float, help="Alternatif: nominal USD untuk dikonversi ke human units dari from-token")
    parser.add_argument("--reason", type=str, default="manual trade", help="Alasan trade")
    parser.add_argument("--total-usd", type=float, help="Total USD untuk batch trade")
    parser.add_argument("--chunk-usd", type=float, default=1.0, help="Nominal USD per chunk batch trade (default 1)")

    args = parser.parse_args()

    hdrs = make_headers(API_KEY)
    print("BASE_URL =", BASE)
    print("API_KEY  =", API_KEY[:6] + "***")

    if args.mode == "rebalance":
        target_pct = float(args.target_pct) / 100.0
        target_token = args.target_token
        cash_token = args.cash_token
        if args.use_llm:
            if not llm_mod:
                raise SystemExit("Modul LLM tidak tersedia. Pastikan file llm.py dan library openai terpasang.")
            # Fetch balances first for LLM
            balances = get_balances(BASE, hdrs)
            try:
                suggestion = llm_mod.suggest_rebalance_with_llm(
                    balances, default_target=target_token, default_cash=cash_token
                )
                target_pct = float(suggestion.get("targetPct", args.target_pct)) / (100.0 if suggestion.get("targetPct", 0) > 1 else 1.0)
                target_token = suggestion.get("targetToken", target_token)
                cash_token = suggestion.get("cashToken", cash_token)
                print("LLM suggestion:", suggestion)
            except Exception as e:
                print("Gagal mendapatkan saran LLM:", e)

        run_rebalance(
            BASE,
            hdrs,
            target_pct=target_pct,
            max_trade_usd=args.max_trade_usd,
            reserve_usd=args.reserve_usd,
            min_trade_usd=args.min_trade_usd,
            target_token=target_token,
            cash_token=cash_token,
            chain=args.chain,
            specific=args.specific_chain,
        )
    elif args.mode == "manual":
        if args.side is None or args.from_token is None or args.to_token is None:
            raise SystemExit("Untuk mode manual, wajib set --side, --from-token, --to-token dan --amount/--amount-usd")
        run_manual(
            BASE,
            hdrs,
            side=args.side,
            from_token=args.from_token,
            to_token=args.to_token,
            amount_human=args.amount,
            amount_usd=args.amount_usd,
            reason=args.reason,
            chain=args.chain,
            specific=args.specific_chain,
        )
    elif args.mode == "batch-buy":
        if args.to_token is None:
            raise SystemExit("Mode batch-buy perlu --to-token untuk token yang dibeli")
        total_usd = args.total_usd if args.total_usd is not None else args.amount_usd
        if total_usd is None or total_usd <= 0:
            raise SystemExit("Mode batch-buy perlu --total-usd (>0)")
        from_token = args.from_token or "USDC"
        chunk_usd = args.chunk_usd or 1.0
        reason = args.reason
        if reason == "manual trade":
            reason = "batch buy"
        run_batch_buy(
            BASE,
            hdrs,
            total_usd=total_usd,
            chunk_usd=chunk_usd,
            from_token=from_token,
            to_token=args.to_token,
            chain=args.chain,
            specific=args.specific_chain,
            reason=reason,
        )
    else:
        total_usd = args.total_usd if args.total_usd is not None else args.amount_usd
        if total_usd is None or total_usd <= 0:
            raise SystemExit("Mode batch-sell perlu --total-usd (>0)")
        from_token = args.from_token or "WETH"
        to_token = args.to_token or "USDC"
        chunk_usd = args.chunk_usd or 1.0
        reason = args.reason
        if reason == "manual trade":
            reason = "batch sell"
        run_batch_sell(
            BASE,
            hdrs,
            total_usd=total_usd,
            chunk_usd=chunk_usd,
            from_token=from_token,
            to_token=to_token,
            chain=args.chain,
            specific=args.specific_chain,
            reason=reason,
        )
