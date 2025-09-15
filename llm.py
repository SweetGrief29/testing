import json
import os
from typing import Any, Dict, List, Optional


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Library 'openai' belum terpasang. Install: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY belum ada di .env")
    base = os.getenv("OPENAI_BASE_URL")  # optional, for proxy/self-hosted
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    return OpenAI(api_key=api_key)


def suggest_rebalance_with_llm(balances: List[Dict[str, Any]],
                               default_target: Optional[str] = None,
                               default_cash: Optional[str] = None,
                               model: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns dict: { targetPct: float, targetToken: address, cashToken: address, reason: str }
    """
    client = _get_openai_client()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Keep balances compact to avoid token bloat
    brief = [
        {
            "symbol": b.get("symbol"),
            "tokenAddress": b.get("tokenAddress"),
            "chain": b.get("chain"),
            "specificChain": b.get("specificChain"),
            "amount": b.get("amount"),
            "price": b.get("price"),
            "value": b.get("value"),
        }
        for b in balances
    ]

    sys = (
        "Anda adalah asisten trading. Outputkan SINGKAT dalam JSON saja, tidak ada teks lain. "
        "Format: {\"targetPct\": number(0..100), \"targetToken\": address, \"cashToken\": address, \"reason\": string}. "
        "Pilih target token utama yang liquid (contoh WETH) dan cash token stable (USDC/DAI). "
        "Jika default disediakan, gunakan sebagai preferensi.")

    usr = {
        "balances": brief,
        "defaults": {"targetToken": default_target, "cashToken": default_cash},
        "instruction": "Sarankan targetPct yang wajar (5..20) dengan alasan singkat."}

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(usr)},
        ],
        temperature=0.3,
    )

    # Extract text
    out_text = ""
    try:
        out_text = resp.output_text  # SDK >= 1.51
    except Exception:
        try:
            out_text = resp.choices[0].message.content  # legacy
        except Exception:
            pass

    if not out_text:
        raise RuntimeError("LLM tidak mengembalikan teks")

    # Try parse JSON
    out_text = out_text.strip()
    if out_text.startswith("```"):
        # remove code fences if provided
        out_text = out_text.strip('`')
        parts = out_text.split("\n", 1)
        if len(parts) == 2:
            out_text = parts[1]

    try:
        data = json.loads(out_text)
    except Exception as e:
        raise RuntimeError(f"Gagal parse JSON dari LLM: {out_text[:200]}") from e

    # Basic validation
    for k in ("targetPct", "targetToken", "cashToken"):
        if k not in data:
            raise RuntimeError(f"Kunci '{k}' tidak ada pada keluaran LLM")
    return data

