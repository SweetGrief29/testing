async function loadTrades() {
// Trade History

checkAIStatus();
// check AI readiness on load

}
  }
    el.classList.add("text-danger");
    el.classList.remove("text-success");
    el.textContent = "AI check error: " + e.message;
  } catch (e) {
    }
      el.classList.add("text-danger");
      el.classList.remove("text-success");
      el.textContent = "AI module not available";
    } else {
      el.classList.add("text-danger");
      el.classList.remove("text-success");
      el.textContent = "AI not configured (missing OPENAI_API_KEY)";
    } else if (!data.hasKey) {
      el.classList.add("text-success");
      el.classList.remove("text-danger");
      el.textContent = `AI ready (${data.model})`;
    if (data.available && data.hasKey) {
    if (data.error) throw new Error(data.error);
    const data = await res.json();
    const res = await fetch("/api/ai/status?test=1");
  try {
  el.textContent = "Checking AI...";
  if (!el) return;
  const el = document.getElementById("aiStatus");
async function checkAIStatus() {

}
  }
    out.textContent = "AI Error: " + e.message;
  } catch (e) {
    out.textContent = JSON.stringify(data, null, 2);
    }
      document.getElementById("rebCashToken").value = data.cashToken;
    if (data.cashToken) {
    }
      document.getElementById("rebTargetToken").value = data.targetToken;
    if (data.targetToken) {
    }
      document.getElementById("targetPct").value = Number(data.targetPct);
    if (data.targetPct) {
    if (data.error) throw new Error(data.error);
    const data = await res.json();
    const res = await fetch("/api/ai/suggest-rebalance");
  try {
  out.textContent = "AI analyzing...";
  const out = document.getElementById("rebOut");
async function aiSuggest() {
// AI Suggestion for rebalance

loadPnl();
loadTrades();
loadTokens();
loadBalances();
// initial load

document.getElementById("btnPnlRefresh").addEventListener("click", loadPnl);
document.getElementById("btnTradesClear").addEventListener("click", clearTrades);
document.getElementById("btnTradesRefresh").addEventListener("click", loadTrades);
document.getElementById("btnReloadTokens").addEventListener("click", loadTokens);
document.getElementById("btnAddToken").addEventListener("click", addToken);
document.getElementById("btnExecute").addEventListener("click", manualTrade);
document.getElementById("btnAISuggest").addEventListener("click", aiSuggest);
document.getElementById("btnRebalance").addEventListener("click", rebalance);
document.getElementById("btnRefresh").addEventListener("click", loadBalances);

}
  }
    out.textContent = "Error: " + e.message;
  } catch (e) {
    loadTrades();
    loadBalances();
    out.textContent = txt;
    const txt = await res.text();
    });
      body: JSON.stringify({ side, amountUsd: amount, reason, fromToken, toToken }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    const res = await fetch("/api/manual-trade", {
  try {
  out.textContent = "Processing...";
  const out = document.getElementById("tradeOut");
  const reason = document.getElementById("reason").value || "manual trade";
  const toToken = document.getElementById("manToToken").value;
  const fromToken = document.getElementById("manFromToken").value;
  const amount = Number(document.getElementById("amountUsd").value || 0);
  const side = document.getElementById("side").value;
async function manualTrade() {

}
  }
    out.textContent = "Error: " + e.message;
  } catch (e) {
    loadTrades();
    loadBalances();
    out.textContent = txt;
    const txt = await res.text();
    });
      }),
        cashToken,
        targetToken,
        reserveUsd: reserve,
        maxTradeUsd: maxTrade,
        targetPct,
      body: JSON.stringify({
      headers: { "Content-Type": "application/json" },
      method: "POST",
    const res = await fetch("/api/rebalance", {
  try {
  out.textContent = "Processing...";
  const out = document.getElementById("rebOut");
  const cashToken = document.getElementById("rebCashToken").value;
  const targetToken = document.getElementById("rebTargetToken").value;
  const reserve = Number(document.getElementById("reserve").value || 50);
  const maxTrade = Number(document.getElementById("maxTrade").value || 500);
  const targetPct = Number(document.getElementById("targetPct").value || 10);
async function rebalance() {

}
  }
    out.textContent = "Error: " + e.message;
  } catch (e) {
    loadTokens();
    out.textContent = txt;
    const txt = await res.text();
    });
      body: JSON.stringify({ symbol: sym, address: addr, specificChain: specific }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    const res = await fetch("/api/tokens", {
  try {
  out.textContent = "Processing...";
  const out = document.getElementById("tokensOut");
  const specific = document.getElementById("tokSpecific").value || "eth";
  const addr = document.getElementById("tokAddress").value.trim();
  const sym = document.getElementById("tokSymbol").value.trim();
async function addToken() {

}
  }
    out.textContent = "Gagal memuat tokens: " + e.message;
  } catch (e) {
    out.textContent = `Loaded ${TOKENS.length} tokens.`;

    if (weth) manTo.value = weth;
    if (usdc) manFrom.value = usdc;
    if (usdc) rebCash.value = usdc;
    if (weth) rebTarget.value = weth;
    const weth = findAddr("WETH");
    const usdc = findAddr("USDC");
    };
      return t ? t.address : "";
      const t = TOKENS.find((x) => (x.symbol || "").toUpperCase() === sym);
    const findAddr = (sym) => {
    // Defaults
    manTo.innerHTML = opts;
    manFrom.innerHTML = opts;
    rebCash.innerHTML = opts;
    rebTarget.innerHTML = opts;
    const manTo = document.getElementById("manToToken");
    const manFrom = document.getElementById("manFromToken");
    const rebCash = document.getElementById("rebCashToken");
    const rebTarget = document.getElementById("rebTargetToken");
    ).join("");
      (t) => `<option value="${t.address}" data-specific="${t.specificChain || "eth"}">${t.symbol} (${t.specificChain || "eth"})</option>`
    const opts = TOKENS.map(
    // Populate selects
    TOKENS = data.tokens || [];
    if (data.error) throw new Error(data.error);
    const data = await res.json();
    const res = await fetch("/api/tokens");
  try {
  out.textContent = "";
  const out = document.getElementById("tokensOut");
async function loadTokens() {

let TOKENS = [];

}
  }
    alertBox.classList.remove("d-none");
    alertBox.textContent = "Gagal memuat balances: " + e.message;
  } catch (e) {
    totalCell.textContent = "$" + total.toFixed(2);
    });
      tbody.appendChild(tr);
      `;
        }</td>
          b.value != null ? "$" + Number(b.value).toFixed(2) : ""
        <td class="text-end">${
        }</td>
          b.price != null ? "$" + Number(b.price).toFixed(4) : ""
        <td class="text-end">${
        <td class="text-end">${(b.amount || 0).toLocaleString()}</td>
        <td>${b.specificChain || ""}</td>
        <td>${b.chain || ""}</td>
        <td>${b.symbol || ""}</td>
      tr.innerHTML = `
      const tr = document.createElement("tr");
      total += b.value || 0;
    (data.balances || []).forEach((b) => {
    let total = 0;

    if (data.error) throw new Error(data.error);
    const data = await res.json();
    const res = await fetch("/api/balances");
  try {
  alertBox.classList.add("d-none");
  tbody.innerHTML = "";
  const alertBox = document.getElementById("balancesAlert");
  const totalCell = document.getElementById("totalCell");
  const tbody = document.getElementById("balancesBody");
async function loadBalances() {
}
