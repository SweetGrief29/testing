// =======================
// Global token registry & chains
// =======================
let TOKENS = [];
const CHAIN_SPECS = {
  evm: ["eth", "polygon", "base"],
  solana: ["sol"],
  // Disable specific for SVM: empty list
  svm: [],
};

// ===== Trade History Pagination =====
let TRADES = []; // semua data trade dari server
let tradesPage = 1; // halaman aktif
const TRADES_PER_PAGE = 10; // 10 baris per halaman

// =======================
// Helpers
// =======================
function setSpecificOptions(selectEl, chain) {
  const specs = CHAIN_SPECS[chain] || [];
  const cur = selectEl.value;
  selectEl.innerHTML = specs
    .map((s) => `<option value="${s}">${s}</option>`)
    .join("");
  if (specs.includes(cur)) selectEl.value = cur;
}

function tokenOptionsFor(chain, specific) {
  const hasSpecifics = (CHAIN_SPECS[chain] || []).length > 0;
  const list = (TOKENS || []).filter((t) => {
    const sameChain = String(t.chain || "") === String(chain || "");
    if (!sameChain) return false;
    if (!hasSpecifics || !specific) return true; // SVM or unspecified => ignore specific
    return String(t.specificChain || "") === String(specific || "");
  });
  return list
    .map(
      (t) =>
        `<option value="${t.address}" data-specific="${
          t.specificChain || ""
        }">${t.symbol} (${t.specificChain || ""})</option>`
    )
    .join("");
}

function symbolOf(addr) {
  try {
    const a = String(addr || "").toLowerCase();
    const t = (TOKENS || []).find(
      (x) => String(x.address || "").toLowerCase() === a
    );
    if (t && t.symbol) return t.symbol;
    if (a) return a.slice(0, 6) + "…" + a.slice(-4);
    return "";
  } catch (_) {
    return "";
  }
}

// formatters
const fmtUSD = (n) =>
  "$" + Number(n || 0).toLocaleString("en-US", { maximumFractionDigits: 2 });
const fmtAmt = (n) =>
  Number(n || 0).toLocaleString("en-US", { maximumFractionDigits: 6 });

// =======================
// Balances table
// =======================
async function loadBalances() {
  const tbody = document.getElementById("balancesBody");
  const totalCell = document.getElementById("totalCell");
  const alertBox = document.getElementById("balancesAlert");
  if (!tbody) return;
  tbody.innerHTML = "";
  alertBox && alertBox.classList.add("d-none");
  try {
    const res = await fetch("/api/balances");
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    let total = 0;
    (data.balances || []).forEach((b) => {
      total += b.value || 0;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${b.symbol || ""}</td>
        <td>${b.chain || ""}</td>
        <td>${b.specificChain || ""}</td>
        <td class="text-end">${(b.amount || 0).toLocaleString()}</td>
        <td class="text-end">${
          b.price != null ? "$" + Number(b.price).toFixed(4) : ""
        }</td>
        <td class="text-end">${
          b.value != null ? "$" + Number(b.value).toFixed(2) : ""
        }</td>
      `;
      tbody.appendChild(tr);
    });
    if (totalCell) totalCell.textContent = "$" + total.toFixed(2);
  } catch (e) {
    if (alertBox) {
      alertBox.textContent = "Gagal memuat balances: " + e.message;
      alertBox.classList.remove("d-none");
    }
  }
}

// =======================
// Tokens registry
// =======================
async function loadTokens() {
  const out = document.getElementById("tokensOut");
  if (out) out.textContent = "";
  try {
    const res = await fetch("/api/tokens");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    TOKENS = data.tokens || [];

    // Rebalance selectors
    const netChainReb = document.getElementById("netChainReb");
    const netSpecReb = document.getElementById("netSpecificReb");
    const rebTarget = document.getElementById("rebTargetToken");
    const rebCash = document.getElementById("rebCashToken");

    // Manual trade selectors
    const netChainMan = document.getElementById("netChainMan");
    const netSpecMan = document.getElementById("netSpecificMan");
    const netChainManTo = document.getElementById("netChainManTo");
    const netSpecManTo = document.getElementById("netSpecificManTo");
    const manFrom = document.getElementById("manFromToken");
    const manTo = document.getElementById("manToToken");

    if (netChainReb && netSpecReb) {
      const opts = tokenOptionsFor(netChainReb.value, netSpecReb.value);
      if (rebTarget) rebTarget.innerHTML = opts;
      if (rebCash) rebCash.innerHTML = opts;

      const pickBySym = (sym) => {
        const hasSpecs = (CHAIN_SPECS[netChainReb.value] || []).length > 0;
        const t = (TOKENS || []).find((x) => {
          const sameChain = String(x.chain || "") === netChainReb.value;
          if (!sameChain) return false;
          const symOk = String(x.symbol || "").toUpperCase() === String(sym || "").toUpperCase();
          if (!symOk) return false;
          if (!hasSpecs || !netSpecReb.value) return true;
          return String(x.specificChain || "") === netSpecReb.value;
        });
        return t ? t.address : "";
      };
      const isSolLike =
        netChainReb.value === "solana" || netChainReb.value === "svm";
      const defCash =
        pickBySym("USDC") || pickBySym("DAI") || pickBySym("USDT");
      const defTarg = pickBySym(isSolLike ? "wSOL" : "WETH");
      if (rebCash && defCash) rebCash.value = defCash;
      if (rebTarget && defTarg) rebTarget.value = defTarg;
    }

    if (netChainMan && netSpecMan) {
      const optsFrom = tokenOptionsFor(netChainMan.value, netSpecMan.value);
      if (manFrom) manFrom.innerHTML = optsFrom;
      const toChainVal = netChainManTo?.value || netChainMan?.value;
      const toSpecVal = netSpecManTo?.value || "";
      let optsTo = tokenOptionsFor(toChainVal, toSpecVal);
      if (!optsTo || optsTo.length === 0) {
        // Fallback: ignore specific and list by chain only (for SVM or empty lists)
        optsTo = tokenOptionsFor(toChainVal, "");
      }
      if (manTo) manTo.innerHTML = optsTo;

      const pickBySymM = (sym) => {
        const t = (TOKENS || []).find(
          (x) =>
            String(x.chain || "") === netChainMan.value &&
            String(x.specificChain || "") === netSpecMan.value &&
            String(x.symbol || "").toUpperCase() ===
              String(sym || "").toUpperCase()
        );
        return t ? t.address : "";
      };
      const pickBySymTo = (sym) => {
        const hasSpecsTo = (CHAIN_SPECS[toChainVal] || []).length > 0;
        const t = (TOKENS || []).find((x) => {
          const sameChain = String(x.chain || "") === String(toChainVal || "");
          if (!sameChain) return false;
          const symOk = String(x.symbol || "").toUpperCase() === String(sym || "").toUpperCase();
          if (!symOk) return false;
          if (!hasSpecsTo || !toSpecVal) return true;
          return String(x.specificChain || "") === String(toSpecVal || "");
        });
        return t ? t.address : "";
      };
      const isSolLikeM =
        netChainMan.value === "solana" || netChainMan.value === "svm";
      const defCashM =
        pickBySymM(isSolLikeM ? "wSOL" : "USDC") ||
        pickBySymM("DAI") ||
        pickBySymM("USDT");
      const defTargM = pickBySymTo(
        (toChainVal === "solana" || toChainVal === "svm") ? "USDC" : "WETH"
      );
      if (manFrom && defCashM) manFrom.value = defCashM;
      if (manTo && defTargM) manTo.value = defTargM;
    }

    if (out) out.textContent = `Loaded ${TOKENS.length} tokens.`;
    return TOKENS;
  } catch (e) {
    if (out) out.textContent = "Gagal memuat tokens: " + e.message;
    return [];
  }
}

async function addToken() {
  const sym = document.getElementById("tokSymbol").value.trim();
  const addr = document.getElementById("tokAddress").value.trim();
  const chain = document.getElementById("tokChain").value || "evm";
  let specific = document.getElementById("tokSpecific").value;
  if (!specific) {
    if (chain === "svm") specific = "svm"; else {
      const defs = CHAIN_SPECS[chain] || [];
      specific = defs.length ? defs[0] : "";
    }
  }
  const out = document.getElementById("tokensOut");
  if (out) out.textContent = "Processing...";
  try {
    const res = await fetch("/api/tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: sym,
        address: addr,
        chain,
        specificChain: specific,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error)
      throw new Error(data.error || "Add token failed");
    await loadTokens();
    const rebTarget = document.getElementById("rebTargetToken");
    const manTo = document.getElementById("manToToken");
    if (rebTarget) rebTarget.value = addr;
    if (manTo) manTo.value = addr;
    if (out) out.textContent = `Added ${sym} (${specific}).`;
  } catch (e) {
    if (out) out.textContent = "Error: " + e.message;
  }
}

async function removeToken() {
  const sym = document.getElementById("tokSymbol").value.trim();
  const addr = document.getElementById("tokAddress").value.trim();
  const chain = document.getElementById("tokChain").value || "evm";
  let specific = document.getElementById("tokSpecific").value;
  if (!specific) {
    if (chain === "svm") specific = "svm"; else {
      const defs = CHAIN_SPECS[chain] || [];
      specific = defs.length ? defs[0] : "";
    }
  }
  const out = document.getElementById("tokensOut");
  if (out) out.textContent = "Removing...";
  try {
    const payload = addr
      ? { address: addr, chain, specificChain: specific }
      : { symbol: sym, chain, specificChain: specific };
    const res = await fetch("/api/tokens/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error)
      throw new Error(data.error || "Remove token failed");
    await loadTokens();
    if (out) out.textContent = "Removed.";
  } catch (e) {
    if (out) out.textContent = "Error: " + e.message;
  }
}

// =======================
// Rebalance
// =======================
async function rebalance() {
  const targetPct = Number(document.getElementById("targetPct").value || 10);
  const maxTrade = Number(document.getElementById("maxTrade").value || 500);
  const reserve = Number(document.getElementById("reserve").value || 50);
  const targetToken = document.getElementById("rebTargetToken").value;
  const cashToken = document.getElementById("rebCashToken").value;
  const allowSell = !!document.getElementById("allowSell")?.checked;
  const chain = document.getElementById("netChainReb")?.value || "evm";
  const specificChain =
    document.getElementById("netSpecificReb")?.value || "eth";
  const out = document.getElementById("rebOut");
  if (out) out.textContent = "Processing...";
  try {
    const res = await fetch("/api/rebalance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        targetPct,
        maxTradeUsd: maxTrade,
        reserveUsd: reserve,
        targetToken,
        cashToken,
        allowSell,
        chain,
        specificChain,
      }),
    });
    const txt = await res.text();
    if (out) out.textContent = txt;
    loadBalances();
    loadTrades();
    loadPnl();
  } catch (e) {
    if (out) out.textContent = "Error: " + e.message;
  }
}

// =======================
// Manual trade
// =======================
async function manualTrade() {
  const side = document.getElementById("side").value;
  const amount = Number(document.getElementById("amountUsd").value || 0);
  const fromToken = document.getElementById("manFromToken").value;
  const toToken = document.getElementById("manToToken").value;
  const chain = document.getElementById("netChainMan")?.value || "evm";
  const specificChain =
    document.getElementById("netSpecificMan")?.value || "eth";
  const toChain = document.getElementById("netChainManTo")?.value || chain;
  const toSpecificChain =
    document.getElementById("netSpecificManTo")?.value || specificChain;
  const reason = document.getElementById("reason").value || "manual trade";
  const out = document.getElementById("tradeOut");
  if (out) out.textContent = "Processing...";
  try {
    // Build payload
    const payload = {
      side,
      amountUsd: amount,
      reason,
      fromToken,
      toToken,
      chain,
      specificChain,
      toChain,
      toSpecificChain,
    };

    const res = await fetch("/api/manual-trade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const txt = await res.text();
    if (out) out.textContent = txt;

    await loadBalances();
    await loadTrades();
    await loadPnl();

    // tampilkan saldo token yang relevan
    try {
      const res2 = await fetch("/api/balances");
      const data2 = await res2.json();
      const list = data2.balances || [];
      const focusAddr = side === "buy" ? toToken : fromToken;
      const hit = list.find(
        (b) =>
          String(b.tokenAddress || "").toLowerCase() ===
          String(focusAddr || "").toLowerCase()
      );
      if (hit) {
        const sym = hit.symbol || symbolOf(hit.tokenAddress);
        const line = `\nUpdated balance ${sym}: ${Number(
          hit.amount || 0
        ).toLocaleString()} (USD ${
          hit.value != null ? Number(hit.value).toFixed(2) : "0.00"
        })`;
        if (out) out.textContent += line;
      }
    } catch (_) {}
  } catch (e) {
    if (out) out.textContent = "Error: " + e.message;
  }
}

// =======================
// Trade History (with pagination)
// =======================
async function loadTrades() {
  const tbody = document.getElementById("tradesBody");
  const note = document.getElementById("tradesNote");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (note) note.textContent = "";
  try {
    // ambil banyak lalu paginasi di client
    const res = await fetch("/api/trades?limit=100");
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    TRADES = (data.trades || []).sort((a, b) => {
      const tb =
        Date.parse(b.serverTime || b.timestamp || b.createdAt || 0) || 0;
      const ta =
        Date.parse(a.serverTime || a.timestamp || a.createdAt || 0) || 0;
      return tb - ta; // newest first
    });

    tradesPage = 1; // reset ke halaman 1
    renderTrades();
  } catch (e) {
    if (note) note.textContent = "Error loading trades: " + e.message;
  }
}

function renderTrades() {
  const tbody = document.getElementById("tradesBody");
  const note = document.getElementById("tradesNote");
  const pageInfo = document.getElementById("tradesPageInfo"); // opsional (kalau ada di HTML)

  const total = TRADES.length;
  const start = (tradesPage - 1) * TRADES_PER_PAGE;
  const end = Math.min(start + TRADES_PER_PAGE, total);
  const slice = TRADES.slice(start, end);

  const rows = slice
    .map((t) => {
      const amount = t.amountFrom ?? t.amountHuman ?? "";
      const usd = t.tradeUsd ?? t.amountUsd ?? "";
      const fromSym = symbolOf(t.fromToken);
      const toSym = symbolOf(t.toToken);
      const timeStr = (t.serverTime || "").replace("T", " ").replace("Z", "");
      const sideStr = (t.side || "").toLowerCase();
      return `
      <tr>
        <td>${timeStr}</td>
        <td>${t.type || ""}</td>
        <td>${sideStr}</td>
        <td>${fromSym || ""} <span class="text-muted">${(
        t.fromToken || ""
      ).slice(0, 10)}...</span></td>
        <td>${toSym || ""} <span class="text-muted">${(t.toToken || "").slice(
        0,
        10
      )}...</span></td>
        <td class="text-end">${amount !== "" ? fmtAmt(amount) : ""}</td>
        <td class="text-end">${usd !== "" ? fmtUSD(usd) : ""}</td>
        <td>${t.status || (t.success ? "ok" : "error")}</td>
      </tr>`;
    })
    .join("");

  tbody.innerHTML = rows;
  if (pageInfo) {
    pageInfo.textContent = `${total ? start + 1 : 0}–${end} of ${total}`;
  } else if (note) {
    note.textContent = `Showing ${total ? start + 1 : 0}–${end} of ${total}`;
  }

  // enable/disable tombol pager jika ada
  const prev = document.getElementById("tradesPrev");
  const next = document.getElementById("tradesNext");
  if (prev) prev.disabled = tradesPage === 1;
  if (next) next.disabled = end >= total;
}

async function clearTrades() {
  const note = document.getElementById("tradesNote");
  try {
    const res = await fetch("/api/trades/clear", { method: "POST" });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    await loadTrades();
    if (note) note.textContent = "Cleared.";
  } catch (e) {
    if (note) note.textContent = "Clear failed: " + e.message;
  }
}

// =======================
// PnL
// =======================
async function loadPnl() {
  const posBody = document.getElementById("pnlPositions");
  const rlzBody = document.getElementById("pnlRealized");
  const totalVal = document.getElementById("pnlTotalValue");
  const totalUnrl = document.getElementById("pnlTotalUnreal");
  const totalRlz = document.getElementById("pnlTotalRealized");
  if (!posBody || !rlzBody) return;
  posBody.innerHTML = "";
  rlzBody.innerHTML = "";
  try {
    const res = await fetch("/api/pnl");
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const positions = data.positions || {};
    Object.values(positions).forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${p.symbol || ""}</td>
        <td class="text-end">${Number(p.amount || 0).toLocaleString()}</td>
        <td class="text-end">${
          p.avgCostPerUnitUsd
            ? "$" + Number(p.avgCostPerUnitUsd).toFixed(4)
            : ""
        }</td>
        <td class="text-end">${
          p.marketPriceUsd ? "$" + Number(p.marketPriceUsd).toFixed(4) : ""
        }</td>
        <td class="text-end">${
          p.marketValueUsd ? "$" + Number(p.marketValueUsd).toFixed(2) : "$0.00"
        }</td>
        <td class="text-end">${
          p.unrealizedUsd ? "$" + Number(p.unrealizedUsd).toFixed(2) : "$0.00"
        }</td>
      `;
      posBody.appendChild(tr);
    });

    const realized = data.realized || [];
    realized.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${(r.time || "").replace("T", " ").replace("Z", "")}</td>
        <td>${r.token || ""}</td>
        <td class="text-end">${Number(r.amount || 0).toLocaleString()}</td>
        <td class="text-end">${
          r.proceedsUsd != null ? "$" + Number(r.proceedsUsd).toFixed(2) : ""
        }</td>
        <td class="text-end">${
          r.costUsd != null ? "$" + Number(r.costUsd).toFixed(2) : ""
        }</td>
        <td class="text-end">${
          r.pnlUsd != null ? "$" + Number(r.pnlUsd).toFixed(2) : ""
        }</td>
      `;
      rlzBody.appendChild(tr);
    });

    if (totalVal)
      totalVal.textContent =
        "$" + Number((data.stats || {}).marketValueUsd || 0).toFixed(2);
    if (totalUnrl)
      totalUnrl.textContent =
        "$" + Number((data.stats || {}).unrealizedUsd || 0).toFixed(2);
    if (totalRlz)
      totalRlz.textContent =
        "$" + Number((data.stats || {}).realizedUsd || 0).toFixed(2);
  } catch (_) {
    // silent
  }
}

// =======================
// PnL Live updater
// =======================
let PNL_TIMER = null;
let PNL_LOADING = false;

function startPnlLive() {
  const secInput = document.getElementById("pnlLiveSec");
  let sec = Number(secInput?.value || 15);
  if (!Number.isFinite(sec) || sec < 5) sec = 15;
  stopPnlLive();
  PNL_TIMER = setInterval(async () => {
    if (PNL_LOADING) return;
    try {
      PNL_LOADING = true;
      await loadPnl();
    } finally {
      PNL_LOADING = false;
    }
  }, sec * 1000);
}

function stopPnlLive() {
  if (PNL_TIMER) {
    clearInterval(PNL_TIMER);
    PNL_TIMER = null;
  }
}

function initPnlLive() {
  const toggle = document.getElementById("pnlLiveToggle");
  const secInput = document.getElementById("pnlLiveSec");
  if (toggle) {
    toggle.addEventListener("change", () => {
      if (toggle.checked) startPnlLive();
      else stopPnlLive();
    });
  }
  if (secInput) {
    secInput.addEventListener("change", () => {
      if (toggle?.checked) startPnlLive();
    });
  }
  // Pause when tab hidden
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPnlLive();
    else if (toggle?.checked) startPnlLive();
  });
  // Start by default if toggle is on
  if (toggle?.checked) startPnlLive();
}

// =======================
// AI Suggestion / Status
// =======================
async function aiSuggest() {
  const out = document.getElementById("rebOut");
  if (out) out.textContent = "AI analyzing...";
  try {
    const res = await fetch("/api/ai/suggest-rebalance");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (data.targetPct)
      document.getElementById("targetPct").value = Number(data.targetPct);
    if (data.targetToken)
      document.getElementById("rebTargetToken").value = data.targetToken;
    if (data.cashToken)
      document.getElementById("rebCashToken").value = data.cashToken;
    if (out) out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    if (out) out.textContent = "AI Error: " + e.message;
  }
}

async function checkAIStatus() {
  const el = document.getElementById("aiStatus");
  if (!el) return;
  el.textContent = "Checking AI...";
  try {
    const res = await fetch("/api/ai/status?test=1");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (data.available && data.hasKey) {
      el.textContent = `AI ready (${data.model})`;
      el.classList.remove("text-danger");
      el.classList.add("text-success");
    } else if (!data.hasKey) {
      el.textContent = "AI not configured (missing OPENAI_API_KEY)";
      el.classList.remove("text-success");
      el.classList.add("text-danger");
    } else {
      el.textContent = "AI module not available";
      el.classList.remove("text-success");
      el.classList.add("text-danger");
    }
  } catch (e) {
    el.textContent = "AI check error: " + e.message;
    el.classList.remove("text-success");
    el.classList.add("text-danger");
  }
}

// =======================
// Event listeners
// =======================
document.getElementById("btnRefresh")?.addEventListener("click", loadBalances);
document.getElementById("btnRebalance")?.addEventListener("click", rebalance);
document.getElementById("btnAISuggest")?.addEventListener("click", aiSuggest);
document.getElementById("btnExecute")?.addEventListener("click", manualTrade);
document.getElementById("btnAddToken")?.addEventListener("click", addToken);
document
  .getElementById("btnRemoveToken")
  ?.addEventListener("click", removeToken);

document
  .getElementById("btnReloadTokens")
  ?.addEventListener("click", loadTokens);
document
  .getElementById("btnTradesRefresh")
  ?.addEventListener("click", loadTrades);
document
  .getElementById("btnTradesClear")
  ?.addEventListener("click", clearTrades);
document.getElementById("btnPnlRefresh")?.addEventListener("click", loadPnl);

// Pager buttons for Trade History
document.getElementById("tradesPrev")?.addEventListener("click", () => {
  if (tradesPage > 1) {
    tradesPage--;
    renderTrades();
  }
});
document.getElementById("tradesNext")?.addEventListener("click", () => {
  if (tradesPage * TRADES_PER_PAGE < TRADES.length) {
    tradesPage++;
    renderTrades();
  }
});

// Swap tokens (manual trade)
document.getElementById("btnSwap")?.addEventListener("click", () => {
  const fromSel = document.getElementById("manFromToken");
  const toSel = document.getElementById("manToToken");
  if (!fromSel || !toSel) return;
  const tmp = fromSel.value;
  fromSel.value = toSel.value;
  toSel.value = tmp;
});

// =======================
// Network selection handlers
// =======================
function initNetworkSelectors() {
  const netChainReb = document.getElementById("netChainReb");
  const netSpecReb = document.getElementById("netSpecificReb");
  const netChainMan = document.getElementById("netChainMan");
  const netSpecMan = document.getElementById("netSpecificMan");
  const netChainManTo = document.getElementById("netChainManTo");
  const netSpecManTo = document.getElementById("netSpecificManTo");
  const tokChain = document.getElementById("tokChain");
  const tokSpecific = document.getElementById("tokSpecific");
  const tokAddress = document.getElementById("tokAddress");

  if (netChainReb && netSpecReb) {
    setSpecificOptions(netSpecReb, netChainReb.value);
    netSpecReb.disabled = (CHAIN_SPECS[netChainReb.value] || []).length === 0;
    netChainReb.addEventListener("change", async () => {
      setSpecificOptions(netSpecReb, netChainReb.value);
      netSpecReb.disabled = (CHAIN_SPECS[netChainReb.value] || []).length === 0;
      await loadTokens();
      if (tokChain && tokSpecific) {
        tokChain.value = netChainReb.value;
        setSpecificOptions(tokSpecific, tokChain.value);
        if (tokAddress)
          tokAddress.placeholder =
            tokChain.value === "evm" ? "0x... (EVM)" : "base58 (Solana/SVM)";
      }
    });
    netSpecReb.addEventListener("change", loadTokens);
  }

  if (netChainMan && netSpecMan) {
    setSpecificOptions(netSpecMan, netChainMan.value);
    netSpecMan.disabled = (CHAIN_SPECS[netChainMan.value] || []).length === 0;
    netChainMan.addEventListener("change", async () => {
      setSpecificOptions(netSpecMan, netChainMan.value);
      netSpecMan.disabled = (CHAIN_SPECS[netChainMan.value] || []).length === 0;
      await loadTokens();
      if (tokChain && tokSpecific) {
        tokChain.value = netChainMan.value;
        setSpecificOptions(tokSpecific, tokChain.value);
        if (tokAddress)
          tokAddress.placeholder =
            tokChain.value === "evm" ? "0x... (EVM)" : "base58 (Solana/SVM)";
      }
    });
    netSpecMan.addEventListener("change", loadTokens);
  }

  if (netChainManTo && netSpecManTo) {
    setSpecificOptions(netSpecManTo, netChainManTo.value);
    netSpecManTo.disabled = (CHAIN_SPECS[netChainManTo.value] || []).length === 0;
    netChainManTo.addEventListener("change", async () => {
      setSpecificOptions(netSpecManTo, netChainManTo.value);
      netSpecManTo.disabled = (CHAIN_SPECS[netChainManTo.value] || []).length === 0;
      // For chains without specifics (svm), clear value; otherwise keep first
      const defaults = (CHAIN_SPECS[netChainManTo.value] || []);
      netSpecManTo.value = defaults[0] || "";
      // Repopulate To tokens immediately using chain-only if needed
      const manTo = document.getElementById("manToToken");
      if (manTo) {
        manTo.innerHTML = tokenOptionsFor(netChainManTo.value, netSpecManTo.value || "");
      }
      await loadTokens();
    });
    netSpecManTo.addEventListener("change", async () => {
      const manTo = document.getElementById("manToToken");
      if (manTo) {
        manTo.innerHTML = tokenOptionsFor(netChainManTo.value, netSpecManTo.value || "");
      }
      await loadTokens();
    });
  }

  // No separate destination chain for manual trade in UI

  if (tokChain && tokSpecific) {
    setSpecificOptions(tokSpecific, tokChain.value);
    tokSpecific.disabled = (CHAIN_SPECS[tokChain.value] || []).length === 0;
    // Manage Tokens: always allow selecting specific for SVM
    if (tokChain.value === "svm") {
      tokSpecific.disabled = false;
      tokSpecific.innerHTML = '<option value="svm">svm</option>';
      tokSpecific.value = "svm";
    } else if (tokSpecific.disabled) {
      tokSpecific.value = "";
    }
    tokChain.addEventListener("change", () =>
      { setSpecificOptions(tokSpecific, tokChain.value);
        tokSpecific.disabled = (CHAIN_SPECS[tokChain.value] || []).length === 0;
        if (tokChain.value === "svm") {
          tokSpecific.disabled = false;
          tokSpecific.innerHTML = '<option value="svm">svm</option>';
          tokSpecific.value = "svm";
        } else if (tokSpecific.disabled) {
          tokSpecific.value = "";
        } }
    );
    if (tokAddress)
      tokAddress.placeholder =
        tokChain.value === "evm" ? "0x... (EVM)" : "base58 (Solana/SVM)";
  }
}

// =======================
// Initial load
// =======================
initNetworkSelectors();
loadBalances();
loadTokens();
loadTrades();
loadPnl();
checkAIStatus();
initPnlLive();
