(() => {
  "use strict";

  const PAGE_SIZE = 15;
  const DETECTORS = ["cloned_tag", "class_mismatch", "duplicate_charge", "fare_anomaly", "api_flood"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const state = {
    run: null, page: 1, total: 0, items: [], selected: null, token: 0, busy: false,
    metrics: null, tab: "alerts",
    filters: { severity: "", detector: "", plaza: "", status: "open", q: "" },
  };

  const $ = (id) => document.getElementById(id);

  // ---------- helpers ----------
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  const num = (n) => Number(n || 0).toLocaleString("en-US");
  const pct = (v) => (v == null ? "n/a" : (v * 100).toFixed(1) + "%");
  const sevClass = (s) => (s === "Critical" ? "crit" : s === "Warning" ? "warn" : "info");
  const riskColor = (s) => (s >= 80 ? "#DC2626" : "#D97706");

  function fmtTime(s) {
    const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/.exec(s || "");
    return m ? `${m[3]} ${MONTHS[Number(m[2]) - 1]} ${m[4]}:${m[5]}` : String(s || "");
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON body */ }
    if (!res.ok) {
      let msg = body && body.detail;
      if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return body;
  }

  function banner(message, kind) {
    const el = $("banner");
    if (!message) { el.hidden = true; return; }
    el.textContent = message;
    el.className = "banner " + (kind || "info");
    el.hidden = false;
  }

  function setBusy(on) {
    state.busy = on;
    $("btn-demo").disabled = on;
    $("btn-upload").disabled = on;
    $("btn-upload").textContent = on ? "Analyzing..." : "Upload CSV";
  }

  function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  // ---------- analysis ----------
  async function runAnalysis(call) {
    if (state.busy) return;
    setBusy(true);
    banner("");
    try {
      const run = await call();
      await loadRun(run);
      if (run.notes && run.notes.length) banner(run.notes.join(" "), "info");
    } catch (e) {
      banner(e.message, "error");
    } finally {
      setBusy(false);
    }
  }

  async function loadRun(run) {
    state.run = run;
    state.page = 1;
    state.selected = null;
    $("run-label").textContent = "Run: " + run.filename;
    renderStats();
    renderCharts();
    renderAnalytics();
    if (state.tab === "model") renderModel();
    await fetchAlerts();
  }

  async function refreshRun() {
    if (!state.run) return;
    try {
      state.run = Object.assign({}, state.run, await api("/api/runs/" + state.run.run_id));
      renderAnalytics();
    } catch (e) { /* non-critical */ }
  }

  async function fetchAlerts() {
    if (!state.run) return;
    const token = ++state.token;
    const p = new URLSearchParams({ page: state.page, page_size: PAGE_SIZE });
    for (const k of ["severity", "detector", "plaza", "status", "q"]) {
      if (state.filters[k]) p.set(k, state.filters[k]);
    }
    try {
      const data = await api(`/api/runs/${state.run.run_id}/alerts?${p}`);
      if (token !== state.token) return;           // a newer request superseded this one
      state.items = data.items;
      state.total = data.total;
      if (!state.items.some((a) => a.txn_id === state.selected)) {
        state.selected = state.items.length ? state.items[0].txn_id : null;
      }
      renderTable();
      renderDetail();
    } catch (e) {
      banner(e.message, "error");
    }
  }

  // ---------- rendering: alerts tab ----------
  function renderStats() {
    const s = state.run ? state.run.summary : null;
    const sev = (s && s.by_severity) || {};
    $("stat-total").textContent = s ? num(s.total_transactions) : "-";
    $("stat-flagged").textContent = s ? num(s.flagged_transactions) : "-";
    $("stat-flagged-sub").textContent = s ? (s.flag_rate * 100).toFixed(1) + "% of transactions" : "\u00a0";
    $("stat-critical").textContent = s ? num(sev.Critical) : "-";
    $("stat-warning").textContent = s ? num(sev.Warning) : "-";
  }

  function renderTable() {
    const body = $("rows");
    if (!state.items.length) {
      body.innerHTML = '<tr><td colspan="6" class="empty">No alerts match these filters.</td></tr>';
    } else {
      body.innerHTML = state.items.map((a) => {
        const chips = a.detected_type.split("+").map((t) => `<span class="tg">${esc(t)}</span>`).join("");
        const w = Math.round(a.risk_score * 0.6);
        return `<tr data-txn="${esc(a.txn_id)}" class="${a.txn_id === state.selected ? "sel" : ""}">
          <td><div class="bar"><b>${a.risk_score.toFixed(1)}</b><i style="width:${w}px;background:${riskColor(a.risk_score)}"></i></div></td>
          <td><button type="button" class="linkbtn" data-txn="${esc(a.txn_id)}">${esc(a.txn_id)}</button><div class="sub">${esc(a.tag_id)}</div></td>
          <td>${esc(a.plaza_name)}</td><td>${chips}</td><td>${esc(fmtTime(a.timestamp))}</td>
          <td><span class="st st-${esc(a.status)}">${esc(a.status)}</span></td></tr>`;
      }).join("");
    }
    const pages = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
    const start = state.total ? (state.page - 1) * PAGE_SIZE + 1 : 0;
    const end = Math.min(state.total, state.page * PAGE_SIZE);
    $("range").textContent = state.total ? `Showing ${num(start)}-${num(end)} of ${num(state.total)}` : "";
    $("pageinfo").textContent = state.total ? `Page ${state.page} of ${num(pages)}` : "";
    $("prev").disabled = state.page <= 1;
    $("next").disabled = state.page >= pages;
  }

  function renderDetail() {
    const box = $("detail");
    const a = state.items.find((x) => x.txn_id === state.selected);
    if (!a) { box.innerHTML = '<div class="empty">Select an alert to see details.</div>'; return; }
    box.innerHTML = `
      <div class="dhead"><div class="dtitle">${esc(a.txn_id)}</div><span class="pill ${sevClass(a.severity)}">${esc(a.severity)}</span></div>
      <div class="kpis">
        <div class="kpi"><div class="sub">Risk score</div><div class="num">${a.risk_score.toFixed(1)}</div></div>
        <div class="kpi"><div class="sub">Confidence</div><div class="num">${a.confidence.toFixed(1)}%</div></div>
      </div>
      <div><div class="sub">Why it was flagged</div><div class="reason">${esc(a.reason).split(" | ").join("<br>")}</div></div>
      <div class="kv">
        <span>Tag</span><span>${esc(a.tag_id)}</span>
        <span>Plaza</span><span>${esc(a.plaza_name)}</span>
        <span>Vehicle</span><span>${esc(a.declared_class)}, ${esc(a.axle_count)} axles</span>
        <span>Billed</span><span>INR ${esc(a.amount_INR)}</span>
        <span>Time</span><span>${esc(a.timestamp)}</span>
        <span>Status</span><span class="st st-${esc(a.status)}">${esc(a.status)}</span>
      </div>
      <label class="fl">Analyst note<input id="note" maxlength="500" placeholder="Optional note" value="${esc(a.note)}"></label>
      <div class="btnrow">
        <button class="btn" type="button" data-act="confirmed">Confirm</button>
        <button class="btn" type="button" data-act="dismissed">Dismiss</button>
        <button class="btn danger" type="button" data-act="blocked">Block tag</button>
      </div>`;
  }

  async function act(status) {
    const a = state.items.find((x) => x.txn_id === state.selected);
    if (!a || !state.run) return;
    const note = $("note") ? $("note").value.trim() : "";
    try {
      const r = await api(`/api/runs/${state.run.run_id}/alerts/${encodeURIComponent(a.txn_id)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, note }),
      });
      Object.assign(a, r.alert);
      if (status === "blocked") {
        banner(`Tag ${a.tag_id} blocked.` + (r.other_alerts_blocked
          ? ` ${r.other_alerts_blocked} other open alert(s) for this tag were blocked too.` : ""), "info");
        await loadBlocked();
        if (r.other_alerts_blocked) await fetchAlerts();
      } else {
        banner(`Alert ${a.txn_id} marked ${status}.`, "info");
      }
      renderTable();
      renderDetail();
      refreshRun();
    } catch (e) {
      banner(e.message, "error");
    }
  }

  // ---------- rendering: charts ----------
  function bars(obj) {
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) return '<div class="sub">No data</div>';
    const max = Math.max(1, ...entries.map((e) => e[1]));
    return entries.map(([k, v]) =>
      `<div class="hbar"><span class="lbl">${esc(k)}</span><span class="track"><i style="width:${(v / max * 100).toFixed(1)}%"></i></span><b>${num(v)}</b></div>`).join("");
  }

  function dayCols(obj) {
    const entries = Object.entries(obj || {});
    if (!entries.length) return '<div class="sub">No data</div>';
    const max = Math.max(1, ...entries.map((e) => e[1]));
    return '<div class="cols">' + entries.map(([d, v], i) =>
      `<div class="col" title="${esc(d)}: ${v}"><i style="height:${(v / max * 100).toFixed(1)}%"></i>${i % 5 === 0 ? `<span>${esc(d.slice(5))}</span>` : ""}</div>`).join("") + "</div>";
  }

  function renderCharts() {
    const s = state.run ? state.run.summary : {};
    $("chart-detector").innerHTML = bars(s.by_detector);
    $("chart-plaza").innerHTML = bars(Object.fromEntries(Object.entries(s.by_plaza || {}).slice(0, 5)));
  }

  function renderAnalytics() {
    const s = state.run ? state.run.summary : {};
    $("an-severity").innerHTML = bars(s.by_severity);
    $("an-status").innerHTML = bars(state.run ? state.run.status_counts : {});
    $("an-days").innerHTML = dayCols(s.by_day);
  }

  // ---------- model evidence: selection, fit diagnostics, controls ----------
  const f3 = (v) => (v == null ? "n/a" : Number(v).toFixed(3));
  const famName = (f) => String(f || "").replace(/_/g, " ");

  function lineChart(title, xs, series, opts = {}) {
    const W = 300, H = 150, L = 38, R = 10, T = 8, B = 26;
    const all = series.flatMap((s) => s.values).filter((v) => v != null);
    let lo = opts.min != null ? opts.min : Math.min(...all), hi = opts.max != null ? opts.max : Math.max(...all);
    if (hi - lo < 0.02) { if (opts.max != null) lo = hi - 0.02; else { lo -= 0.01; hi += 0.01; } }
    const x = (i) => L + (xs.length === 1 ? 0 : (i * (W - L - R)) / (xs.length - 1));
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const lines = series.map((s) => {
      const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      return `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${pts}"/>` +
        s.values.map((v, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" fill="${s.color}"/>`).join("");
    }).join("");
    const xl = xs.map((v, i) => `<text x="${x(i).toFixed(1)}" y="${H - 8}" font-size="9" text-anchor="middle" fill="#4B5563">${esc(v)}</text>`).join("");
    const mark = opts.mark == null ? "" : `<line x1="${x(opts.mark)}" x2="${x(opts.mark)}" y1="${T}" y2="${H - B}" stroke="#9CA3AF" stroke-dasharray="3 3"/>`;
    return `<div class="chartbox"><h4>${esc(title)}</h4>
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
        <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#E5E7EB"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#E5E7EB"/>
        <text x="${L - 4}" y="${T + 8}" font-size="9" text-anchor="end" fill="#4B5563">${hi.toFixed(2)}</text>
        <text x="${L - 4}" y="${H - B}" font-size="9" text-anchor="end" fill="#4B5563">${lo.toFixed(2)}</text>
        ${mark}${lines}${xl}</svg>
      <div class="legend">${series.map((s) => `<span><i class="dot" style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div></div>`;
  }

  function selectionCard(sel) {
    const blocks = Object.entries(sel.detectors).map(([name, d]) => {
      const rows = [...d.candidates].filter((c) => !c.error)
        .sort((a, b) => b.val_pr_auc - a.val_pr_auc).map((c) => {
          const isSel = c.id === d.selected.id;
          const kind = c.supervised ? "supervised reference" : c.anchor ? "low-capacity anchor" : "unsupervised";
          return `<tr class="${isSel ? "selrow" : ""}"><td>${isSel ? "<b>" : ""}${esc(c.id)}${isSel ? "</b> <span class=\"tg\">selected</span>" : ""}</td>
            <td class="sub">${kind}</td><td>${f3(c.val_pr_auc)} &plusmn; ${f3(c.val_pr_auc_sd)}</td><td>${f3(c.train_pr_auc)}</td><td>${(c.gap >= 0 ? "+" : "") + f3(c.gap)}</td><td>${c.fit_seconds}s</td></tr>`;
        }).join("");
      return `<div class="h3">${esc(name)}: headroom over the best supervised reference ${(d.headroom_vs_supervised_pr_auc >= 0 ? "+" : "") + f3(d.headroom_vs_supervised_pr_auc)} PR-AUC</div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Candidate</th><th>Type</th><th>Validation PR-AUC (4 folds)</th><th>Train PR-AUC</th><th>Gap</th><th>Fit time</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }).join("");
    return `<div class="card sect"><h2>Model selection</h2>
      <div class="note">Every candidate is fitted on everything before a time chunk and validated on that chunk, four times over, inside the fit window (the test window is never touched). The winner is the cheapest deployable model that is not significantly worse than the best. Only unsupervised models can be deployed, because real toll data has no labels until analysts confirm alerts. Supervised models are listed only to show how much headroom the label-free approach leaves. Density-based models (LOF, GMM, One-Class SVM) lose badly because the anomalies recur in the training data and get absorbed as a normal mode, while Isolation Forest&rsquo;s random subsampling resists that.</div>${blocks}</div>`;
  }

  const CHECK_LABEL = {
    train_test_gap: ["Train minus test gap (PR-AUC or F1)", "&le; 0.05"],
    fresh_world_f1: ["Test F1 minus fresh-world F1", "&le; 0.05"],
    headroom: ["Headroom vs supervised reference (PR-AUC)", "&le; 0.05"],
    beats_baselines: ["Test F1 minus best simple baseline", "&gt; 0"],
    capacity_saturated: ["Gain from more trees (PR-AUC)", "&le; 0.01"],
  };

  function diagnosticsCard(dg) {
    const blocks = Object.entries(dg.detectors).map(([name, e]) => {
      const d = e.data, good = e.verdict === "good fit";
      const checks = Object.entries(e.checks).map(([k, c]) => {
        const lab = CHECK_LABEL[k] || [k, ""];
        return `<tr><td>${lab[0]}</td><td>${(c.value >= 0 && k !== "beats_baselines" ? "" : "") + f3(c.value)}</td><td>${lab[1]}</td><td class="${c.pass ? "pass" : "fail"}">${c.pass ? "pass" : "FAIL"}</td></tr>`;
      }).join("");
      const tt = d.train_vs_test;
      const lc = d.learning_curve;
      const charts = [lineChart("Learning curve (PR-AUC vs share of fit data)", lc.map((r) => Math.round(r.fraction * 100) + "%"),
        [{ name: "train", color: "#1E3A8A", values: lc.map((r) => r.train_pr_auc) }, { name: "validation", color: "#D97706", values: lc.map((r) => r.val_pr_auc) }], { max: 1 })];
      Object.entries(d.capacity_curves || {}).forEach(([knob, c]) => {
        charts.push(lineChart(knob === "n_estimators" ? "Capacity: number of trees" : "Sensitivity: subsample size", c.curve.map((r) => r.value),
          [{ name: "train", color: "#1E3A8A", values: c.curve.map((r) => r.train_pr_auc) }, { name: "validation", color: "#D97706", values: c.curve.map((r) => r.val_pr_auc) }],
          { max: 1, mark: c.curve.findIndex((r) => r.value === c.chosen) }));
      });
      const ro = d.rolling_origin;
      charts.push(lineChart("Rolling-origin windows (test F1)", ro.folds.map((r) => "w" + (r.test_chunk - 3)),
        [{ name: "F1", color: "#166534", values: ro.folds.map((r) => r.f1) }], { max: 1 }));
      const rr = ro.folds.map((r) => `<tr><td>${esc(String(r.test_starts).slice(0, 10))}</td><td>${num(r.train_rows)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td></tr>`).join("");
      const ci = d.bootstrap_95ci;
      return `<div class="h3">${esc(name)} <span class="sub">${esc(famName(e.data.model))}</span><span class="verdict ${good ? "v-good" : "v-bad"}">${esc(e.verdict.toUpperCase())}</span></div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Check</th><th>Result</th><th>Limit</th><th>Status</th></tr></thead><tbody>${checks}</tbody></table></div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Window</th><th>PR-AUC</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
          <tr><td>Fit rows (train)</td><td>${f3(tt.train.pr_auc)}</td><td>${pct(tt.train.f1)}</td><td>${pct(tt.train.precision)}</td><td>${pct(tt.train.recall)}</td></tr>
          <tr><td>Selection folds (validation, mean)</td><td>${f3(tt.validation_folds.pr_auc_mean)} &plusmn; ${f3(tt.validation_folds.pr_auc_sd)}</td><td></td><td></td><td></td></tr>
          <tr><td>Test (looked at once)</td><td>${f3(tt.test.pr_auc)}</td><td>${pct(tt.test.f1)}</td><td>${pct(tt.test.precision)}</td><td>${pct(tt.test.recall)}</td></tr>
          <tr><td>Test, 95% bootstrap interval</td><td></td><td>${pct(ci.f1[0])} to ${pct(ci.f1[1])}</td><td>${pct(ci.precision[0])} to ${pct(ci.precision[1])}</td><td>${pct(ci.recall[0])} to ${pct(ci.recall[1])}</td></tr>
        </tbody></table></div>
        <div class="charts">${charts.join("")}</div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Rolling window starts</th><th>Rows fitted</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${rr}
          <tr><td><b>Mean &plusmn; sd</b></td><td></td><td>${pct(ro.precision.mean)} &plusmn; ${pct(ro.precision.sd)}</td><td>${pct(ro.recall.mean)} &plusmn; ${pct(ro.recall.sd)}</td><td>${pct(ro.f1.mean)} &plusmn; ${pct(ro.f1.sd)}</td></tr></tbody></table></div>
        <div class="note">Calibration on the test window: expected calibration error ${f3(d.calibration.ece)}, Brier score ${f3(d.calibration.brier)} (a predicted 80% really is about 80%).</div>`;
    }).join("");
    return `<div class="card sect"><h2>Overfitting and underfitting checks</h2>
      <div class="note">Five pass/fail checks (limits fixed before looking at the numbers) decide the verdict, and the learning curve, capacity curves and rolling windows show whether the result is stable. Overfitting shows up as a large gap between fit rows and new rows. Underfitting shows up as poor scores even on the fit rows, a supervised model on the same features being far better, a simple rule doing as well, or more capacity still helping. The learning curve should flatten with train and validation together, and the chosen setting should sit on the plateau of the capacity curves (dashed line).</div>${blocks}</div>`;
  }

  function freshControlsCard(dg) {
    const fw = dg.fresh_worlds, names = Object.keys(fw.summary);
    const rows = names.map((n) => {
      const s = fw.summary[n].f1;
      return `<tr><td><b>${esc(n)}</b></td><td>${pct(s.mean)} &plusmn; ${pct(s.sd)}</td><td>${pct(s.min)} to ${pct(s.max)}</td><td>${pct(fw.summary[n].precision.mean)}</td><td>${pct(fw.summary[n].recall.mean)}</td></tr>`;
    }).join("");
    const ctrl = Object.entries(dg.controls || {}).flatMap(([n, cs]) => Object.entries(cs).map(([k, c]) =>
      `<tr><td><b>${esc(n)}</b></td><td>${esc(c.description)}</td><td>${f3(c.train.pr_auc)}</td><td>${f3(c.test.pr_auc)}</td><td>${esc(c.verdict)}</td><td class="${c.detected ? "pass" : "fail"}">${c.detected ? "caught" : "MISSED"}</td></tr>`)).join("");
    return `<div class="card sect"><h2>Brand-new datasets (never seen, never refitted)</h2>
      <div class="note">The final models and calibrators are applied unchanged to ${fw.seeds.length} freshly generated datasets (new vehicles, trips and anomalies; seeds ${fw.seeds.join(", ")}). Stable scores here mean the models learned the pattern, not this one dataset. The data comes from the same generator, so this shows robustness to sampling, not to a different real-world distribution.</div>
      <div class="scroll"><table class="mtable"><thead><tr><th>Detector</th><th>F1 mean &plusmn; sd</th><th>F1 range</th><th>Precision</th><th>Recall</th></tr></thead><tbody>${rows}</tbody></table></div></div>
      <div class="card sect"><h2>Can the checks actually fail? (positive controls)</h2>
      <div class="note">A check that never fails proves nothing, so each run also tests deliberately broken models: a memoriser that only knows when and where anomalies happened (should be flagged as overfit) and a model starved of its informative features (should be flagged as underfit).</div>
      <div class="scroll"><table class="mtable"><thead><tr><th>Detector</th><th>Control</th><th>Train PR-AUC</th><th>Test PR-AUC</th><th>Verdict</th><th>Result</th></tr></thead><tbody>${ctrl}</tbody></table></div></div>`;
  }

  // ---------- rendering: model tab ----------
  async function renderModel() {
    const box = $("model-body");
    if (!state.metrics) {
      try { state.metrics = await api("/api/metrics"); }
      catch (e) { box.innerHTML = `<div class="card empty">${esc(e.message)}</div>`; return; }
    }
    const m = state.metrics.models, meta = state.metrics.meta;
    const mlRows = ["api_flood", "fare_anomaly"].map((n) => {
      const d = m[n], c = d["calibrated_p_ge_0.5"], b = d.baseline_iforest_default_threshold;
      return `<tr><td><b>${esc(n)}</b>${d.selected_model ? `<div class="sub">${esc(famName(d.selected_model.family))}</div>` : ""}</td><td>Calibrated (p &gt;= 0.5)</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${num(c.fp)}</td><td>${d.roc_auc}</td><td>${d.pr_auc}</td></tr>
        <tr><td></td><td class="sub">Default Isolation Forest threshold</td><td>${pct(b.precision)}</td><td>${pct(b.recall)}</td><td>${pct(b.f1)}</td><td>${num(b.fp)}</td><td></td><td></td></tr>`;
    }).join("");
    const ruleRows = Object.entries(m.rules || {}).map(([n, r]) =>
      `<tr><td><b>${esc(n)}</b></td><td>${num(r.flagged)}</td><td>${num(r.labelled)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${r.probability_used}</td></tr>`).join("");

    let live = "";
    const ev = state.run && state.run.evaluation;
    if (ev) {
      const rows = Object.entries(ev).map(([n, r]) =>
        `<tr><td><b>${esc(n)}</b></td><td>${num(r.flagged)}</td><td>${num(r.labelled)}</td><td>${num(r.true_positives)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td></tr>`).join("");
      live = `<div class="card sect"><h2>Live check on the current upload</h2>
        <div class="note">The file has an anomaly_type column, so every detector is scored against it. If the file contains rows the models trained on, these numbers look slightly better than the held-out ones above.</div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Detector</th><th>Flagged</th><th>Labelled</th><th>True positives</th><th>Precision</th><th>Recall</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
    }

    box.innerHTML = `<div class="panel">
      <div class="card sect"><h2>Machine-learning detectors (held-out test window)</h2>
        <div class="note">Chronological split: ${num(meta.split.train)} rows to fit, ${num(meta.split.calibration)} to calibrate, ${num(meta.split.test)} to test (from ${esc(meta.split.test_starts)}). Labels are used only to calibrate scores and to evaluate.</div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Detector</th><th>Variant</th><th>Precision</th><th>Recall</th><th>F1</th><th>False positives</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead><tbody>${mlRows}</tbody></table></div></div>
      <div class="card sect"><h2>Rule detectors (full dataset)</h2>
        <div class="note">Rules have no fitted parameters, so they are evaluated on all rows. Probability used is Laplace-smoothed precision, so no rule is ever treated as 100% certain.</div>
        <div class="scroll"><table class="mtable"><thead><tr><th>Detector</th><th>Flagged</th><th>Labelled</th><th>Precision</th><th>Recall</th><th>Probability used</th></tr></thead><tbody>${ruleRows}</tbody></table></div></div>
      ${live}</div>`;
    try {
      state.diag = state.diag || await api("/api/diagnostics");
      box.insertAdjacentHTML("beforeend", `<div class="panel" style="margin-top:20px">${selectionCard(state.diag.selection)}${diagnosticsCard(state.diag.diagnostics)}${freshControlsCard(state.diag.diagnostics)}</div>`);
    } catch (e) { /* evidence files not generated yet: the rest of the tab still works */ }
  }

  // ---------- rendering: blocked tab ----------
  async function loadBlocked() {
    try {
      const tags = await api("/api/blocked-tags");
      $("tab-blocked").textContent = tags.length ? `Blocked tags (${tags.length})` : "Blocked tags";
      $("blocked-body").innerHTML = tags.length
        ? `<div class="card scroll"><table class="mtable"><thead><tr><th>Tag</th><th>Blocked at</th><th>From transaction</th><th></th></tr></thead><tbody>${tags.map((t) =>
          `<tr><td><b>${esc(t.tag_id)}</b></td><td>${esc(t.blocked_at)}</td><td>${esc(t.txn_id)}</td><td><button class="btn small" type="button" data-unblock="${esc(t.tag_id)}">Unblock</button></td></tr>`).join("")}</tbody></table></div>`
        : '<div class="card empty">No blocked tags. Use "Block tag" on an alert to add one.</div>';
    } catch (e) {
      banner(e.message, "error");
    }
  }

  // ---------- tabs and wiring ----------
  function showTab(name) {
    state.tab = name;
    document.querySelectorAll("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    ["alerts", "analytics", "model", "blocked"].forEach((t) => { $("panel-" + t).hidden = t !== name; });
    if (name === "model") renderModel();
    if (name === "blocked") loadBlocked();
  }

  function applyFilters() {
    state.filters = {
      severity: $("f-severity").value, detector: $("f-detector").value, plaza: $("f-plaza").value,
      status: $("f-status").value, q: $("f-q").value.trim(),
    };
    state.page = 1;
    fetchAlerts();
  }

  function wire() {
    $("btn-demo").addEventListener("click", () => runAnalysis(() => api("/api/analyze-demo", { method: "POST" })));
    $("btn-upload").addEventListener("click", () => $("file").click());
    $("file").addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      runAnalysis(() => {
        const fd = new FormData();
        fd.append("file", file);
        return api("/api/analyze", { method: "POST", body: fd });
      });
      e.target.value = "";
    });
    document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
    ["f-severity", "f-detector", "f-plaza", "f-status"].forEach((id) => $(id).addEventListener("change", applyFilters));
    $("f-q").addEventListener("input", debounce(applyFilters, 300));
    $("prev").addEventListener("click", () => { if (state.page > 1) { state.page--; fetchAlerts(); } });
    $("next").addEventListener("click", () => { state.page++; fetchAlerts(); });
    $("rows").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-txn]");
      if (!tr) return;
      state.selected = tr.dataset.txn;
      renderTable();
      renderDetail();
      const btn = $("rows").querySelector("tr.sel .linkbtn");
      if (btn && e.target.closest(".linkbtn")) btn.focus();
    });
    $("detail").addEventListener("click", (e) => {
      const b = e.target.closest("[data-act]");
      if (b) act(b.dataset.act);
    });
    $("blocked-body").addEventListener("click", async (e) => {
      const b = e.target.closest("[data-unblock]");
      if (!b) return;
      try {
        await api("/api/blocked-tags/" + encodeURIComponent(b.dataset.unblock), { method: "DELETE" });
        banner(`Tag ${b.dataset.unblock} unblocked.`, "info");
        await loadBlocked();
      } catch (err) { banner(err.message, "error"); }
    });
  }

  async function init() {
    wire();
    $("f-detector").insertAdjacentHTML("beforeend", DETECTORS.map((d) => `<option>${esc(d)}</option>`).join(""));
    try {
      const health = await api("/api/health");
      $("storage").textContent = health.storage === "memory" ? "Storage: in-memory (resets on restart)" : "Storage: MongoDB";
      const plazas = await api("/api/plazas");
      $("f-plaza").insertAdjacentHTML("beforeend",
        plazas.map((p) => p.name).sort().map((n) => `<option>${esc(n)}</option>`).join(""));
      const runs = await api("/api/runs?limit=1");
      if (runs.length) await loadRun(await api("/api/runs/" + runs[0].run_id));
      loadBlocked();
    } catch (e) {
      banner("Could not reach the TollGuard API: " + e.message, "error");
    }
  }

  init();
})();