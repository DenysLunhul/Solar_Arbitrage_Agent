import { useState, useCallback, useRef, useEffect } from "react";
import {
  AreaChart, Area, ComposedChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Line,
} from "recharts";
import * as api from "./api.js";

// ─── Helpers ─────────────────────────────────────────────────────────
const clamp  = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt    = (n, d = 2) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtTs  = (ts) => ts ? String(ts).slice(11, 16) : "—";
const fmtT   = (i)  => { const h = Math.floor(i/4), m = (i%4)*15; return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}`; };

const parseCsv = (text) => {
  const [header, ...rows] = text.trim().split("\n");
  const keys = header.split(",").map(k => k.trim());
  return rows.filter(r => r.trim()).map(row => {
    const vals = row.split(",");
    return Object.fromEntries(keys.map((k, i) => [k, isNaN(vals[i]) ? vals[i] : parseFloat(vals[i])]));
  });
};

const computeSummary = (plan) => {
  if (!plan?.length) return null;
  return {
    total_money_earned: plan.reduce((s,r) => s + (r.money_earned_ts ?? 0), 0),
    sold_kwh:           plan.filter(r => (r.grid_kwh??0) < 0).reduce((s,r) => s + Math.abs(r.grid_kwh), 0),
    bought_kwh:         plan.filter(r => (r.grid_kwh??0) > 0).reduce((s,r) => s + r.grid_kwh, 0),
    solar_kwh:          plan.reduce((s,r) => s + (r.solar_kwh ?? r.solar_gen_kwh ?? 0), 0),
    unmet_load_kwh:     plan.reduce((s,r) => s + (r.unmet_load_kwh ?? 0), 0),
    lcos_total_uah:     plan.reduce((s,r) => s + (r.lcos_cost ?? 0), 0),
    initial_soc:        plan[0]?.soc ?? 0,
    final_soc:          plan[plan.length-1]?.soc ?? 0,
    steps:              plan.length,
  };
};

// ─── CSS ─────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f1f5f9; font-family: 'Sora', sans-serif; color: #0f172a; }

  /* ── Login ── */
  .login-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: #f1f5f9; }
  .login-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 36px 32px; width: 100%; max-width: 360px; box-shadow: 0 4px 24px rgba(0,0,0,.07); }
  .login-logo  { display: flex; flex-direction: column; align-items: center; margin-bottom: 24px; gap: 10px; }
  .login-title { font-size: 18px; font-weight: 700; letter-spacing: -.02em; }
  .login-sub   { font-size: 12px; color: #94a3b8; }
  .ok-msg { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 9px 12px; font-size: 12px; color: #15803d; margin-bottom: 12px; }

  /* ── App shell ── */
  .app { min-height: 100vh; display: flex; flex-direction: column; }
  .topbar { background: #fff; border-bottom: 1px solid #e2e8f0; height: 60px; padding: 0 28px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
  .logo-wrap { display: flex; align-items: center; gap: 11px; }
  .logo-icon { width: 34px; height: 34px; background: #1d4ed8; border-radius: 9px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(29,78,216,.35); }
  .brand     { font-size: 15px; font-weight: 700; letter-spacing: -.02em; }
  .brand-sub { font-size: 11px; color: #94a3b8; margin-top: 1px; font-weight: 400; }
  .topbar-right { display: flex; align-items: center; gap: 8px; }
  .chip { display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px; border-radius: 20px; font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .chip-blue  { background: #dbeafe; color: #1d4ed8; }
  .chip-green { background: #dcfce7; color: #15803d; }
  .dot-green  { width: 6px; height: 6px; border-radius: 50%; background: #16a34a; animation: blink 2s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Layout ── */
  .layout  { display: grid; grid-template-columns: 260px 1fr; flex: 1; }
  .sidebar { background: #fff; border-right: 1px solid #e2e8f0; padding: 14px 12px 16px; overflow-y: auto; height: calc(100vh - 60px); position: sticky; top: 60px; display: flex; flex-direction: column; gap: 2px; }
  .content { padding: 22px; display: flex; flex-direction: column; gap: 18px; overflow-y: auto; }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 2px; background: #f1f5f9; border-radius: 8px; padding: 3px; margin-bottom: 10px; }
  .tab  { flex: 1; padding: 6px 4px; border: none; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; background: transparent; color: #64748b; font-family: 'Sora', sans-serif; transition: all .15s; }
  .tab.active { background: #fff; color: #1d4ed8; box-shadow: 0 1px 3px rgba(0,0,0,.08); }

  /* ── Sidebar fields ── */
  .sec-title { font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #94a3b8; padding: 8px 4px 3px; margin-top: 2px; }
  .field { display: flex; flex-direction: column; gap: 3px; padding: 1px 4px; margin-bottom: 4px; }
  .field-hdr { display: flex; justify-content: space-between; align-items: center; }
  .field label { font-size: 11px; color: #64748b; font-weight: 500; }
  .field input, .field select {
    border: 1.5px solid #e2e8f0; border-radius: 8px; padding: 7px 10px;
    font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #0f172a;
    background: #f8fafc; outline: none; width: 100%;
    transition: border-color .15s, box-shadow .15s, background .15s;
  }
  .field input:focus, .field select:focus { border-color: #1d4ed8; box-shadow: 0 0 0 3px #dbeafe; background: #fff; }
  .field select { cursor: pointer; }
  .field input::placeholder { color: #cbd5e1; }
  .ref-btn { border: none; background: transparent; color: #94a3b8; cursor: pointer; font-size: 14px; line-height: 1; padding: 1px 3px; transition: color .15s; }
  .ref-btn:hover { color: #1d4ed8; }

  /* ── Buttons ── */
  .btn { display: flex; align-items: center; justify-content: center; gap: 7px; border: none; border-radius: 9px; cursor: pointer; font-family: 'Sora', sans-serif; font-weight: 600; transition: all .15s; }
  .btn-run { width: 100%; padding: 11px; font-size: 13px; color: #fff; background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%); box-shadow: 0 1px 3px rgba(29,78,216,.3), inset 0 1px 0 rgba(255,255,255,.12); }
  .btn-run:hover:not(:disabled) { background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); box-shadow: 0 4px 14px rgba(29,78,216,.4); transform: translateY(-1px); }
  .btn-run:active  { transform: translateY(0); }
  .btn-run:disabled { opacity: .4; cursor: not-allowed; transform: none; }
  .btn-ghost { background: transparent; color: #64748b; border: 1.5px solid #e2e8f0; width: 100%; padding: 9px; font-size: 12px; }
  .btn-ghost:hover { background: #f8fafc; color: #334155; border-color: #cbd5e1; }

  /* ── Upload ── */
  .divider { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
  .divider-line { flex: 1; height: 1px; background: #e2e8f0; }
  .divider-text { font-size: 10px; color: #94a3b8; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
  .upload-box { border: 1.5px dashed #cbd5e1; border-radius: 9px; padding: 12px 10px; text-align: center; cursor: pointer; transition: all .15s; background: #fafafa; }
  .upload-box:hover    { border-color: #1d4ed8; background: #eff6ff; }
  .upload-box.ok       { border-color: #16a34a; background: #f0fdf4; }
  .upload-box p        { font-size: 11px; color: #94a3b8; margin-top: 4px; line-height: 1.4; }
  .upload-box.ok p     { color: #15803d; font-weight: 500; }

  /* ── KPIs ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px 16px; position: relative; overflow: hidden; }
  .kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 12px 12px 0 0; }
  .kpi-label { font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
  .kpi-val   { font-size: 24px; font-weight: 700; letter-spacing: -.03em; line-height: 1; font-family: 'JetBrains Mono', monospace; }
  .kpi-unit  { font-size: 12px; font-weight: 400; color: #94a3b8; margin-left: 3px; }
  .kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 5px; }

  /* ── Charts ── */
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .chart-full  { grid-column: 1 / -1; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
  .card-head { padding: 13px 16px; border-bottom: 1px solid #f1f5f9; display: flex; align-items: center; justify-content: space-between; }
  .card-title { font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: .06em; text-transform: uppercase; }
  .card-body  { padding: 14px 16px; }
  .legend-row { display: flex; gap: 14px; font-size: 10px; color: #64748b; font-family: 'JetBrains Mono', monospace; }

  /* ── Table ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; font-family: 'JetBrains Mono', monospace; }
  th { padding: 8px 12px; text-align: left; font-size: 9px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #94a3b8; border-bottom: 1px solid #f1f5f9; white-space: nowrap; background: #fafafa; }
  td { padding: 7px 12px; border-bottom: 1px solid #f8fafc; white-space: nowrap; color: #334155; }
  tr:hover td { background: #f8fafc; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-flex; align-items: center; gap: 3px; padding: 2px 7px; border-radius: 4px; font-size: 9px; font-weight: 700; font-family: 'Sora', sans-serif; letter-spacing: .04em; }
  .b-sell { background: #dcfce7; color: #15803d; }
  .b-buy  { background: #fee2e2; color: #b91c1c; }
  .b-idle { background: #f1f5f9; color: #94a3b8; }
  .b-chg  { background: #dbeafe; color: #1d4ed8; }
  .b-dchg { background: #fef3c7; color: #b45309; }
  .soc-row   { display: flex; align-items: center; gap: 7px; }
  .soc-track { width: 52px; height: 5px; background: #f1f5f9; border-radius: 3px; overflow: hidden; }
  .soc-fill  { height: 100%; border-radius: 3px; }
  .soc-txt   { font-size: 10px; color: #64748b; min-width: 28px; }

  /* ── Pager ── */
  .pager { display: flex; align-items: center; gap: 5px; }
  .pg-btn { width: 26px; height: 26px; border-radius: 6px; border: 1.5px solid #e2e8f0; background: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #64748b; transition: all .1s; }
  .pg-btn:hover:not(:disabled) { background: #f8fafc; border-color: #cbd5e1; }
  .pg-btn:disabled { opacity: .3; cursor: not-allowed; }
  .pg-info { font-size: 10px; color: #94a3b8; padding: 0 4px; }

  /* ── States ── */
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: 80px 20px; }
  .empty-icon { width: 52px; height: 52px; background: #f1f5f9; border-radius: 14px; display: flex; align-items: center; justify-content: center; }
  .empty h3 { font-size: 14px; font-weight: 600; color: #64748b; }
  .empty p  { font-size: 12px; color: #94a3b8; text-align: center; max-width: 260px; line-height: 1.6; }
  .loader { display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 60px; }
  .spin { width: 26px; height: 26px; border: 2.5px solid #e2e8f0; border-top-color: #1d4ed8; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .err { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #b91c1c; }

  /* ── Tooltip ── */
  .ct { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 9px 12px; box-shadow: 0 4px 16px rgba(0,0,0,.08); font-family: 'JetBrains Mono', monospace; font-size: 11px; }
  .ct-lbl { color: #94a3b8; margin-bottom: 5px; font-weight: 500; }
  .ct-row { display: flex; gap: 7px; align-items: center; margin-top: 2px; }
  .ct-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 2px; }
`;

// ─── Tooltip ─────────────────────────────────────────────────────────
const CT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="ct">
      <div className="ct-lbl">{label}</div>
      {payload.map((p, i) => (
        <div className="ct-row" key={i}>
          <div className="ct-dot" style={{ background: p.color }} />
          <span style={{ color:"#64748b" }}>{p.name}:</span>
          <span style={{ fontWeight:600, color:"#0f172a" }}>{fmt(p.value, 3)}</span>
        </div>
      ))}
    </div>
  );
};

// ─── SelField ─────────────────────────────────────────────────────────
// Shows a <select> when options are available, plain <input> otherwise.
function SelField({ label, value, onChange, options, placeholder, onRefresh, editable = false }) {
  return (
    <div className="field">
      <div className="field-hdr">
        <label>{label}</label>
        <button className="ref-btn" type="button" onClick={onRefresh} title="Оновити">↻</button>
      </div>
      {options.length > 0 ? (
        <select value={value} onChange={e => onChange(e.target.value)}>
          {placeholder && <option value="">{placeholder}</option>}
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : editable ? (
        <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} />
      ) : (
        <select disabled><option>{placeholder || "Немає даних"}</option></select>
      )}
    </div>
  );
}

// ─── Login ────────────────────────────────────────────────────────────
function Login({ onLogin }) {
  const [tab,     setTab]     = useState("login");
  const [user,    setUser]    = useState("");
  const [email,   setEmail]   = useState("");
  const [pass,    setPass]    = useState("");
  const [loading, setLoading] = useState(false);
  const [err,     setErr]     = useState(null);
  const [ok,      setOk]      = useState(null);

  const switchTab = (t) => { setTab(t); setErr(null); setOk(null); };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setErr(null); setOk(null);
    try {
      if (tab === "login") {
        const data = await api.login(user, pass);
        api.setToken(data.access_token);
        onLogin();
      } else {
        await api.register(user, email, pass);
        setOk("Акаунт створено. Тепер увійдіть.");
        switchTab("login");
        setPass("");
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <style>{CSS}</style>
      <div className="login-card">
        <div className="login-logo">
          <div className="logo-icon">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
            </svg>
          </div>
          <div className="login-title">EMS Dashboard</div>
          <div className="login-sub">Energy Management · RL Agent</div>
        </div>

        <div className="tabs">
          <button className={`tab${tab==="login" ? " active" : ""}`} onClick={() => switchTab("login")}>Вхід</button>
          <button className={`tab${tab==="register" ? " active" : ""}`} onClick={() => switchTab("register")}>Реєстрація</button>
        </div>

        <form onSubmit={handleSubmit}>
          {ok  && <div className="ok-msg">{ok}</div>}
          {err && <div className="err" style={{marginBottom:12}}>{err}</div>}
          <div className="field">
            <label>Ім'я користувача</label>
            <input type="text" value={user} onChange={e => setUser(e.target.value)} required autoFocus />
          </div>
          {tab === "register" && (
            <div className="field">
              <label>Email</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} required />
            </div>
          )}
          <div className="field" style={{marginBottom:16}}>
            <label>Пароль</label>
            <input type="password" value={pass} onChange={e => setPass(e.target.value)} required />
          </div>
          <button className="btn btn-run" type="submit" disabled={loading}>
            {loading
              ? <><div className="spin" style={{width:13,height:13,borderWidth:2}}/>Завантаження...</>
              : tab === "login" ? "Увійти" : "Зареєструватись"
            }
          </button>
        </form>
      </div>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────
function Sidebar({ mode, setMode, configs, strategies, histDates, configName, setConfigName, strategyName, setStrategyName, histDate, setHistDate, initialSoc, setInitialSoc, onRun, loading, onUpload, fileName, onRefreshConfigs, onRefreshStrategies, onRefreshHistDates, onLogout }) {
  const fileRef = useRef();
  const runLabel = { sac: "Запустити SAC модель", default: "Запустити Default стратегію", history: "Завантажити з БД" }[mode];
  const canRun   = !!configName && (mode !== "default" || !!strategyName) && !loading;

  return (
    <aside className="sidebar">
      <div className="tabs">
        {["sac","default","history"].map(m => (
          <button key={m} className={`tab${mode===m ? " active" : ""}`} onClick={() => setMode(m)}>
            {m === "sac" ? "SAC" : m === "default" ? "Default" : "Історія"}
          </button>
        ))}
      </div>

      <div className="sec-title">Конфігурація</div>
      <SelField
        label="Назва конфігурації" value={configName} onChange={setConfigName}
        options={configs} placeholder="— оберіть або введіть —"
        onRefresh={onRefreshConfigs} editable
      />

      {mode === "default" && (
        <>
          <div className="sec-title">Стратегія</div>
          <SelField
            label="Назва стратегії" value={strategyName} onChange={setStrategyName}
            options={strategies} placeholder="— оберіть або введіть —"
            onRefresh={onRefreshStrategies} editable
          />
        </>
      )}

      {mode === "history" && (
        <>
          <div className="sec-title">Дата прогнозу</div>
          <SelField
            label="Дата" value={histDate} onChange={setHistDate}
            options={histDates} placeholder="— остання —"
            onRefresh={onRefreshHistDates}
          />
        </>
      )}

      {mode !== "history" && (
        <>
          <div className="sec-title">Параметри запуску</div>
          <div className="field">
            <label>Початковий SoC (0–1)</label>
            <input
              type="number" step="0.05" min="0" max="1"
              value={initialSoc} onChange={e => setInitialSoc(e.target.value)}
              placeholder="авто (з БД)"
            />
          </div>
        </>
      )}

      <div style={{marginTop:12, display:"flex", flexDirection:"column", gap:8}}>
        <button className="btn btn-run" onClick={onRun} disabled={!canRun}>
          {loading
            ? <><div className="spin" style={{width:13,height:13,borderWidth:2}}/>Завантаження...</>
            : <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                {runLabel}
              </>
          }
        </button>

        <div className="divider">
          <div className="divider-line"/><span className="divider-text">або</span><div className="divider-line"/>
        </div>

        <input ref={fileRef} type="file" accept=".csv" onChange={onUpload} style={{display:"none"}}/>
        <div className={`upload-box${fileName ? " ok" : ""}`} onClick={() => fileRef.current?.click()}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={fileName ? "#15803d" : "#94a3b8"} strokeWidth="2" style={{margin:"0 auto",display:"block"}}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <p>{fileName || "Завантажити dispatch_plan.csv"}</p>
        </div>
      </div>

      <div style={{marginTop:"auto", paddingTop:16, borderTop:"1px solid #f1f5f9"}}>
        <button className="btn btn-ghost" onClick={onLogout}>Вийти</button>
      </div>
    </aside>
  );
}

// ─── KPIs ────────────────────────────────────────────────────────────
function Kpis({ summary }) {
  const { total_money_earned: earned, sold_kwh: sold, bought_kwh: bought, solar_kwh: solar, unmet_load_kwh: unmet, lcos_total_uah: lcos, initial_soc: iSoc, final_soc: fSoc, steps } = summary;
  const days = Math.max(1, Math.round(steps / 96));

  const K = ({ color, label, value, unit, sub }) => (
    <div className="kpi">
      <div className="kpi-accent" style={{ background: color }}/>
      <div className="kpi-label">{label}</div>
      <div className="kpi-val" style={{ color }}>{fmt(value)}<span className="kpi-unit">{unit}</span></div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );

  return (
    <div className="kpi-grid">
      <K color={earned >= 0 ? "#16a34a" : "#dc2626"} label="Зароблено"
         value={earned} unit=" UAH" sub={`${days} дн · ${steps} кроків`} />
      <K color="#1d4ed8" label="Продано в мережу"
         value={sold} unit=" кВт·год" sub={`куплено: ${fmt(bought)} кВт·год`} />
      <K color="#d97706" label="Сонячна генерація"
         value={solar} unit=" кВт·год" sub={`деградація: ${fmt(lcos)} UAH`} />
      <K color={unmet > 0 ? "#dc2626" : "#16a34a"} label="Непокрите навант."
         value={unmet} unit=" кВт·год" sub={`SoC: ${fmt(iSoc*100,0)}% → ${fmt(fSoc*100,0)}%`} />
    </div>
  );
}

// ─── Charts ──────────────────────────────────────────────────────────
function Charts({ plan }) {
  const data = plan.map((r, i) => ({
    t:    r.timestamp ? fmtTs(r.timestamp) : fmtT(i),
    soc:  +((r.soc ?? 0) * 100).toFixed(1),
    tgt:  +((r.target_soc ?? 0) * 100).toFixed(1),
    sol:  +(r.solar_kwh ?? r.solar_gen_kwh ?? 0).toFixed(4),
    grid: +(r.grid_kwh ?? 0).toFixed(4),
  }));

  const ticks = data.filter((_, i) => i % 8 === 0).map(d => d.t);
  const xP = { dataKey:"t", ticks, tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" } };
  const yP = { tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" }, width:38 };
  const grid = <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>;

  return (
    <div className="charts-grid">
      {/* SoC — full width */}
      <div className="card chart-full">
        <div className="card-head">
          <span className="card-title">Стан заряду батареї (SoC)</span>
          <div className="legend-row">
            <span><span style={{color:"#1d4ed8"}}>●</span> SoC %</span>
            <span><span style={{color:"#d97706"}}>‒ ‒</span> Ціль %</span>
            <span style={{color:"#dc2626"}}>— 20% мін</span>
            <span style={{color:"#d97706"}}>— 80% макс</span>
          </div>
        </div>
        <div className="card-body" style={{height:200}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              <defs>
                <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#1d4ed8" stopOpacity={.12}/>
                  <stop offset="95%" stopColor="#1d4ed8" stopOpacity={0}/>
                </linearGradient>
              </defs>
              {grid}
              <XAxis {...xP}/><YAxis {...yP} domain={[0,100]} unit="%"/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={20} stroke="#dc2626" strokeDasharray="4 3" strokeWidth={1}/>
              <ReferenceLine y={80} stroke="#d97706" strokeDasharray="4 3" strokeWidth={1}/>
              <Area type="monotone" dataKey="soc" name="SoC"  stroke="#1d4ed8" fill="url(#gS)" strokeWidth={2} dot={false}/>
              <Line type="monotone" dataKey="tgt" name="Ціль" stroke="#d97706" strokeWidth={1.5} strokeDasharray="5 3" dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Solar */}
      <div className="card">
        <div className="card-head"><span className="card-title">Генерація сонця (кВт·год)</span></div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              <defs>
                <linearGradient id="gSol" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#d97706" stopOpacity={.18}/>
                  <stop offset="95%" stopColor="#d97706" stopOpacity={0}/>
                </linearGradient>
              </defs>
              {grid}
              <XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <Area type="monotone" dataKey="sol" name="Сонце" stroke="#d97706" fill="url(#gSol)" strokeWidth={2} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid exchange — green = sell, red = buy */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Обмін з мережею (кВт·год)</span>
          <div className="legend-row">
            <span><span style={{color:"#dc2626"}}>■</span> Купівля (+)</span>
            <span><span style={{color:"#16a34a"}}>■</span> Продаж (−)</span>
          </div>
        </div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              {grid}
              <XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#e2e8f0"/>
              <Bar dataKey="grid" name="Мережа" radius={[2,2,0,0]}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.grid < -0.001 ? "#16a34a" : entry.grid > 0.001 ? "#dc2626" : "#e2e8f0"}/>
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─── Table ────────────────────────────────────────────────────────────
function Table({ plan }) {
  const [page, setPage] = useState(0);
  const PER   = 20;
  const total = Math.ceil(plan.length / PER);
  const rows  = plan.slice(page * PER, (page + 1) * PER);

  const gridB = v => {
    if (v < -.001) return <span className="badge b-sell">↑ Продаж</span>;
    if (v >  .001) return <span className="badge b-buy">↓ Купівля</span>;
    return <span className="badge b-idle">—</span>;
  };
  const battB = v => {
    if (v >  .001) return <span className="badge b-chg">⬆ Заряд</span>;
    if (v < -.001) return <span className="badge b-dchg">⬇ Розряд</span>;
    return <span className="badge b-idle">—</span>;
  };
  const socBar = v => {
    const pct   = clamp(v * 100, 0, 100);
    const color = pct < 20 ? "#dc2626" : pct > 80 ? "#d97706" : "#16a34a";
    return (
      <div className="soc-row">
        <div className="soc-track"><div className="soc-fill" style={{width:`${pct}%`,background:color}}/></div>
        <span className="soc-txt">{pct.toFixed(0)}%</span>
      </div>
    );
  };

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">Dispatch Plan · {plan.length} кроків</span>
        <div className="pager">
          <button className="pg-btn" onClick={() => setPage(0)}        disabled={page===0}>«</button>
          <button className="pg-btn" onClick={() => setPage(p=>p-1)}   disabled={page===0}>‹</button>
          <span className="pg-info">{page+1} / {total}</span>
          <button className="pg-btn" onClick={() => setPage(p=>p+1)}   disabled={page===total-1}>›</button>
          <button className="pg-btn" onClick={() => setPage(total-1)}  disabled={page===total-1}>»</button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Час</th><th>SoC</th><th>Ціль</th>
              <th>Батарея</th><th>Мережа</th>
              <th>Сонце кВт·год</th><th>Навант. кВт·год</th>
              <th>Ціна DAM</th><th>P&L UAH</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const timeLabel = r.timestamp ? fmtTs(r.timestamp) : fmtT(r.step ?? (page * PER + i));
              const earned    = r.money_earned_ts ?? 0;
              return (
                <tr key={i}>
                  <td style={{color:"#94a3b8",fontWeight:500}}>{timeLabel}</td>
                  <td>{socBar(r.soc ?? 0)}</td>
                  <td style={{color:"#d97706"}}>{(((r.target_soc??0)*100)).toFixed(0)}%</td>
                  <td>{battB(r.battery_kwh ?? 0)}</td>
                  <td>{gridB(r.grid_kwh ?? 0)}</td>
                  <td style={{color:(r.solar_kwh??r.solar_gen_kwh??0) > .001 ? "#d97706" : "#94a3b8"}}>
                    {fmt(r.solar_kwh ?? r.solar_gen_kwh ?? 0, 4)}
                  </td>
                  <td style={{color:"#64748b"}}>{fmt(r.load_kwh ?? 0, 4)}</td>
                  <td style={{color:"#64748b",fontFamily:"JetBrains Mono,monospace"}}>
                    {r.dam_price != null ? fmt(r.dam_price, 2) : "—"}
                  </td>
                  <td style={{color: earned >= 0 ? "#16a34a" : "#dc2626", fontWeight:600}}>
                    {earned >= 0 ? "+" : ""}{fmt(earned, 3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────
export default function App() {
  const [authed,       setAuthed]       = useState(!!api.getToken());
  const [mode,         setMode]         = useState("sac");
  const [configs,      setConfigs]      = useState([]);
  const [strategies,   setStrategies]   = useState([]);
  const [histDates,    setHistDates]    = useState([]);
  const [configName,   setConfigName]   = useState("");
  const [strategyName, setStrategyName] = useState("");
  const [histDate,     setHistDate]     = useState("");
  const [initialSoc,   setInitialSoc]   = useState("");
  const [plan,         setPlan]         = useState(null);
  const [summary,      setSummary]      = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [fileName,     setFileName]     = useState(null);

  const loadConfigs    = useCallback(() => api.listConfigs().then(d => { const ns = d.map(c => c.config_name); setConfigs(ns); if (ns.length && !configName) setConfigName(ns[0]); }).catch(()=>{}), [configName]);
  const loadStrategies = useCallback(() => api.listStrategies().then(d => { const ns = d.map(s => s.strategy_name); setStrategies(ns); if (ns.length && !strategyName) setStrategyName(ns[0]); }).catch(()=>{}), [strategyName]);
  const loadHistDates  = useCallback(() => {
    if (!configName) return;
    api.getHistoryDates(configName).then(dates => { setHistDates(dates); if (dates.length) setHistDate(d => d || dates[0]); }).catch(()=>{});
  }, [configName]);

  useEffect(() => { if (authed) { loadConfigs(); loadStrategies(); } }, [authed]);
  useEffect(() => { if (authed && mode === "history") loadHistDates(); }, [authed, mode, configName]);

  const handleLogin  = useCallback(() => setAuthed(true), []);
  const handleLogout = useCallback(() => { api.clearToken(); setAuthed(false); setPlan(null); setSummary(null); }, []);

  const handleRun = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const soc = initialSoc !== "" ? parseFloat(initialSoc) : undefined;
      let result;
      if (mode === "sac")     result = await api.getPredictions(configName, soc);
      else if (mode === "default") result = await api.getDefaultPredictions(configName, strategyName, soc);
      else                    result = await api.getHistory(configName, histDate || undefined);
      setPlan(result.dispatch_plan);
      setSummary(result.summary);
      setFileName(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [mode, configName, strategyName, histDate, initialSoc]);

  const handleUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const data = parseCsv(ev.target.result);
        setPlan(data);
        setSummary(computeSummary(data));
        setFileName(file.name);
        setError(null);
      } catch (err) {
        setError("Не вдалося прочитати CSV: " + err.message);
      }
    };
    reader.readAsText(file);
  }, []);

  if (!authed) return <Login onLogin={handleLogin}/>;

  const modeLabel = { sac: "SAC", default: "Default", history: "Історія" }[mode];

  return (
    <>
      <style>{CSS}</style>
      <div className="app">
        <header className="topbar">
          <div className="logo-wrap">
            <div className="logo-icon">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
              </svg>
            </div>
            <div>
              <div className="brand">EMS Dashboard</div>
              <div className="brand-sub">Energy Management · RL Agent</div>
            </div>
          </div>
          <div className="topbar-right">
            {loading && <div className="spin" style={{width:16,height:16,borderWidth:2}}/>}
            {plan && !loading && (
              <div className="chip chip-blue">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                {plan.length} кроків · {modeLabel}
              </div>
            )}
            <div className="chip chip-green"><div className="dot-green"/>Ready</div>
          </div>
        </header>

        <div className="layout">
          <Sidebar
            mode={mode} setMode={setMode}
            configs={configs} strategies={strategies} histDates={histDates}
            configName={configName}   setConfigName={setConfigName}
            strategyName={strategyName} setStrategyName={setStrategyName}
            histDate={histDate}       setHistDate={setHistDate}
            initialSoc={initialSoc}   setInitialSoc={setInitialSoc}
            onRun={handleRun} loading={loading}
            onUpload={handleUpload} fileName={fileName}
            onRefreshConfigs={loadConfigs}
            onRefreshStrategies={loadStrategies}
            onRefreshHistDates={loadHistDates}
            onLogout={handleLogout}
          />

          <main className="content">
            {error && <div className="err">⚠ {error}</div>}

            {loading && (
              <div className="loader">
                <div className="spin"/>
                <span style={{fontSize:13,color:"#64748b"}}>Завантаження...</span>
              </div>
            )}

            {!loading && !plan && (
              <div className="empty">
                <div className="empty-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                  </svg>
                </div>
                <h3>Немає даних</h3>
                <p>Оберіть конфігурацію та натисни кнопку запуску або завантаж dispatch_plan.csv</p>
              </div>
            )}

            {!loading && plan && summary && (
              <>
                <Kpis summary={summary}/>
                <Charts plan={plan}/>
                <Table plan={plan}/>
              </>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
