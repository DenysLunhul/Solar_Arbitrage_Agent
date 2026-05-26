import { useState, useCallback, useRef, useEffect } from "react";
import {
  AreaChart, Area, BarChart, Bar, ComposedChart, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Line, Legend,
} from "recharts";
import * as api from "./api.js";

const clamp  = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt    = (n, d = 2) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtTs  = (ts) => ts ? String(ts).slice(11, 16) : "—";
const fmtT   = (i)  => { const h = Math.floor(i/4), m = (i%4)*15; return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}`; };
const fmtK   = (n)  => typeof n === "number" ? (Math.abs(n) >= 1000 ? (n/1000).toFixed(1)+"k" : n.toFixed(0)) : "—";

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

const buildTime = (r, i) => r.timestamp ? fmtTs(r.timestamp) : fmtT(i);

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
  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px 16px; position: relative; overflow: hidden; }
  .kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 12px 12px 0 0; }
  .kpi-label { font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
  .kpi-val   { font-size: 22px; font-weight: 700; letter-spacing: -.03em; line-height: 1; font-family: 'JetBrains Mono', monospace; }
  .kpi-unit  { font-size: 11px; font-weight: 400; color: #94a3b8; margin-left: 3px; }
  .kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 5px; }

  /* ── Charts ── */
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .chart-full  { grid-column: 1 / -1; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
  .card-head { padding: 13px 16px; border-bottom: 1px solid #f1f5f9; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; }
  .card-title { font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: .06em; text-transform: uppercase; }
  .card-body  { padding: 14px 16px; }
  .legend-row { display: flex; gap: 10px; font-size: 10px; color: #64748b; font-family: 'JetBrains Mono', monospace; flex-wrap: wrap; }

  /* ── Compare ── */
  .cmp-wrap { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
  .cmp-head-row { display: grid; grid-template-columns: 1.6fr 1fr 1fr 0.7fr; padding: 10px 18px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
  .cmp-row  { display: grid; grid-template-columns: 1.6fr 1fr 1fr 0.7fr; padding: 9px 18px; border-bottom: 1px solid #f8fafc; transition: background .1s; }
  .cmp-row:last-child { border-bottom: none; }
  .cmp-row:hover { background: #f8fafc; }
  .cmp-label { font-size: 11px; color: #64748b; display: flex; align-items: center; }
  .cmp-val   { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; color: #0f172a; display: flex; align-items: center; }
  .cmp-head-lbl { font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #94a3b8; display: flex; align-items: center; }
  .tag { display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: 700; letter-spacing: .04em; }
  .tag-sac { background: #dbeafe; color: #1d4ed8; }
  .tag-def { background: #fce7f3; color: #be185d; }
  .delta-up { color: #16a34a; font-size: 11px; font-weight: 700; }
  .delta-dn { color: #dc2626; font-size: 11px; font-weight: 700; }
  .delta-ne { color: #94a3b8; font-size: 11px; }
  .cmp-section-head { padding: 8px 18px 6px; font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #94a3b8; background: #fafafa; border-bottom: 1px solid #f1f5f9; border-top: 1px solid #f1f5f9; margin-top: 2px; }

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

  /* ── Modal ── */
  .modal-overlay { position: fixed; inset: 0; background: rgba(15,23,42,.45); z-index: 200; display: flex; align-items: center; justify-content: center; padding: 20px; }
  .modal { background: #fff; border-radius: 16px; width: 100%; max-width: 560px; max-height: 90vh; display: flex; flex-direction: column; box-shadow: 0 20px 60px rgba(0,0,0,.18); }
  .modal-head { padding: 18px 22px 14px; border-bottom: 1px solid #f1f5f9; display: flex; align-items: center; justify-content: space-between; }
  .modal-title { font-size: 15px; font-weight: 700; letter-spacing: -.02em; }
  .modal-close { width: 28px; height: 28px; border: none; background: #f1f5f9; border-radius: 7px; cursor: pointer; font-size: 14px; color: #64748b; display: flex; align-items: center; justify-content: center; transition: background .15s; }
  .modal-close:hover { background: #e2e8f0; }
  .modal-body { padding: 18px 22px; overflow-y: auto; flex: 1; }
  .modal-footer { padding: 14px 22px; border-top: 1px solid #f1f5f9; display: flex; gap: 8px; justify-content: flex-end; }
  .modal-tabs { display: flex; gap: 2px; background: #f1f5f9; border-radius: 8px; padding: 3px; margin-bottom: 16px; }
  .modal-tab { flex: 1; padding: 7px 4px; border: none; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; background: transparent; color: #64748b; font-family: 'Sora', sans-serif; transition: all .15s; }
  .modal-tab.active { background: #fff; color: #1d4ed8; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .form-section { font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #94a3b8; margin: 14px 0 6px; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .form-field { display: flex; flex-direction: column; gap: 3px; }
  .form-field label { font-size: 11px; color: #64748b; font-weight: 500; }
  .form-field input, .form-field select { border: 1.5px solid #e2e8f0; border-radius: 8px; padding: 7px 10px; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #0f172a; background: #f8fafc; outline: none; width: 100%; transition: border-color .15s, box-shadow .15s; }
  .form-field input:focus, .form-field select:focus { border-color: #1d4ed8; box-shadow: 0 0 0 3px #dbeafe; background: #fff; }
  .form-field.full { grid-column: 1 / -1; }
  .toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f8fafc; }
  .toggle-label { font-size: 12px; color: #334155; }
  .toggle { position: relative; width: 36px; height: 20px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle-slider { position: absolute; inset: 0; background: #e2e8f0; border-radius: 20px; cursor: pointer; transition: background .2s; }
  .toggle-slider:before { content: ""; position: absolute; width: 14px; height: 14px; left: 3px; bottom: 3px; background: #fff; border-radius: 50%; transition: transform .2s; box-shadow: 0 1px 3px rgba(0,0,0,.2); }
  .toggle input:checked + .toggle-slider { background: #1d4ed8; }
  .toggle input:checked + .toggle-slider:before { transform: translateX(16px); }
  .btn-secondary { padding: 9px 18px; font-size: 12px; color: #64748b; background: #f1f5f9; border: none; border-radius: 8px; cursor: pointer; font-family: 'Sora', sans-serif; font-weight: 600; transition: background .15s; }
  .btn-secondary:hover { background: #e2e8f0; }
  .btn-primary { padding: 9px 18px; font-size: 12px; color: #fff; background: #1d4ed8; border: none; border-radius: 8px; cursor: pointer; font-family: 'Sora', sans-serif; font-weight: 600; transition: background .15s; }
  .btn-primary:hover:not(:disabled) { background: #1e40af; }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
`;

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
    } catch (e) { setErr(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="login-page">
      <style>{CSS}</style>
      <div className="login-card">
        <div className="login-logo">
          <div className="logo-icon">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          </div>
          <div className="login-title">EMS Dashboard</div>
          <div className="login-sub">Energy Management · RL Agent</div>
        </div>
        <div className="tabs">
          <button className={`tab${tab==="login"?" active":""}`} onClick={() => switchTab("login")}>Вхід</button>
          <button className={`tab${tab==="register"?" active":""}`} onClick={() => switchTab("register")}>Реєстрація</button>
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

// ─── Settings Modal ──────────────────────────────────────────────────
function SettingsModal({ onClose, onSaved }) {
  const [tab, setTab] = useState("config");

  // Config state
  const [cfg, setCfg] = useState({
    config_name: "", battery_capacity_kwh: "100", battery_min_reserve: "15",
    battery_lcos: "1.5", battery_max_charge_power: "50", battery_max_discharge_power: "50",
    battery_efficiency: "0.95", inverter_max_power: "100", inverter_efficiency: "0.95",
    solar_peak_power: "100", solar_efficiency: "0.20", solar_tilt: "35", solar_azimuth: "0",
    grid_capacity: "100",
  });

  // Strategy state
  const [strat, setStrat] = useState({
    strategy_name: "", target_soc: "0.70", max_soc: "0.95",
    min_solar_threshold: "10", high_solar_threshold: "400",
    solar_surplus_priority: "charge_first",
    night_discharge: true, night_sell: false,
    allow_grid_charging: false, outage_reserve: "0.0",
  });

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [ok, setOk] = useState(null);

  const setC = (k, v) => setCfg(p => ({ ...p, [k]: v }));
  const setS = (k, v) => setStrat(p => ({ ...p, [k]: v }));
  const n = (v) => parseFloat(v);

  const saveConfig = async () => {
    setLoading(true); setErr(null); setOk(null);
    try {
      await api.saveConfig({
        config_name: cfg.config_name,
        battery: {
          battery_capacity_kwh:        n(cfg.battery_capacity_kwh),
          battery_min_reserve:         n(cfg.battery_min_reserve),
          battery_lcos:                n(cfg.battery_lcos),
          battery_max_charge_power:    n(cfg.battery_max_charge_power),
          battery_max_discharge_power: n(cfg.battery_max_discharge_power),
          battery_efficiency:          n(cfg.battery_efficiency),
        },
        inverter: { max_power: n(cfg.inverter_max_power), efficiency: n(cfg.inverter_efficiency) },
        solar: {
          solar_peak_power:  n(cfg.solar_peak_power),
          solar_efficiency:  n(cfg.solar_efficiency),
          solar_tilt:        n(cfg.solar_tilt),
          solar_azimuth:     n(cfg.solar_azimuth),
        },
        grid: { grid_capacity: n(cfg.grid_capacity) },
      });
      setOk("Конфігурацію збережено!");
      onSaved();
    } catch (e) { setErr(e.message); }
    finally { setLoading(false); }
  };

  const saveStrategy = async () => {
    setLoading(true); setErr(null); setOk(null);
    try {
      await api.saveStrategy({
        strategy_name:          strat.strategy_name,
        target_soc:             n(strat.target_soc),
        max_soc:                n(strat.max_soc),
        min_solar_threshold:    n(strat.min_solar_threshold),
        high_solar_threshold:   n(strat.high_solar_threshold),
        solar_surplus_priority: strat.solar_surplus_priority,
        night_discharge:        strat.night_discharge,
        night_sell:             strat.night_sell,
        allow_grid_charging:    strat.allow_grid_charging,
        outage_reserve:         n(strat.outage_reserve),
      });
      setOk("Стратегію збережено!");
      onSaved();
    } catch (e) { setErr(e.message); }
    finally { setLoading(false); }
  };

  const F = ({ label, k, state, set, type = "number", step, min, max, full }) => (
    <div className={`form-field${full ? " full" : ""}`}>
      <label>{label}</label>
      <input type={type} step={step} min={min} max={max}
        value={state[k]} onChange={e => set(k, e.target.value)} />
    </div>
  );

  const Toggle = ({ label, k }) => (
    <div className="toggle-row">
      <span className="toggle-label">{label}</span>
      <label className="toggle">
        <input type="checkbox" checked={strat[k]} onChange={e => setS(k, e.target.checked)} />
        <span className="toggle-slider" />
      </label>
    </div>
  );

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-head">
          <span className="modal-title">Налаштування</span>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="modal-tabs">
            <button className={`modal-tab${tab==="config"?" active":""}`} onClick={() => { setTab("config"); setErr(null); setOk(null); }}>Конфігурація системи</button>
            <button className={`modal-tab${tab==="strategy"?" active":""}`} onClick={() => { setTab("strategy"); setErr(null); setOk(null); }}>Стратегія</button>
          </div>

          {err && <div className="err" style={{marginBottom:12}}>⚠ {err}</div>}
          {ok  && <div className="ok-msg" style={{marginBottom:12}}>{ok}</div>}

          {tab === "config" && (
            <>
              <div className="form-grid">
                <F label="Назва конфігурації" k="config_name" state={cfg} set={setC} type="text" full />
              </div>
              <div className="form-section">Батарея</div>
              <div className="form-grid">
                <F label="Ємність (кВт·год)"    k="battery_capacity_kwh"        state={cfg} set={setC} step="1" min="0" />
                <F label="Мін. резерв (%)"       k="battery_min_reserve"         state={cfg} set={setC} step="1" min="0" max="100" />
                <F label="LCOS (UAH/кВт·год)"   k="battery_lcos"                state={cfg} set={setC} step="0.1" min="0" />
                <F label="КПД (0–1)"             k="battery_efficiency"          state={cfg} set={setC} step="0.01" min="0" max="1" />
                <F label="Макс. заряд (кВт)"     k="battery_max_charge_power"    state={cfg} set={setC} step="1" min="0" />
                <F label="Макс. розряд (кВт)"    k="battery_max_discharge_power" state={cfg} set={setC} step="1" min="0" />
              </div>
              <div className="form-section">Інвертор</div>
              <div className="form-grid">
                <F label="Макс. потужність (кВт)" k="inverter_max_power"    state={cfg} set={setC} step="1" min="0" />
                <F label="КПД (0–1)"               k="inverter_efficiency"  state={cfg} set={setC} step="0.01" min="0" max="1" />
              </div>
              <div className="form-section">Сонячні панелі</div>
              <div className="form-grid">
                <F label="Пікова потужність (кВт)" k="solar_peak_power"  state={cfg} set={setC} step="1" min="0" />
                <F label="КПД (0–1)"                k="solar_efficiency"  state={cfg} set={setC} step="0.01" min="0" max="1" />
                <F label="Нахил (°)"                k="solar_tilt"        state={cfg} set={setC} step="1" min="0" max="90" />
                <F label="Азимут (°)"               k="solar_azimuth"     state={cfg} set={setC} step="1" min="-180" max="180" />
              </div>
              <div className="form-section">Мережа</div>
              <div className="form-grid">
                <F label="Ємність мережі (кВт)" k="grid_capacity" state={cfg} set={setC} step="1" min="0" />
              </div>
            </>
          )}

          {tab === "strategy" && (
            <>
              <div className="form-grid">
                <F label="Назва стратегії" k="strategy_name" state={strat} set={setS} type="text" full />
              </div>
              <div className="form-section">Параметри SoC</div>
              <div className="form-grid">
                <F label="Цільовий SoC (0–1)"  k="target_soc"     state={strat} set={setS} step="0.05" min="0" max="1" />
                <F label="Макс. SoC (0–1)"      k="max_soc"        state={strat} set={setS} step="0.05" min="0" max="1" />
                <F label="Резерв при відключенні (0–1)" k="outage_reserve" state={strat} set={setS} step="0.05" min="0" max="1" />
              </div>
              <div className="form-section">Сонячна енергія</div>
              <div className="form-grid">
                <F label="Мін. поріг GTI (Вт/м²)"   k="min_solar_threshold"  state={strat} set={setS} step="1" min="0" />
                <F label="Висок. поріг GTI (Вт/м²)"  k="high_solar_threshold" state={strat} set={setS} step="10" min="0" />
                <div className="form-field full">
                  <label>Пріоритет надлишку сонця</label>
                  <select value={strat.solar_surplus_priority} onChange={e => setS("solar_surplus_priority", e.target.value)}>
                    <option value="charge_first">Спочатку заряд батареї</option>
                    <option value="sell_first">Спочатку продаж в мережу</option>
                  </select>
                </div>
              </div>
              <div className="form-section">Режими роботи</div>
              <Toggle label="Розряд батареї вночі" k="night_discharge" />
              <Toggle label="Продаж в мережу вночі" k="night_sell" />
              <Toggle label="Заряд від мережі (якщо SoC < цілі)" k="allow_grid_charging" />
            </>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Скасувати</button>
          <button className="btn-primary" disabled={loading}
            onClick={tab === "config" ? saveConfig : saveStrategy}>
            {loading ? "Збереження..." : "Зберегти"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Sidebar ─────────────────────────────────────────────────────────
function Sidebar({ mode, setMode, configs, strategies, histDates, configName, setConfigName, strategyName, setStrategyName, histDate, setHistDate, initialSoc, setInitialSoc, onRun, loading, onUpload, fileName, onRefreshConfigs, onRefreshStrategies, onRefreshHistDates, onLogout, onOpenSettings }) {
  const fileRef  = useRef();
  const isCompare = mode === "compare";
  const runLabel = {
    sac:     "Запустити SAC модель",
    default: "Запустити Default стратегію",
    compare: "Порівняти SAC vs Default",
    history: "Завантажити з БД",
  }[mode];
  const canRun = !!configName && (mode === "sac" || mode === "history" || (mode !== "sac" && !!strategyName)) && !loading;

  return (
    <aside className="sidebar">
      <div className="tabs">
        {[["sac","SAC"],["default","Default"],["compare","Порівняти"],["history","Історія"]].map(([m,l]) => (
          <button key={m} className={`tab${mode===m?" active":""}`} onClick={() => setMode(m)}>{l}</button>
        ))}
      </div>

      <div className="sec-title">Конфігурація</div>
      <SelField label="Назва конфігурації" value={configName} onChange={setConfigName}
        options={configs} placeholder="— оберіть або введіть —"
        onRefresh={onRefreshConfigs} editable />

      {(mode === "default" || mode === "compare") && (
        <>
          <div className="sec-title">Стратегія</div>
          <SelField label="Назва стратегії" value={strategyName} onChange={setStrategyName}
            options={strategies} placeholder="— оберіть або введіть —"
            onRefresh={onRefreshStrategies} editable />
        </>
      )}

      {mode === "history" && (
        <>
          <div className="sec-title">Дата прогнозу</div>
          <SelField label="Дата" value={histDate} onChange={setHistDate}
            options={histDates} placeholder="— остання —"
            onRefresh={onRefreshHistDates} />
        </>
      )}

      {mode !== "history" && (
        <>
          <div className="sec-title">Параметри запуску</div>
          <div className="field">
            <label>Початковий SoC (0–1)</label>
            <input type="number" step="0.05" min="0" max="1"
              value={initialSoc} onChange={e => setInitialSoc(e.target.value)}
              placeholder="авто (з БД)" />
          </div>
        </>
      )}

      <div style={{marginTop:12, display:"flex", flexDirection:"column", gap:8}}>
        <button className="btn btn-run" onClick={onRun} disabled={!canRun}>
          {loading
            ? <><div className="spin" style={{width:13,height:13,borderWidth:2}}/>Завантаження...</>
            : <><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>{runLabel}</>
          }
        </button>

        {!isCompare && (
          <>
            <div className="divider">
              <div className="divider-line"/><span className="divider-text">або</span><div className="divider-line"/>
            </div>
            <input ref={fileRef} type="file" accept=".csv" onChange={onUpload} style={{display:"none"}}/>
            <div className={`upload-box${fileName?" ok":""}`} onClick={() => fileRef.current?.click()}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={fileName?"#15803d":"#94a3b8"} strokeWidth="2" style={{margin:"0 auto",display:"block"}}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <p>{fileName || "Завантажити dispatch_plan.csv"}</p>
            </div>
          </>
        )}
      </div>

      <div style={{marginTop:"auto", paddingTop:16, borderTop:"1px solid #f1f5f9", display:"flex", flexDirection:"column", gap:6}}>
        <button className="btn btn-ghost" onClick={onOpenSettings} style={{display:"flex",alignItems:"center",gap:6}}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          Налаштування
        </button>
        <button className="btn btn-ghost" onClick={onLogout}>Вийти</button>
      </div>
    </aside>
  );
}

// ─── KPIs ────────────────────────────────────────────────────────────
function Kpis({ summary }) {
  const { total_money_earned: earned, economic_savings_uah: savings, sold_kwh: sold, bought_kwh: bought, solar_kwh: solar, unmet_load_kwh: unmet, lcos_total_uah: lcos, initial_soc: iSoc, final_soc: fSoc, steps } = summary;
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
      <K color={earned>=0?"#16a34a":"#dc2626"} label="Cash Flow"
         value={earned} unit=" UAH" sub={`${days} дн · ${steps} кроків`} />
      <K color={(savings??0)>=0?"#16a34a":"#dc2626"} label="Економія (vs мережа)"
         value={savings??0} unit=" UAH" sub="vs grid-only baseline" />
      <K color="#1d4ed8" label="Продано / Куплено"
         value={sold} unit=" кВт·год" sub={`куплено: ${fmt(bought)} кВт·год`} />
      <K color="#d97706" label="Сонячна генерація"
         value={solar} unit=" кВт·год" sub={`деградація: ${fmt(lcos)} UAH`} />
      <K color={unmet>0?"#dc2626":"#16a34a"} label="Непокрите навант."
         value={unmet} unit=" кВт·год" sub={`SoC: ${fmt(iSoc*100,0)}% → ${fmt(fSoc*100,0)}%`} />
    </div>
  );
}

// ─── Charts (SoC + Solar + Grid) ────────────────────────────────────
function Charts({ plan }) {
  const data = plan.map((r, i) => ({
    t:    buildTime(r, i),
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
            <span><span style={{color:"#d97706"}}>‒‒</span> Ціль</span>
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
              {grid}<XAxis {...xP}/><YAxis {...yP} domain={[0,100]} unit="%"/>
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
              {grid}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <Area type="monotone" dataKey="sol" name="Сонце" stroke="#d97706" fill="url(#gSol)" strokeWidth={2} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid */}
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
              {grid}<XAxis {...xP}/><YAxis {...yP}/>
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

// ─── Flow Charts ─────────────────────────────────────────────────────
function FlowCharts({ plan }) {
  const data = plan.map((r, i) => {
    const solar = +(r.solar_kwh ?? r.solar_gen_kwh ?? 0);
    const s2l   = +(r.solar_to_load_kwh    ?? 0).toFixed(4);
    const s2b   = +(r.solar_to_battery_kwh ?? 0).toFixed(4);
    const s2g   = +(r.solar_to_grid_kwh    ?? 0).toFixed(4);
    const curt  = +Math.max(0, solar - s2l - s2b - s2g).toFixed(4);
    return {
      t:    buildTime(r, i),
      s2l, s2b, s2g, curt,
      b2l:  +(r.battery_to_load_kwh ?? 0).toFixed(4),
      g2l:  +(r.grid_to_load_kwh    ?? 0).toFixed(4),
      unmet:+(r.unmet_load_kwh      ?? 0).toFixed(4),
    };
  });
  const ticks = data.filter((_, i) => i % 8 === 0).map(d => d.t);
  const xP = { dataKey:"t", ticks, tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" } };
  const yP = { tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" }, width:38 };
  const grd = <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>;

  return (
    <div className="charts-grid">
      {/* Where solar goes */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Куди йде сонячна енергія</span>
          <div className="legend-row">
            <span><span style={{color:"#f59e0b"}}>■</span> →Навант.</span>
            <span><span style={{color:"#f97316"}}>■</span> →Батарея</span>
            <span><span style={{color:"#10b981"}}>■</span> →Мережа</span>
            <span><span style={{color:"#e2e8f0"}}>■</span> Обрізано</span>
          </div>
        </div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <Area stackId="s" type="monotone" dataKey="s2l"  name="Сонце→Навант." stroke="#f59e0b" fill="#f59e0b" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="s" type="monotone" dataKey="s2b"  name="Сонце→Батарея" stroke="#f97316" fill="#f97316" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="s" type="monotone" dataKey="s2g"  name="Сонце→Мережа"  stroke="#10b981" fill="#10b981" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="s" type="monotone" dataKey="curt" name="Обрізано"       stroke="#e2e8f0" fill="#e2e8f0" fillOpacity={.6}  strokeWidth={0} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* How load is covered */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Покриття навантаження</span>
          <div className="legend-row">
            <span><span style={{color:"#f59e0b"}}>■</span> Сонце</span>
            <span><span style={{color:"#3b82f6"}}>■</span> Батарея</span>
            <span><span style={{color:"#ef4444"}}>■</span> Мережа</span>
            <span><span style={{color:"#7f1d1d"}}>■</span> Непокрито</span>
          </div>
        </div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <Area stackId="l" type="monotone" dataKey="s2l"  name="Сонце→Навант." stroke="#f59e0b" fill="#f59e0b" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="l" type="monotone" dataKey="b2l"  name="Батарея→Навант." stroke="#3b82f6" fill="#3b82f6" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="l" type="monotone" dataKey="g2l"  name="Мережа→Навант." stroke="#ef4444" fill="#ef4444" fillOpacity={.85} strokeWidth={0} dot={false}/>
              <Area stackId="l" type="monotone" dataKey="unmet" name="Непокрито"      stroke="#7f1d1d" fill="#7f1d1d" fillOpacity={.85} strokeWidth={0} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─── Battery + P&L Charts ────────────────────────────────────────────
function BatteryPnlCharts({ plan }) {
  let cum = 0;
  const data = plan.map((r, i) => {
    cum += r.money_earned_ts ?? 0;
    return {
      t:    buildTime(r, i),
      batt: +(r.battery_kwh ?? 0).toFixed(4),
      pnl:  +cum.toFixed(2),
    };
  });
  const ticks = data.filter((_, i) => i % 8 === 0).map(d => d.t);
  const xP = { dataKey:"t", ticks, tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" } };
  const yP = { tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" }, width:46 };
  const grd = <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>;
  const finalPnl = data[data.length - 1]?.pnl ?? 0;

  return (
    <div className="charts-grid">
      {/* Battery flow */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Потік батареї (кВт·год)</span>
          <div className="legend-row">
            <span><span style={{color:"#3b82f6"}}>■</span> Заряд (+)</span>
            <span><span style={{color:"#f59e0b"}}>■</span> Розряд (−)</span>
          </div>
        </div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#e2e8f0"/>
              <Bar dataKey="batt" name="Батарея" radius={[2,2,0,0]}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.batt > 0.001 ? "#3b82f6" : entry.batt < -0.001 ? "#f59e0b" : "#e2e8f0"}/>
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Cumulative P&L */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Накопичений P&L (UAH)</span>
          <div className="legend-row">
            <span style={{color: finalPnl >= 0 ? "#16a34a" : "#dc2626", fontWeight:700}}>
              {finalPnl >= 0 ? "▲" : "▼"} {fmt(finalPnl)} UAH підсумок
            </span>
          </div>
        </div>
        <div className="card-body" style={{height:180}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{top:4,right:6,bottom:0,left:0}}>
              <defs>
                <linearGradient id="gPnl" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={finalPnl>=0?"#16a34a":"#dc2626"} stopOpacity={.15}/>
                  <stop offset="95%" stopColor={finalPnl>=0?"#16a34a":"#dc2626"} stopOpacity={0}/>
                </linearGradient>
              </defs>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3"/>
              <Area type="monotone" dataKey="pnl" name="Накоп. P&L"
                stroke={finalPnl>=0?"#16a34a":"#dc2626"} fill="url(#gPnl)" strokeWidth={2} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─── Compare KPIs ────────────────────────────────────────────────────
function CompareKpis({ sac, def }) {
  const rows = [
    {
      group: "Фінанси",
      items: [
        { label: "Cash flow (за день)", sacV: sac.total_money_earned, defV: def.total_money_earned, unit: " UAH", higherBetter: true, d: 0 },
        { label: "Економія vs мережа",  sacV: sac.economic_savings_uah ?? null, defV: def.economic_savings_uah ?? null, unit: " UAH", higherBetter: true, d: 0 },
        { label: "LCOS (деградація)",   sacV: sac.lcos_total_uah, defV: def.lcos_total_uah, unit: " UAH", higherBetter: false, d: 1 },
      ],
    },
    {
      group: "Енергія",
      items: [
        { label: "Куплено з мережі",   sacV: sac.bought_kwh, defV: def.bought_kwh, unit: " кВт·год", higherBetter: false, d: 1 },
        { label: "Продано в мережу",   sacV: sac.sold_kwh,   defV: def.sold_kwh,   unit: " кВт·год", higherBetter: true, d: 1 },
        { label: "Сонячна генерація",  sacV: sac.solar_kwh,  defV: def.solar_kwh,  unit: " кВт·год", higherBetter: null, d: 1 },
        { label: "Непокрите навант.",  sacV: sac.unmet_load_kwh, defV: def.unmet_load_kwh, unit: " кВт·год", higherBetter: false, d: 3 },
      ],
    },
    {
      group: "SoC",
      items: [
        { label: "Початковий SoC",  sacV: sac.initial_soc * 100, defV: def.initial_soc * 100, unit: "%", higherBetter: null, d: 1 },
        { label: "Кінцевий SoC",    sacV: sac.final_soc   * 100, defV: def.final_soc   * 100, unit: "%", higherBetter: true, d: 1 },
      ],
    },
  ];

  const Delta = ({ sacV, defV, higherBetter }) => {
    if (sacV == null || defV == null) return <span className="delta-ne">—</span>;
    const d = sacV - defV;
    if (Math.abs(d) < 0.001) return <span className="delta-ne">≈ 0</span>;
    const sacWins = higherBetter === true ? d > 0 : higherBetter === false ? d < 0 : null;
    const cls = sacWins === true ? "delta-up" : sacWins === false ? "delta-dn" : "delta-ne";
    const arrow = sacWins === true ? "▲" : sacWins === false ? "▼" : "";
    return <span className={cls}>{arrow} {d > 0 ? "+" : ""}{fmtK(d)}</span>;
  };

  return (
    <div className="cmp-wrap">
      <div className="cmp-head-row">
        <div className="cmp-head-lbl">Метрика</div>
        <div className="cmp-head-lbl"><span className="tag tag-sac">SAC (RL)</span></div>
        <div className="cmp-head-lbl"><span className="tag tag-def">Default</span></div>
        <div className="cmp-head-lbl">SAC delta</div>
      </div>
      {rows.map(({ group, items }) => (
        <div key={group}>
          <div className="cmp-section-head">{group}</div>
          {items.map(({ label, sacV, defV, unit, higherBetter, d }) => (
            <div className="cmp-row" key={label}>
              <div className="cmp-label">{label}</div>
              <div className="cmp-val" style={{ color: "#1d4ed8" }}>
                {sacV != null ? fmt(sacV, d) + unit : "—"}
              </div>
              <div className="cmp-val" style={{ color: "#be185d" }}>
                {defV != null ? fmt(defV, d) + unit : "—"}
              </div>
              <div className="cmp-val"><Delta sacV={sacV} defV={defV} higherBetter={higherBetter}/></div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ─── Compare Charts ──────────────────────────────────────────────────
function CompareCharts({ sacPlan, defPlan }) {
  const n = Math.min(sacPlan.length, defPlan.length);

  // Merge step data
  let sacCum = 0, defCum = 0;
  const merged = Array.from({ length: n }, (_, i) => {
    const s = sacPlan[i], d = defPlan[i];
    sacCum += s.money_earned_ts ?? 0;
    defCum += d.money_earned_ts ?? 0;
    return {
      t:       buildTime(s, i),
      sacSoc:  +((s.soc ?? 0) * 100).toFixed(1),
      defSoc:  +((d.soc ?? 0) * 100).toFixed(1),
      sacTgt:  +((s.target_soc ?? 0) * 100).toFixed(1),
      defTgt:  +((d.target_soc ?? 0) * 100).toFixed(1),
      sacPnl:  +sacCum.toFixed(2),
      defPnl:  +defCum.toFixed(2),
      sacGrid: +(s.grid_kwh ?? 0).toFixed(3),
      defGrid: +(d.grid_kwh ?? 0).toFixed(3),
    };
  });

  const ticks = merged.filter((_, i) => i % 8 === 0).map(d => d.t);
  const xP = { dataKey:"t", ticks, tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" } };
  const yP = { tick:{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" }, width:42 };
  const grd = <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>;
  const sacFinal = merged[merged.length - 1]?.sacPnl ?? 0;
  const defFinal = merged[merged.length - 1]?.defPnl ?? 0;

  // Summary bar data
  const barData = [
    { name: "Куплено, кВт·год", sac: +(sacPlan.filter(r => r.grid_kwh > 0).reduce((a,r) => a+r.grid_kwh,0)).toFixed(1), def: +(defPlan.filter(r => r.grid_kwh > 0).reduce((a,r) => a+r.grid_kwh,0)).toFixed(1) },
    { name: "Продано, кВт·год", sac: +(sacPlan.filter(r => r.grid_kwh < 0).reduce((a,r) => a+Math.abs(r.grid_kwh),0)).toFixed(1), def: +(defPlan.filter(r => r.grid_kwh < 0).reduce((a,r) => a+Math.abs(r.grid_kwh),0)).toFixed(1) },
    { name: "Непокрито, кВт·год", sac: +(sacPlan.reduce((a,r) => a+(r.unmet_load_kwh??0),0)).toFixed(2), def: +(defPlan.reduce((a,r) => a+(r.unmet_load_kwh??0),0)).toFixed(2) },
  ];

  return (
    <div className="charts-grid">
      {/* SoC comparison */}
      <div className="card chart-full">
        <div className="card-head">
          <span className="card-title">Порівняння SoC: SAC vs Default</span>
          <div className="legend-row">
            <span><span style={{color:"#1d4ed8"}}>●</span> SAC SoC</span>
            <span><span style={{color:"#1d4ed8",opacity:.4}}>‒‒</span> SAC ціль</span>
            <span><span style={{color:"#be185d"}}>●</span> Default SoC</span>
            <span><span style={{color:"#be185d",opacity:.4}}>‒‒</span> Default ціль</span>
          </div>
        </div>
        <div className="card-body" style={{height:210}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={merged} margin={{top:4,right:6,bottom:0,left:0}}>
              <defs>
                <linearGradient id="gSacSoc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#1d4ed8" stopOpacity={.12}/>
                  <stop offset="95%" stopColor="#1d4ed8" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="gDefSoc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#be185d" stopOpacity={.08}/>
                  <stop offset="95%" stopColor="#be185d" stopOpacity={0}/>
                </linearGradient>
              </defs>
              {grd}<XAxis {...xP}/><YAxis {...yP} domain={[0,100]} unit="%"/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={20} stroke="#dc2626" strokeDasharray="4 3" strokeWidth={1}/>
              <ReferenceLine y={80} stroke="#d97706" strokeDasharray="4 3" strokeWidth={1}/>
              <Area type="monotone" dataKey="sacSoc" name="SAC SoC"    stroke="#1d4ed8" fill="url(#gSacSoc)" strokeWidth={2} dot={false}/>
              <Area type="monotone" dataKey="defSoc" name="Default SoC" stroke="#be185d" fill="url(#gDefSoc)" strokeWidth={2} dot={false}/>
              <Line type="monotone" dataKey="sacTgt" name="SAC ціль"    stroke="#1d4ed8" strokeWidth={1} strokeDasharray="5 3" dot={false} strokeOpacity={.5}/>
              <Line type="monotone" dataKey="defTgt" name="Default ціль" stroke="#be185d" strokeWidth={1} strokeDasharray="5 3" dot={false} strokeOpacity={.5}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Cumulative P&L comparison */}
      <div className="card chart-full">
        <div className="card-head">
          <span className="card-title">Накопичений P&L: SAC vs Default (UAH)</span>
          <div className="legend-row">
            <span style={{color:"#1d4ed8",fontWeight:700}}>SAC: {fmt(sacFinal)} UAH</span>
            <span style={{color:"#be185d",fontWeight:700}}>Default: {fmt(defFinal)} UAH</span>
            <span style={{color:sacFinal-defFinal>=0?"#16a34a":"#dc2626",fontWeight:700}}>
              Delta: {sacFinal-defFinal>=0?"+":""}{fmt(sacFinal-defFinal)} UAH
            </span>
          </div>
        </div>
        <div className="card-body" style={{height:200}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={merged} margin={{top:4,right:6,bottom:0,left:0}}>
              <defs>
                <linearGradient id="gSacPnl" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#1d4ed8" stopOpacity={.1}/>
                  <stop offset="95%" stopColor="#1d4ed8" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="gDefPnl" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#be185d" stopOpacity={.07}/>
                  <stop offset="95%" stopColor="#be185d" stopOpacity={0}/>
                </linearGradient>
              </defs>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3"/>
              <Area type="monotone" dataKey="sacPnl" name="SAC P&L"    stroke="#1d4ed8" fill="url(#gSacPnl)" strokeWidth={2.5} dot={false}/>
              <Area type="monotone" dataKey="defPnl" name="Default P&L" stroke="#be185d" fill="url(#gDefPnl)" strokeWidth={2} dot={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid exchange SAC */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Мережа — SAC</span>
          <div className="legend-row">
            <span><span style={{color:"#dc2626"}}>■</span> Купівля</span>
            <span><span style={{color:"#16a34a"}}>■</span> Продаж</span>
          </div>
        </div>
        <div className="card-body" style={{height:170}}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={merged} margin={{top:4,right:6,bottom:0,left:0}}>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#e2e8f0"/>
              <Bar dataKey="sacGrid" name="SAC мережа" radius={[2,2,0,0]}>
                {merged.map((e, i) => (
                  <Cell key={i} fill={e.sacGrid < -0.001 ? "#16a34a" : e.sacGrid > 0.001 ? "#dc2626" : "#e2e8f0"}/>
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid exchange Default */}
      <div className="card">
        <div className="card-head">
          <span className="card-title">Мережа — Default</span>
          <div className="legend-row">
            <span><span style={{color:"#f87171"}}>■</span> Купівля</span>
            <span><span style={{color:"#6ee7b7"}}>■</span> Продаж</span>
          </div>
        </div>
        <div className="card-body" style={{height:170}}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={merged} margin={{top:4,right:6,bottom:0,left:0}}>
              {grd}<XAxis {...xP}/><YAxis {...yP}/>
              <Tooltip content={<CT/>}/>
              <ReferenceLine y={0} stroke="#e2e8f0"/>
              <Bar dataKey="defGrid" name="Default мережа" radius={[2,2,0,0]}>
                {merged.map((e, i) => (
                  <Cell key={i} fill={e.defGrid < -0.001 ? "#6ee7b7" : e.defGrid > 0.001 ? "#f87171" : "#e2e8f0"}/>
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary bar comparison */}
      <div className="card chart-full">
        <div className="card-head">
          <span className="card-title">Підсумкові метрики (кВт·год)</span>
          <div className="legend-row">
            <span><span style={{color:"#1d4ed8"}}>■</span> SAC</span>
            <span><span style={{color:"#be185d"}}>■</span> Default</span>
          </div>
        </div>
        <div className="card-body" style={{height:160}}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} layout="vertical" margin={{top:4,right:20,bottom:0,left:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false}/>
              <XAxis type="number" tick={{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#94a3b8" }}/>
              <YAxis type="category" dataKey="name" tick={{ fontSize:10, fontFamily:"JetBrains Mono,monospace", fill:"#64748b" }} width={120}/>
              <Tooltip content={<CT/>}/>
              <Bar dataKey="sac" name="SAC"     fill="#1d4ed8" radius={[0,3,3,0]} barSize={12}/>
              <Bar dataKey="def" name="Default" fill="#be185d" radius={[0,3,3,0]} barSize={12}/>
            </BarChart>
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
  const [sacResult,    setSacResult]    = useState(null);
  const [defResult,    setDefResult]    = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [fileName,     setFileName]     = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  const loadConfigs    = useCallback(() =>
    api.listConfigs().then(d => { const ns = d.map(c => c.config_name); setConfigs(ns); if (ns.length && !configName) setConfigName(ns[0]); }).catch(()=>{}),
    [configName]);
  const loadStrategies = useCallback(() =>
    api.listStrategies().then(d => { const ns = d.map(s => s.strategy_name); setStrategies(ns); if (ns.length && !strategyName) setStrategyName(ns[0]); }).catch(()=>{}),
    [strategyName]);
  const loadHistDates  = useCallback(() => {
    if (!configName) return;
    api.getHistoryDates(configName).then(dates => { setHistDates(dates); if (dates.length) setHistDate(d => d || dates[0]); }).catch(()=>{});
  }, [configName]);

  useEffect(() => { if (authed) { loadConfigs(); loadStrategies(); } }, [authed]);
  useEffect(() => { if (authed && mode === "history") loadHistDates(); }, [authed, mode, configName]);

  const handleLogin  = useCallback(() => setAuthed(true), []);
  const handleLogout = useCallback(() => { api.clearToken(); setAuthed(false); setPlan(null); setSummary(null); setSacResult(null); setDefResult(null); }, []);

  const handleRun = useCallback(async () => {
    setLoading(true); setError(null);
    setPlan(null); setSummary(null); setSacResult(null); setDefResult(null);
    try {
      const soc = initialSoc !== "" ? parseFloat(initialSoc) : undefined;
      if (mode === "compare") {
        const [sacData, defData] = await Promise.all([
          api.getPredictions(configName, soc),
          api.getDefaultPredictions(configName, strategyName, soc),
        ]);
        setSacResult(sacData);
        setDefResult(defData);
      } else {
        let result;
        if      (mode === "sac")     result = await api.getPredictions(configName, soc);
        else if (mode === "default") result = await api.getDefaultPredictions(configName, strategyName, soc);
        else                         result = await api.getHistory(configName, histDate || undefined);
        setPlan(result.dispatch_plan);
        setSummary(result.summary);
        setFileName(null);
      }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
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
      } catch (err) { setError("Не вдалося прочитати CSV: " + err.message); }
    };
    reader.readAsText(file);
  }, []);

  if (!authed) return <Login onLogin={handleLogin}/>;

  const modeLabel = { sac:"SAC", default:"Default", compare:"Порівняння", history:"Історія" }[mode];
  const hasCompare = mode === "compare" && sacResult && defResult;
  const hasSingle  = mode !== "compare" && plan && summary;

  return (
    <>
      <style>{CSS}</style>
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSaved={() => { loadConfigs(); loadStrategies(); }}
        />
      )}
      <div className="app">
        <header className="topbar">
          <div className="logo-wrap">
            <div className="logo-icon">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
            </div>
            <div>
              <div className="brand">EMS Dashboard</div>
              <div className="brand-sub">Energy Management · RL Agent</div>
            </div>
          </div>
          <div className="topbar-right">
            {loading && <div className="spin" style={{width:16,height:16,borderWidth:2}}/>}
            {(hasSingle || hasCompare) && !loading && (
              <div className="chip chip-blue">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                {hasSingle ? `${plan.length} кроків · ${modeLabel}` : `SAC vs Default · ${sacResult.dispatch_plan.length} кроків`}
              </div>
            )}
            <div className="chip chip-green"><div className="dot-green"/>Ready</div>
          </div>
        </header>

        <div className="layout">
          <Sidebar
            mode={mode} setMode={setMode}
            configs={configs} strategies={strategies} histDates={histDates}
            configName={configName}     setConfigName={setConfigName}
            strategyName={strategyName} setStrategyName={setStrategyName}
            histDate={histDate}         setHistDate={setHistDate}
            initialSoc={initialSoc}     setInitialSoc={setInitialSoc}
            onRun={handleRun} loading={loading}
            onUpload={handleUpload} fileName={fileName}
            onRefreshConfigs={loadConfigs}
            onRefreshStrategies={loadStrategies}
            onRefreshHistDates={loadHistDates}
            onLogout={handleLogout}
            onOpenSettings={() => setShowSettings(true)}
          />

          <main className="content">
            {error && <div className="err">⚠ {error}</div>}

            {loading && (
              <div className="loader">
                <div className="spin"/>
                <span style={{fontSize:13,color:"#64748b"}}>Завантаження{mode==="compare"?" SAC та Default...":"..."}</span>
              </div>
            )}

            {!loading && !hasSingle && !hasCompare && (
              <div className="empty">
                <div className="empty-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                </div>
                <h3>Немає даних</h3>
                <p>
                  {mode === "compare"
                    ? "Оберіть конфігурацію та стратегію, потім натисни «Порівняти SAC vs Default»"
                    : "Оберіть конфігурацію та натисни кнопку запуску або завантаж dispatch_plan.csv"
                  }
                </p>
              </div>
            )}

            {!loading && hasCompare && (
              <>
                <CompareKpis sac={sacResult.summary} def={defResult.summary}/>
                <CompareCharts sacPlan={sacResult.dispatch_plan} defPlan={defResult.dispatch_plan}/>
              </>
            )}

            {!loading && hasSingle && (
              <>
                <Kpis summary={summary}/>
                <Charts plan={plan}/>
                <FlowCharts plan={plan}/>
                <BatteryPnlCharts plan={plan}/>
                <Table plan={plan}/>
              </>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
