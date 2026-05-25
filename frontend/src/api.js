const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export const getToken   = ()  => localStorage.getItem('ems_token');
export const setToken   = (t) => localStorage.setItem('ems_token', t);
export const clearToken = ()  => localStorage.removeItem('ems_token');

const authHdr = () => ({ Authorization: `Bearer ${getToken()}` });

async function handle(res) {
  if (res.status === 401) { clearToken(); window.location.reload(); }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const d    = body.detail;
    if (d && typeof d === 'object' && d.message)
      throw new Error(`${d.message} Доступно після ${d.available_after} (через ${d.retry_after_minutes} хв).`);
    throw new Error(typeof d === 'string' ? d : JSON.stringify(d));
  }
  return res.json();
}

export const login = (username, password) =>
  fetch(`${BASE}/auth/login`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body:    new URLSearchParams({ username, password }),
  }).then(handle);

export const register = (username, email, password) =>
  fetch(`${BASE}/auth/register`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ username, email, password }),
  }).then(handle);

export const listConfigs    = () => fetch(`${BASE}/config/list`,   { headers: authHdr() }).then(handle);
export const listStrategies = () => fetch(`${BASE}/strategy/list`, { headers: authHdr() }).then(handle);

export const getPredictions = (config_name, initial_soc) => {
  const p = new URLSearchParams({ config_name });
  if (initial_soc != null && initial_soc !== '') p.set('initial_soc', initial_soc);
  return fetch(`${BASE}/predictions/?${p}`, { headers: authHdr() }).then(handle);
};

export const getDefaultPredictions = (config_name, strategy_name, initial_soc) => {
  const p = new URLSearchParams({ config_name, strategy_name });
  if (initial_soc != null && initial_soc !== '') p.set('initial_soc', initial_soc);
  return fetch(`${BASE}/predictions/default?${p}`, { headers: authHdr() }).then(handle);
};

export const getHistory = (config_name, date) => {
  const p = new URLSearchParams({ config_name });
  if (date) p.set('date', date);
  return fetch(`${BASE}/predictions/history?${p}`, { headers: authHdr() }).then(handle);
};

export const getHistoryDates = (config_name) => {
  const p = new URLSearchParams({ config_name });
  return fetch(`${BASE}/predictions/history/dates?${p}`, { headers: authHdr() }).then(handle);
};

export const saveConfig = ({ config_name, ...settings }) =>
  fetch(`${BASE}/config/?config_name=${encodeURIComponent(config_name)}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', ...authHdr() },
    body:    JSON.stringify(settings),
  }).then(handle);

export const saveStrategy = (data) =>
  fetch(`${BASE}/strategy/`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', ...authHdr() },
    body:    JSON.stringify(data),
  }).then(handle);
