/**
 * Hosted AI pool — Groq + OpenRouter + Google Gemini, rotated.
 *
 * Nothing runs on a CPU any more. Every request goes to a hosted model and walks through ALL the
 * configured keys: when a key hits a rate limit / quota / outage it rests for a while and the next
 * key (of any provider) answers instead, so a script, a plan or a picture check is always delivered.
 *
 *  - Text: Groq, Gemini and OpenRouter keys are used round-robin, spreading the load.
 *  - Video editing (task "editor_*"): Gemini first; Groq / OpenRouter models are used only as
 *    fallbacks, and for pictures only when the model can really see (vision-capable).
 *  - Model names are DISCOVERED from each provider's live model list (names change and old ones
 *    are retired — Groq retired its Llama 3.x models in August 2026), with env-configurable
 *    preferences. A model that answers "not found" is skipped for a few hours.
 *  - GitHub runners never hold the keys: they use `createRelay()` which asks the app server (that
 *    owns the keys) to run the request. The server uses the same pool directly.
 *
 * The class keeps the interface every caller already uses: attempts(), hasKeys, hasVision,
 * maxWaitMs, lastErrors, exhausted.
 */
export type Provider = 'gemini' | 'groq' | 'openrouter' | 'relay' | (string & {});
export interface WebSource { title: string; uri: string }
export interface LlmRequest {
  system: string;
  user: string;
  saferUser?: string;
  temperature?: number;
  maxTokens?: number;
  json?: boolean;
  timeoutMs?: number;
  /** Research: the answer is grounded on live search results. */
  webSearch?: boolean;
  /** Pictures to look at (vision-capable models only). */
  images?: { mime: string; data: string }[];
  /** Label of the job ("editor_plan", "script", …). "editor_*" means Gemini first. */
  task?: string;
}
export interface LlmAttempt { provider: Provider; model: string; text: string; ms: number; sources?: WebSource[] }
export interface RelayAnswer { text: string; provider?: string; model?: string; sources?: WebSource[] }
export interface LlmConfig {
  geminiKeys?: string[]; groqKeys?: string[]; openrouterKeys?: string[];
  /** A signed-in person's OWN keys (any provider in OWN_PROVIDERS). They are tried before the shared keys. */
  userKeys?: { provider: string; key: string; /** Models the person picked for this key, best first. Empty = the app chooses. */ models?: string[] }[];
  /** Preferred model ids, best first (discovery fills in the rest). */
  geminiModels?: string[]; groqModels?: string[]; openrouterModels?: string[];
  /** Models known to accept pictures (Groq has no capability flag in its list). */
  groqVisionModels?: string[];
  /** Runner side: ask the app server (which owns the keys) to run the request. */
  relay?: (req: LlmRequest) => Promise<string | RelayAnswer>;
  fetchImpl?: typeof fetch;
  log?: (msg: string) => void;
  seed?: number;
  /** Legacy fields (ignored). */
  localUrl?: string; visionUrl?: string; model?: string; geminiBase?: string; groqBase?: string;
}

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';
const GROQ_BASE = 'https://api.groq.com/openai/v1';
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

/**
 * Providers a person can add their own key for (Settings > AI integration).
 * style: how the API is spoken. Most providers copy OpenAI's chat/completions format.
 */
export interface OwnProvider { id: string; label: string; style: 'gemini' | 'openai' | 'anthropic'; base: string }
export const OWN_PROVIDERS: Record<string, OwnProvider> = {
  gemini: { id: 'gemini', label: 'Google Gemini (AI Studio)', style: 'gemini', base: GEMINI_BASE },
  openrouter: { id: 'openrouter', label: 'OpenRouter', style: 'openai', base: OPENROUTER_BASE },
  xai: { id: 'xai', label: 'Grok (xAI)', style: 'openai', base: 'https://api.x.ai/v1' },
  groq: { id: 'groq', label: 'Groq', style: 'openai', base: GROQ_BASE },
  openai: { id: 'openai', label: 'OpenAI', style: 'openai', base: 'https://api.openai.com/v1' },
  anthropic: { id: 'anthropic', label: 'Claude (Anthropic)', style: 'anthropic', base: 'https://api.anthropic.com/v1' },
  deepseek: { id: 'deepseek', label: 'DeepSeek', style: 'openai', base: 'https://api.deepseek.com' },
  mistral: { id: 'mistral', label: 'Mistral', style: 'openai', base: 'https://api.mistral.ai/v1' },
  together: { id: 'together', label: 'Together AI', style: 'openai', base: 'https://api.together.xyz/v1' },
  cerebras: { id: 'cerebras', label: 'Cerebras', style: 'openai', base: 'https://api.cerebras.ai/v1' },
  fireworks: { id: 'fireworks', label: 'Fireworks AI', style: 'openai', base: 'https://api.fireworks.ai/inference/v1' },
};

/** Guess the provider from the look of a key. null = cannot tell (the person picks it). */
export function detectProvider(raw: string): string | null {
  const k = cleanKey(raw);
  if (/^AIza[0-9A-Za-z_-]{20,}$/.test(k) || /^AQ\.[0-9A-Za-z_.-]{20,}$/.test(k)) return 'gemini';
  if (/^sk-or-/i.test(k)) return 'openrouter';
  if (/^xai-/i.test(k)) return 'xai';
  if (/^gsk_/i.test(k)) return 'groq';
  if (/^sk-ant-/i.test(k)) return 'anthropic';
  if (/^csk-/i.test(k)) return 'cerebras';
  if (/^fw_/i.test(k)) return 'fireworks';
  if (/^sk-[a-f0-9]{32}$/i.test(k)) return 'deepseek';
  if (/^sk-/i.test(k)) return 'openai';
  return null;
}

/** Keys arrive pasted from chats / dashboards: trim quotes, stray dots and whitespace. */
export function cleanKey(k: string): string { return String(k || '').trim().replace(/^["'`.]+|["'`,;]+$/g, '').trim(); }
export function splitKeys(...chunks: (string | string[] | undefined)[]): string[] {
  const out: string[] = [];
  for (const c of chunks) for (const part of (Array.isArray(c) ? c : String(c || '').split(/[\s,;]+/))) { const k = cleanKey(part); if (k.length > 8 && !out.includes(k)) out.push(k); }
  return out;
}
const mask = (k: string) => `…${k.slice(-4)}`;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Shared health state (one per process / isolate): which keys and models are resting.
// ---------------------------------------------------------------------------
const keyRest = new Map<string, number>();      // `${provider}:${key}` → resting until (ms)
const modelRest = new Map<string, number>();    // `${provider}:${model}` → resting until (ms)
let cursor = 0;                                 // round-robin position, advances every request
const modelCache = new Map<string, { at: number; list: ModelInfo[] }>();
interface ModelInfo { id: string; vision: boolean }

const restKey = (p: string, k: string, ms: number) => keyRest.set(`${p}:${k}`, Date.now() + ms);
const keyReady = (p: string, k: string) => (keyRest.get(`${p}:${k}`) || 0) <= Date.now();
const restModel = (p: string, m: string, ms: number) => modelRest.set(`${p}:${m}`, Date.now() + ms);
const modelReady = (p: string, m: string) => (modelRest.get(`${p}:${m}`) || 0) <= Date.now();

/** What kind of failure was it? → how long the key / model rests. */
function classify(status: number, body: string, headers?: Headers): { kind: 'rate' | 'quota' | 'auth' | 'model' | 'server' | 'format' | 'other'; restMs: number } {
  const b = String(body || '');
  const retry = Number(headers?.get('retry-after')) || 0;
  // A model that is restricted to certain apps (OpenRouter 403 "only available on…") is the MODEL's problem, not the key's.
  if (/only available|not available (on|for|in)|agentic|harness|not allowed for this model|does not have access to (this )?model|model.*(restricted|gated)/i.test(b) && status !== 401) return { kind: 'model', restMs: 24 * 3600_000 };
  if (status === 401 || status === 403 || /API_KEY_INVALID|invalid api key|ACCESS_TOKEN_TYPE_UNSUPPORTED|permission_denied|api key not valid|has been blocked/i.test(b) && status < 500) {
    return { kind: 'auth', restMs: 6 * 3600_000 };
  }
  if (status === 413 || /request too large|reduce your message size|context.?length|maximum context|too many tokens/i.test(b)) return { kind: 'model', restMs: 20 * 60_000 };
  if (status === 429 || /RESOURCE_EXHAUSTED|rate.?limit|quota|too many requests|tokens per (minute|day)|requests per (minute|day)/i.test(b)) {
    const daily = /per day|daily|TPD|RPD|billing|exceeded your current quota/i.test(b) && !/per minute|TPM|RPM/i.test(b);
    return { kind: daily ? 'quota' : 'rate', restMs: daily ? 60 * 60_000 : Math.min(5 * 60_000, Math.max(20_000, retry * 1000 || 45_000)) };
  }
  if (status === 402) return { kind: 'quota', restMs: 3 * 3600_000 };
  if (status === 404 || /model.*(not found|does not exist|decommission|deprecat|retired|no longer)|no endpoints found|unknown model|not supported for generateContent/i.test(b)) return { kind: 'model', restMs: 6 * 3600_000 };
  if (status === 400 && /response_format|json_object|json mode|responseMimeType|reasoning_effort|unsupported parameter/i.test(b)) return { kind: 'format', restMs: 0 };
  if (status >= 500 || status === 408 || status === 0) return { kind: 'server', restMs: 30_000 };
  return { kind: 'other', restMs: 0 };
}

/** For callers that talk to a provider themselves (speech-to-text, music): healthy keys first, rotated. */
export function healthyFirst(provider: string, keys: string[]): string[] {
  const n = cursor++;
  const rot = keys.length ? [...keys.slice(n % keys.length), ...keys.slice(0, n % keys.length)] : [];
  return [...rot.filter((k) => keyReady(provider, k)), ...rot.filter((k) => !keyReady(provider, k))];
}
/** Tell the pool a key just failed so every caller rests it (rate limit, quota, bad key, outage). */
export function reportKeyFailure(provider: string, key: string, status: number, body = '', headers?: Headers): string {
  const c = classify(status, body, headers);
  if (c.restMs && c.kind !== 'model') restKey(provider, key, c.restMs);
  return c.kind;
}

export const stripThink = (s: string): string => String(s || '').replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------
const NOT_CHAT = /whisper|guard|safeguard|tts|orpheus|playai|embed|moderation|imagen|veo|lyria|image|live|audio|robotics|aqa|computer-use|native-audio|transcribe|rerank/i;
function versionOf(id: string): number { const m = id.match(/(\d+(?:\.\d+)?)/); return m ? parseFloat(m[1]) : 0; }

function rankGemini(a: string, b: string): number {
  const tier = (id: string) => (/hydrogen/.test(id) ? 4 : /flash-lite/.test(id) ? 1 : /flash/.test(id) ? 3 : /pro|argon|carbon/.test(id) ? 2 : 0);
  return tier(b) - tier(a) || versionOf(b) - versionOf(a) || (/preview|exp/.test(a) ? 1 : 0) - (/preview|exp/.test(b) ? 1 : 0);
}
function rankGroq(a: string, b: string): number {
  const pref = [/gpt-oss-120b/, /qwen.*(3\.[5-9]|3\.\d{2})/, /gpt-oss-20b/, /llama-3\.3-70b/, /llama-4/, /qwen/, /llama/];
  const s = (id: string) => { const i = pref.findIndex((r) => r.test(id)); return i < 0 ? 99 : i; };
  return s(a) - s(b);
}

async function discover(provider: 'gemini' | 'groq' | 'openrouter', key: string, f: typeof fetch, preferred: string[], groqVision: string[]): Promise<ModelInfo[]> {
  const ck = provider;
  const cached = modelCache.get(ck);
  if (cached && Date.now() - cached.at < 6 * 3600_000) return cached.list;
  let list: ModelInfo[] = [];
  try {
    if (provider === 'gemini') {
      const r = await f(`${GEMINI_BASE}/models?pageSize=200`, { headers: { 'x-goog-api-key': key }, signal: AbortSignal.timeout(15000) });
      if (r.ok) {
        const j: any = await r.json();
        list = (j.models || [])
          .filter((m: any) => (m.supportedGenerationMethods || []).includes('generateContent'))
          .map((m: any) => String(m.name || '').replace(/^models\//, ''))
          .filter((id: string) => /^gemini/.test(id) && !NOT_CHAT.test(id) && /flash|pro/.test(id))
          .sort(rankGemini).map((id: string) => ({ id, vision: true }));
      }
    } else if (provider === 'groq') {
      const r = await f(`${GROQ_BASE}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(15000) });
      if (r.ok) {
        const j: any = await r.json();
        list = (j.data || []).map((m: any) => String(m.id || '')).filter((id: string) => id && !NOT_CHAT.test(id) && !/compound/.test(id))
          .sort(rankGroq).map((id: string) => ({ id, vision: groqVision.includes(id) || /vision|llama-4|scout|maverick|-vl\b|vl-/i.test(id) }));
      }
    } else {
      const r = await f(`${OPENROUTER_BASE}/models`, { signal: AbortSignal.timeout(20000) });
      if (r.ok) {
        const j: any = await r.json();
        list = (j.data || [])
          .filter((m: any) => String(m.id || '').endsWith(':free') && !NOT_CHAT.test(String(m.id)) && Number(m.context_length || 0) >= 16000 && (m.architecture?.output_modalities || ['text']).includes('text'))
          .map((m: any) => ({ id: String(m.id), vision: (m.architecture?.input_modalities || []).includes('image'), ctx: Number(m.context_length || 0) }))
          .sort((a: any, b: any) => b.ctx - a.ctx || versionOf(b.id) - versionOf(a.id))
          .map((m: any) => ({ id: m.id, vision: m.vision }));
      }
    }
  } catch { /* use the preferences below */ }
  // Preferred names first (kept even when discovery failed), then everything discovered.
  const seen = new Set<string>();
  const out: ModelInfo[] = [];
  const known = new Map(list.map((m) => [m.id, m]));
  for (const id of preferred) if (!seen.has(id) && (!list.length || known.has(id))) { seen.add(id); out.push(known.get(id) || { id, vision: provider === 'gemini' || groqVision.includes(id) }); }
  for (const m of list) if (!seen.has(m.id)) { seen.add(m.id); out.push(m); }
  if (out.length) modelCache.set(ck, { at: Date.now(), list: out });
  return out;
}

/** Models for a person's own key: read from THEIR provider account (what their key can really use). */
const OWN_FAST = /fast|mini|flash|haiku|turbo|small|lite|nano|instant|8b|20b|27b/i;
const OWN_SKIP = /embed|moderation|whisper|tts|transcribe|audio|realtime|image|imagine|video|dall|sora|rerank|guard|speech|instruct-\d|search|codex|computer-use|vision-only|ocr|robotics|aqa|veo|imagen|lyria|live|native-audio/i;
const OWN_KEEP: Record<string, RegExp> = {
  xai: /^grok/i, openai: /^(gpt-|o\d|chatgpt)/i, anthropic: /^claude/i, deepseek: /^deepseek/i, mistral: /^(mistral|ministral|magistral|open-mistral|pixtral)/i,
};
const OWN_DEFAULTS: Record<string, string[]> = {
  gemini: ['gemini-3.8-flash', 'gemini-3.5-flash-lite'], openai: ['gpt-4o-mini'], anthropic: ['claude-haiku-4-5'], deepseek: ['deepseek-chat'], mistral: ['mistral-small-latest'],
};
export function ownVision(provider: string, id: string): boolean {
  if (provider === 'gemini') return true;
  if (provider === 'anthropic') return true;
  if (provider === 'openai') return /^(gpt-4o|gpt-4\.1|gpt-4-turbo|gpt-5|o[134])/i.test(id);
  if (provider === 'xai') return /vision|grok-4/i.test(id);
  return /vision|-vl\b|vl-|llama-4|pixtral|scout|maverick/i.test(id);
}
const ownCache = new Map<string, { at: number; list: ModelInfo[] }>();
async function discoverOwn(provider: string, key: string, f: typeof fetch): Promise<ModelInfo[]> {
  const ck = `${provider}:${key.slice(-10)}`;
  const hit = ownCache.get(ck);
  if (hit && Date.now() - hit.at < 6 * 3600_000) return hit.list;
  const info = OWN_PROVIDERS[provider];
  if (!info) return [];
  let ids: string[] = [];
  try {
    if (info.style === 'gemini') {
      const r = await f(`${GEMINI_BASE}/models?pageSize=200`, { headers: { 'x-goog-api-key': key }, signal: AbortSignal.timeout(15000) });
      if (r.ok) {
        const j: any = await r.json();
        ids = (j.models || []).filter((m: any) => (m.supportedGenerationMethods || []).includes('generateContent')).map((m: any) => String(m.name || '').replace(/^models\//, '')).filter((id: string) => /^(gemini|gemma)/.test(id) && !NOT_CHAT.test(id) && /flash|pro|lite|hydrogen|argon|carbon|gemma/.test(id)).sort(rankGemini);
      }
    } else {
      const headers: Record<string, string> = info.style === 'anthropic' ? { 'x-api-key': key, 'anthropic-version': '2023-06-01' } : { Authorization: `Bearer ${key}` };
      const r = await f(`${info.base}/models${info.style === 'anthropic' ? '?limit=100' : ''}`, { headers, signal: AbortSignal.timeout(20000) });
      if (r.ok) {
        const j: any = await r.json();
        const keep = OWN_KEEP[provider];
        let list: { id: string; ctx: number }[] = (j.data || []).map((m: any) => ({ id: String(m.id || ''), ctx: Number(m.context_length || 0) })).filter((m: any) => m.id && !OWN_SKIP.test(m.id) && (!keep || keep.test(m.id)));
        if (provider === 'openrouter') list = list.filter((m) => m.ctx >= 16000);
        if (provider === 'groq') list = list.filter((m) => !/compound/.test(m.id));
        ids = list.sort((a, b) => (OWN_FAST.test(a.id) ? 0 : 1) - (OWN_FAST.test(b.id) ? 0 : 1) || versionOf(b.id) - versionOf(a.id) || b.ctx - a.ctx).map((m) => m.id);
      }
    }
  } catch { /* the defaults are used */ }
  const out = ids.map((id) => ({ id, vision: ownVision(provider, id) }));
  if (out.length) ownCache.set(ck, { at: Date.now(), list: out });
  return out;
}

/**
 * Which providers have a free tier, and how to tell its models apart:
 *  flag  = the provider's own model list says which models cost nothing
 *  probe = no flag exists, so each candidate gets one tiny live request with the person's key
 *  all   = the free plan allows every model (with lower limits)
 *  none  = no free tier
 */
export type FreeSupport = 'flag' | 'probe' | 'all' | 'none';
export const OWN_FREE: Record<string, FreeSupport> = {
  openrouter: 'flag', together: 'flag', gemini: 'probe', groq: 'all', cerebras: 'all', mistral: 'all',
  xai: 'none', openai: 'none', anthropic: 'none', deepseek: 'none', fireworks: 'none',
};

/**
 * How strong a FREE model is likely to be for coding / chat, from what its id and context size say (a heuristic: providers publish no quality score).
 * Bigger parameter counts, known strong families, coder variants and long context rank up; tiny / mini / lite variants rank down.
 */
export function freeStrength(id: string, ctx = 0): number {
  const s = String(id).toLowerCase();
  const sizes = [...s.matchAll(/(\d+(?:\.\d+)?)b(?![a-z])/g)].map((m) => parseFloat(m[1])).filter((n) => n > 0 && n < 5000);
  let n = Math.min(sizes.length ? Math.max(...sizes) : 0, 500) / 5;                       // up to 100 for 500B
  if (/gpt-oss-120b|qwen3-coder|qwen3-next|qwen3-235|qwen3\.[5-9]|deepseek-(r1|v3|v4)|kimi-k[2-9]|glm-(4\.[5-9]|[5-9])|minimax-m|llama-3\.3-70b|llama-4|mistral-(large|medium)|devstral|nemotron|gemini-(2\.5|3|4)|gemma-[34]-(27|31)/.test(s)) n += 40;
  if (/coder|devstral|code/.test(s)) n += 12;
  if (/thinking|reason|r1/.test(s)) n += 6;
  n += Math.min(ctx, 262144) / 8192;                                                         // up to 32 for 256K context
  if (/(^|[-/:_.])(mini|small|nano|lite|tiny|1b|3b|4b|1\.5b)(?![a-z0-9])/.test(s)) n -= 25;
  if (/vision|-vl\b|ocr|embed|audio|image/.test(s)) n -= 10;
  return n;
}

/** The models a person can choose from for one of their keys: the free ones or every one (paid). */
export async function listOwnModelsFor(provider: string, key: string, f: typeof fetch, tier: 'free' | 'paid'): Promise<{ models: { id: string; vision: boolean }[]; note?: string }> {
  const info = OWN_PROVIDERS[provider];
  if (!info) return { models: [] };
  const all = await discoverOwn(provider, key, f);
  const cap = (list: { id: string; vision: boolean }[]) => list.slice(0, 80);
  if (tier === 'paid') return { models: cap(all) };
  const support = OWN_FREE[provider] || 'none';
  if (support === 'none') return { models: [], note: `${info.label} has no free tier, so only paid models can be used.` };
  if (support === 'flag') {
    try {
      const headers: Record<string, string> = provider === 'together' ? { Authorization: `Bearer ${key}` } : {};
      const r = await f(`${info.base}/models`, { headers, signal: AbortSignal.timeout(20000) });
      if (!r.ok) return { models: [], note: `${info.label} did not return its model list. Try again in a moment.` };
      const j: any = await r.json();
      const rows: any[] = Array.isArray(j) ? j : j.data || [];
      const free = new Set<string>();
      for (const m of rows) {
        const id = String(m?.id || '');
        if (!id || OWN_SKIP.test(id)) continue;
        const pr = m?.pricing || {};
        const zero = (v: any) => v !== undefined && v !== null && v !== '' && Number(v) === 0;
        if (/:free$/.test(id) || (provider === 'openrouter' && zero(pr.prompt) && zero(pr.completion)) || (provider === 'together' && zero(pr.input) && zero(pr.output))) free.add(id);
      }
      const ranked = all.filter((m) => free.has(m.id));
      const extra = [...free].filter((id) => !ranked.some((m) => m.id === id) && (provider !== 'openrouter' || rows.find((m) => m.id === id && Number(m.context_length || 0) >= 16000))).map((id) => ({ id, vision: ownVision(provider, id) }));
      // Best performing first: the first free model is what the app and the Coding Agent use unless the person picks another.
      const ctxOf = new Map<string, number>(rows.map((m: any) => [String(m?.id || ''), Number(m?.context_length || 0)]));
      const merged = [...ranked, ...extra].sort((a, b) => freeStrength(b.id, ctxOf.get(b.id) || 0) - freeStrength(a.id, ctxOf.get(a.id) || 0));
      return { models: cap(merged) };
    } catch { return { models: [], note: `Could not reach ${info.label}. Try again in a moment.` }; }
  }
  // probe / all: these providers do not label free models, so the researched list (FREE_HINTS) plus every
  // other likely candidate is tested live with the person's own key, and only the ones that answer are shown.
  const hints = FREE_HINTS[provider] || [];
  const paidOnly = PAID_ONLY[provider];
  const listed = new Set(all.map((m) => m.id));
  const researched = hints;                                                       // researched free models: always sent a real request, whether or not the provider's model list shows them
  const extra = all.filter((m) => !hints.includes(m.id) && !(paidOnly && paidOnly.test(m.id)) && (support === 'all' || /flash|lite|gemma|hydrogen/i.test(m.id)));
  const candidates = [...researched.map((id) => all.find((m) => m.id === id) || { id, vision: ownVision(provider, id) }), ...extra].slice(0, 12);
  const ck = `${provider}:${key.slice(-10)}:${candidates.map((m) => m.id).join(',')}`;
  const hit = probeCache.get(ck);
  if (hit && Date.now() - hit.at < 10 * 60_000) return { models: hit.models, note: hit.note };
  const details: string[] = [];
  const results = await probeModels(provider, key, f, candidates.map((m) => m.id), details);
  const ok = candidates.filter((m) => results.get(m.id) === 'ok' || results.get(m.id) === 'busy');
  const busy = ok.filter((m) => results.get(m.id) === 'busy').length;
  const unreachable = candidates.length > 0 && candidates.every((m) => results.get(m.id) === 'error');
  if (unreachable && provider === 'gemini' && details.length && details.every((d) => /: 401 /.test(d))) return { models: [], note: 'Google accepted this key for the model list but refuses to let it chat (401). This usually means it is not a Google AI Studio key. Create a new key at aistudio.google.com/apikey (it starts with "AIza"), remove this one, and add the new one.' };
  if (unreachable) return { models: [], note: `${info.label} did not accept the test requests. Reply from ${info.label}: ${details.slice(0, 2).join(' | ') || 'no answer (timed out)'}` };
  const note = `${info.label} does not label free models, so ${candidates.length} candidate${candidates.length === 1 ? ' was' : 's were'} tested with your key; ${ok.length} connected.` + (busy ? ` ${busy} answered "too many requests" right now (free limits), so they may be slow.` : '');
  const out = { models: cap(ok), note: ok.length ? note : `${note} Google said: ${details.slice(0, 2).join(' | ') || 'no details'}` };
  if (ok.length) probeCache.set(ck, { at: Date.now(), ...out });   // failures are never cached, so a retry tests again
  return out;
}

/**
 * Free models, researched (Nov 2026). These are tried FIRST and tested live; the provider's own model list can add more.
 * Gemini: Google's pricing page lists these as free of charge (rate limited). Gemini 4 Argon / Carbon and 3.1 Pro are paid only.
 */
const FREE_HINTS: Record<string, string[]> = {
  gemini: ['gemini-4-hydrogen', 'gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemma-4-31b-it', 'gemma-4-26b-a4b-it', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
  groq: ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
  cerebras: ['gpt-oss-120b', 'llama-3.3-70b', 'llama3.1-8b'],
  mistral: ['mistral-small-latest', 'open-mistral-nemo', 'ministral-8b-latest'],
};
/** Models known to have no free tier: never tested, never shown under Free. */
const PAID_ONLY: Record<string, RegExp> = { gemini: /argon|carbon|3\.1-pro|pro-preview|ultra/i };
const probeCache = new Map<string, { at: number; models: { id: string; vision: boolean }[]; note: string }>();

/** One tiny live request per model (a few tokens). ok = answered; busy = real but rate limited; no = not available on this key / not free; error = could not reach it. */
type ProbeResult = 'ok' | 'busy' | 'no' | 'error';
async function probeModels(provider: string, key: string, f: typeof fetch, ids: string[], details: string[] = []): Promise<Map<string, ProbeResult>> {
  const info = OWN_PROVIDERS[provider];
  const out = new Map<string, ProbeResult>();
  const one = async (id: string): Promise<ProbeResult> => {
    try {
      let r: Response;
      if (info.style === 'gemini') {
        const url = `${GEMINI_BASE}/models/${encodeURIComponent(id)}:generateContent`;
        const body = JSON.stringify({ contents: [{ role: 'user', parts: [{ text: 'Reply with the word OK.' }] }], generationConfig: { maxOutputTokens: 16 } });
        r = await f(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key }, body, signal: AbortSignal.timeout(15000) });
        // Newer "AQ." style keys are sent as a Bearer token (same fallback the key check uses).
        if (r.status === 401 || r.status === 403) r = await f(url, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body, signal: AbortSignal.timeout(15000) });
      } else {
        r = await f(`${info.base}/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify({ model: id, messages: [{ role: 'user', content: 'Reply with the word OK.' }], max_tokens: 16 }), signal: AbortSignal.timeout(15000) });
      }
      if (r.ok) return 'ok';
      const text = await r.text().catch(() => '');
      let why = text;
      try { why = JSON.parse(text)?.error?.message || text; } catch { /* keep raw */ }
      details.push(`${id}: ${r.status} ${String(why).replace(/\s+/g, ' ').slice(0, 160)}`);
      if (r.status === 429) {
        // Google answers 429 with "limit: 0" for a model that has no free quota; other 429s are just a busy free tier.
        return /limit"?\s*:\s*0\b|not available on the free|free_tier.*limit: 0/i.test(text) ? 'no' : 'busy';
      }
      if (r.status === 400 || r.status === 403 || r.status === 404 || r.status === 402) return 'no';
      return 'error';
    } catch (e: any) { details.push(`${id}: ${String(e?.message || e).slice(0, 100)}`); return 'error'; }
  };
  for (let i = 0; i < ids.length; i += 6) {            // 6 at a time: friendly to Workers' connection limit
    const batch = ids.slice(i, i + 6);
    const res = await Promise.all(batch.map(one));
    batch.forEach((id, j) => out.set(id, res[j]));
  }
  return out;
}

/** Fallback model names when discovery is impossible (offline / key rejected). */
const DEFAULTS = {
  gemini: ['gemini-3.8-flash', 'gemini-3.5-flash-lite'],
  groq: ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.6-27b'],
  openrouter: [] as string[],
};

// ---------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------
interface Slot { provider: string; key: string; /** A person's own key: tried first, with its own model list. */ own?: boolean; /** The models the person picked for this key, in the order to try them. */ models?: string[] }
/** Where a person's own key rests models: per provider AND key, so two keys never share a model's cooldown. */
const ownNs = (s: { provider: string; key: string }) => `${s.provider}:own:${s.key.slice(-8)}`;

export class LlmPool {
  cfg: LlmConfig;
  maxWaitMs = 4 * 60 * 1000;
  lastErrors: string[] = [];
  exhausted = new Set<string>();
  private f: typeof fetch;
  constructor(cfg: LlmConfig) { this.cfg = cfg; // Wrapped so `fetch` is never called with the pool as `this` (Cloudflare Workers: "Illegal invocation").
    this.f = ((u: any, i?: any) => (cfg.fetchImpl || fetch)(u, i)) as typeof fetch; }

  private keys(p: 'gemini' | 'groq' | 'openrouter'): string[] { return splitKeys(p === 'gemini' ? this.cfg.geminiKeys : p === 'groq' ? this.cfg.groqKeys : this.cfg.openrouterKeys); }
  private ownKeys(): { provider: string; key: string; models: string[] }[] { return (this.cfg.userKeys || []).map((k) => ({ provider: String(k?.provider || ''), key: cleanKey(k?.key || ''), models: Array.isArray(k?.models) ? k.models.map((m) => String(m || '').trim()).filter(Boolean) : [] })).filter((k) => OWN_PROVIDERS[k.provider] && k.key.length > 8); }
  get hasKeys() { return !!this.cfg.relay || this.ownKeys().length + this.keys('gemini').length + this.keys('groq').length + this.keys('openrouter').length > 0; }
  get hasVision() { return !!this.cfg.relay || this.ownKeys().length > 0 || this.keys('gemini').length > 0 || this.keys('openrouter').length > 0 || this.keys('groq').length > 0; }
  private log(m: string) { this.cfg.log?.(m); }
  /** The distinct recent failures, for error messages. */
  errorSummary(n = 6): string { return [...new Set(this.lastErrors)].slice(-n).join(' | '); }
  private note(m: string) { this.lastErrors.push(m); if (this.lastErrors.length > 40) this.lastErrors.shift(); this.log(m); }

  /** Healthy (provider, key) slots in the order this request should try them. */
  private slots(req: LlmRequest): Slot[] {
    const editor = /^editor_/.test(String(req.task || ''));
    const all: Slot[] = [];
    for (const p of ['gemini', 'groq', 'openrouter'] as const) for (const key of this.keys(p)) all.push({ provider: p, key });
    const usable = all.filter((s) => keyReady(s.provider, s.key));
    const rest = all.filter((s) => !keyReady(s.provider, s.key));
    const spin = (arr: Slot[], n: number) => (arr.length ? [...arr.slice(n % arr.length), ...arr.slice(0, n % arr.length)] : arr);
    const n = cursor++ + (this.cfg.seed || 0);
    let ordered: Slot[];
    if (editor) {
      // Video editing: Gemini first (rotated among its keys); other providers only as a fallback.
      const g = spin(usable.filter((s) => s.provider === 'gemini'), n);
      const others = spin(usable.filter((s) => s.provider !== 'gemini'), n);
      ordered = [...g, ...others];
    } else {
      // Text: spread over all providers, one key of each in turn, so no single key is hammered.
      const by: Record<string, Slot[]> = { groq: spin(usable.filter((s) => s.provider === 'groq'), n), gemini: spin(usable.filter((s) => s.provider === 'gemini'), n), openrouter: spin(usable.filter((s) => s.provider === 'openrouter'), n) };
      const order = spin([{ provider: 'groq' }, { provider: 'gemini' }, { provider: 'openrouter' }] as any[], n).map((x: any) => x.provider as string);
      ordered = [];
      for (let i = 0; ordered.length < usable.length; i++) for (const p of order) if (by[p][i]) ordered.push(by[p][i]);
    }
    if (req.images?.length) {
      // Pictures: providers that can see go first.
      ordered = [...ordered.filter((s) => s.provider === 'gemini'), ...ordered.filter((s) => s.provider !== 'gemini')];
    }
    // The person's OWN keys go first (rotated among themselves); the shared keys are the fallback.
    const own: Slot[] = this.ownKeys().map((k) => ({ provider: k.provider, key: k.key, own: true, models: k.models }));
    const ownUsable = spin(own.filter((s) => keyReady(s.provider, s.key)), n);
    const ownRest = own.filter((s) => !keyReady(s.provider, s.key));
    return [...ownUsable, ...ordered, ...ownRest, ...rest];   // resting keys last: a final chance when everything is resting
  }

  private async modelsFor(s: Slot, req: LlmRequest): Promise<ModelInfo[]> {
    if (s.own && s.models?.length) {
      // The person chose these models: try exactly them, in their order, with this one key.
      let chosen = s.models.map((id) => ({ id, vision: ownVision(s.provider, id) })).filter((m) => modelReady(ownNs(s), m.id));
      if (req.images?.length) chosen = chosen.filter((m) => m.vision);
      return chosen;
    }
    if (s.own) {
      // No models chosen in the settings: FREE models by default (best first) wherever the provider has a free tier. Never spends the person's money unasked on OpenRouter / Together.
      const sup = OWN_FREE[s.provider] || 'none';
      if (sup !== 'none') {
        const fr = (await listOwnModelsFor(s.provider, s.key, this.f, 'free').catch(() => ({ models: [] as ModelInfo[] }))).models;
        let freeOwn = fr.filter((m) => modelReady(ownNs(s), m.id));
        if (req.images?.length) freeOwn = freeOwn.filter((m) => m.vision);
        if (freeOwn.length || sup === 'flag') return freeOwn.slice(0, 3);
      }
      let own = await discoverOwn(s.provider, s.key, this.f);
      if (!own.length) own = (OWN_DEFAULTS[s.provider] || []).map((id) => ({ id, vision: ownVision(s.provider, id) }));
      own = own.filter((m) => modelReady(ownNs(s), m.id));
      if (req.images?.length) own = own.filter((m) => m.vision);
      return own.slice(0, 3);
    }
    const pref = s.provider === 'gemini' ? this.cfg.geminiModels : s.provider === 'groq' ? this.cfg.groqModels : this.cfg.openrouterModels;
    const preferred = splitKeys(pref);   // only what the owner configured; otherwise the live list decides
    let list = await discover(s.provider as 'gemini' | 'groq' | 'openrouter', s.key, this.f, preferred, splitKeys(this.cfg.groqVisionModels));
    if (!list.length) list = DEFAULTS[s.provider as 'gemini' | 'groq' | 'openrouter'].map((id) => ({ id, vision: s.provider === 'gemini' }));
    list = list.filter((m) => modelReady(s.provider, m.id));
    if (req.images?.length) list = list.filter((m) => m.vision);
    return list.slice(0, s.provider === 'openrouter' ? 5 : 3);
  }

  /** Up to three answers (the caller stops when one is good); every failure moves to the next key. */
  async *attempts(req: LlmRequest): AsyncGenerator<LlmAttempt> {
    if (!this.hasKeys) return;
    let sources: WebSource[] = [];
    let user = req.user;
    if (req.webSearch) {
      try {
        const found = await webContext(req.user, this.f);
        sources = found.sources;
        if (found.text) user = `LIVE SEARCH RESULTS (use ONLY these for facts; cite the outlet):\n${found.text}\n\n${req.user}`;
        else this.log('Web research: no search results could be read — answering without sources.');
      } catch (e: any) { this.log(`Web research failed (${e?.message || e}).`); }
    }
    const deadline = Date.now() + Math.min(this.maxWaitMs, Math.max(60_000, (req.timeoutMs || 60_000) * 2.5));
    let answers = 0;

    // ---- Runner: the app server runs the rotation ----
    if (this.cfg.relay) {
      for (let i = 0; i < 3 && Date.now() < deadline; i++) {
        const t0 = Date.now();
        const r = { ...req, user: i > 0 && req.saferUser ? req.saferUser : user, temperature: Math.min(1.1, (req.temperature ?? 0.7) + i * 0.15) };
        try {
          const a = await this.cfg.relay(r);
          const text = stripThink(typeof a === 'string' ? a : a.text);
          if (!text) throw new Error('empty answer');
          const info = typeof a === 'string' ? {} as RelayAnswer : a;
          yield { provider: 'relay', model: `${info.provider || 'ai'}/${info.model || 'hosted'}`, text, ms: Date.now() - t0, sources: info.sources?.length ? info.sources : sources };
        } catch (e: any) {
          this.note(`AI relay: ${String(e?.message || e).slice(0, 200)}`);
          await sleep(2500 * (i + 1));
        }
      }
      return;
    }

    // ---- Server / direct: rotate through every key ----
    let tries = 0;
    const maxTries = 40;
    while (answers < 3 && tries < maxTries && Date.now() < deadline) {
      const slots = this.slots(req);
      let progressed = false;
      for (const s of slots) {
        if (answers >= 3 || tries >= maxTries || Date.now() >= deadline) break;
        const resting = !keyReady(s.provider, s.key);
        if (resting && slots.some((x) => keyReady(x.provider, x.key))) continue;   // healthy keys exist: skip resting ones
        const models = await this.modelsFor(s, req);
        for (const m of models) {
          if (answers >= 3 || tries >= maxTries || Date.now() >= deadline) break;
          tries++;
          const t0 = Date.now();
          const r = { ...req, user: answers > 0 && req.saferUser ? req.saferUser : user, temperature: Math.min(1.1, (req.temperature ?? 0.7) + answers * 0.15) };
          const res = await this.callOne(s, m.id, r);
          if (res.ok) {
            progressed = true;
            const clean = stripThink(res.text);
            if (!clean) { this.note(`${s.provider}/${m.id} ${mask(s.key)}: empty answer`); continue; }
            answers++;
            yield { provider: s.provider, model: m.id, text: clean, ms: Date.now() - t0, sources };
            break;   // the next attempt (if the caller wants one) uses the next key
          }
          this.note(`${s.provider}/${m.id} ${mask(s.key)}: ${res.error}`);
          if (res.kind === 'model') { restModel(s.own ? ownNs(s) : s.provider, m.id, res.restMs); this.exhausted.add(`${s.provider}:${m.id}`); continue; }   // next model, same key
          if (res.kind === 'format') continue;
          // A person's own key with several chosen models: limits, outages and odd answers are per model, so the NEXT chosen
          // model is tried with the same key. Only a rejected key (auth) rests the whole key.
          if (s.own && (s.models?.length || 0) > 1 && res.kind !== 'auth') { if (res.restMs) restModel(ownNs(s), m.id, res.restMs); continue; }
          if (res.restMs) restKey(s.provider, s.key, res.restMs);
          break;   // rate-limit / auth / outage: the next KEY
        }
      }
      if (answers > 0 || tries >= maxTries) break;
      if (!progressed) {
        // Everything is resting or failing: wait for the earliest key to come back (at most 20 s), then go again.
        const soonest = Math.min(...[...keyRest.entries()].map(([, until]) => until).filter((u) => u > Date.now()), Date.now() + 20_000);
        const wait = Math.min(20_000, Math.max(1500, soonest - Date.now()));
        if (Date.now() + wait >= deadline) break;
        this.log(`All AI keys are busy — waiting ${Math.round(wait / 1000)}s for one to free up…`);
        await sleep(wait);
      }
    }
  }

  /** One request to one provider with one key. */
  private async callOne(s: Slot, model: string, req: LlmRequest): Promise<{ ok: true; text: string } | { ok: false; error: string; kind: ReturnType<typeof classify>['kind']; restMs: number }> {
    const timeout = Math.max(15_000, Math.min(req.timeoutMs || 90_000, 180_000));
    const fail = (status: number, body: string, headers?: Headers) => { const c = classify(status, body, headers); return { ok: false as const, error: `HTTP ${status} ${String(body).replace(/\s+/g, ' ').slice(0, 180)}`, kind: c.kind, restMs: c.restMs }; };
    try {
      const style = s.own ? OWN_PROVIDERS[s.provider]?.style : s.provider === 'gemini' ? 'gemini' : 'openai';
      if (style === 'gemini') return await this.callGemini(s.key, model, req, timeout, fail);
      if (style === 'anthropic') return await this.callAnthropic(s.key, model, req, timeout, fail);
      return await this.callOpenAiStyle(s, model, req, timeout, fail);
    } catch (e: any) {
      const msg = String(e?.name === 'TimeoutError' || e?.name === 'AbortError' ? `timed out after ${Math.round(timeout / 1000)}s` : e?.message || e).slice(0, 180);
      return { ok: false, error: msg, kind: 'server', restMs: 20_000 };
    }
  }

  private async callGemini(key: string, model: string, req: LlmRequest, timeout: number, fail: (s: number, b: string, h?: Headers) => any) {
    const parts: any[] = [];
    for (const im of req.images || []) parts.push({ inlineData: { mimeType: im.mime, data: im.data } });
    parts.push({ text: req.user });
    const body: any = {
      systemInstruction: { parts: [{ text: req.system }] },
      contents: [{ role: 'user', parts }],
      generationConfig: { temperature: req.temperature ?? 0.7, maxOutputTokens: Math.max(256, (req.maxTokens || 2048) + 1024), ...(req.json ? { responseMimeType: 'application/json' } : {}), thinkingConfig: /gemini-3/.test(model) ? { thinkingLevel: 'low' } : { thinkingBudget: 0 } },
    };
    const url = `${GEMINI_BASE}/models/${encodeURIComponent(model)}:generateContent`;
    const go = (headers: Record<string, string>) => this.f(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body), signal: AbortSignal.timeout(timeout) });
    let res = await go({ 'x-goog-api-key': key });
    let text = await res.text();
    // New-style "AQ." keys are accepted as a bearer credential on some projects.
    if (res.status === 401 && /ACCESS_TOKEN_TYPE_UNSUPPORTED|invalid authentication/i.test(text)) { res = await go({ Authorization: `Bearer ${key}` }); text = await res.text(); }
    // This model rejects the thinking setting: ask again without it.
    if (res.status === 400 && /thinking/i.test(text)) { delete body.generationConfig.thinkingConfig; res = await go({ 'x-goog-api-key': key }); text = await res.text(); }
    if (!res.ok) return fail(res.status, text, res.headers);
    const j = JSON.parse(text);
    const out = (j?.candidates?.[0]?.content?.parts || []).filter((p: any) => typeof p?.text === 'string' && !p.thought).map((p: any) => p.text).join('');
    if (!out) {
      const why = j?.promptFeedback?.blockReason || j?.candidates?.[0]?.finishReason || 'empty';
      return { ok: false as const, error: `no text (${why})`, kind: 'other' as const, restMs: 0 };
    }
    return { ok: true as const, text: out };
  }

  private async callAnthropic(key: string, model: string, req: LlmRequest, timeout: number, fail: (s: number, b: string, h?: Headers) => any) {
    const content: any = req.images?.length
      ? [...req.images.map((im) => ({ type: 'image', source: { type: 'base64', media_type: im.mime, data: im.data } })), { type: 'text', text: req.user }]
      : req.user;
    const system = req.json ? `${req.system}\n\nAnswer with one JSON object only: no text before or after it, no code fences.` : req.system;
    const body = { model, max_tokens: Math.min(16000, Math.max(256, req.maxTokens || 2048)), temperature: Math.min(1, req.temperature ?? 0.7), system, messages: [{ role: 'user', content }] };
    const res = await this.f(`${OWN_PROVIDERS.anthropic.base}/messages`, { method: 'POST', headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(timeout) });
    const text = await res.text();
    if (!res.ok) return fail(res.status, text, res.headers);
    const j = JSON.parse(text);
    const out = (j?.content || []).filter((p: any) => p?.type === 'text' && typeof p.text === 'string').map((p: any) => p.text).join('');
    if (!out) return { ok: false as const, error: 'empty answer', kind: 'other' as const, restMs: 0 };
    return { ok: true as const, text: out };
  }

  private async callOpenAiStyle(s: Slot, model: string, req: LlmRequest, timeout: number, fail: (s: number, b: string, h?: Headers) => any) {
    const base = s.own ? (OWN_PROVIDERS[s.provider]?.base || OPENROUTER_BASE) : s.provider === 'groq' ? GROQ_BASE : OPENROUTER_BASE;
    // OpenAI's newer models (gpt-5, o-series) take max_completion_tokens and only the default temperature.
    const newOpenAi = s.own && s.provider === 'openai' && /^(o\d|gpt-5)/i.test(model);
    const vision = !!req.images?.length;
    const content: any = vision ? [...req.images!.map((im) => ({ type: 'image_url', image_url: { url: `data:${im.mime};base64,${im.data}` } })), { type: 'text', text: req.user }] : req.user;
    const reasoning = /gpt-oss|qwen3|deepseek-r1|reason/i.test(model);
    const build = (withJson: boolean) => {
      const b: any = { model, messages: [{ role: 'system', content: req.system }, { role: 'user', content }], temperature: req.temperature ?? 0.7, max_tokens: Math.min(s.provider === 'groq' ? 8000 : 16000, (req.maxTokens || 2048) + (reasoning ? 1500 : 0)) };
      if (newOpenAi) { b.max_completion_tokens = Math.max(b.max_tokens, 4000); delete b.max_tokens; delete b.temperature; }
      if (withJson && req.json) b.response_format = { type: 'json_object' };
      if (s.provider === 'groq' && /gpt-oss/.test(model)) b.reasoning_effort = 'low';
      return b;
    };
    const headers: Record<string, string> = { Authorization: `Bearer ${s.key}`, 'Content-Type': 'application/json', ...(s.provider === 'openrouter' ? { 'HTTP-Referer': 'https://animato.studio', 'X-Title': 'Animato Studio' } : {}) };
    let res = await this.f(`${base}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(build(true)), signal: AbortSignal.timeout(timeout) });
    let text = await res.text();
    if (res.status === 400 && /response_format|json_object|json mode|reasoning_effort|unsupported/i.test(text)) {
      const plain = build(false); delete plain.reasoning_effort;
      res = await this.f(`${base}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(plain), signal: AbortSignal.timeout(timeout) });
      text = await res.text();
    }
    if (!res.ok) return fail(res.status, text, res.headers);
    const j = JSON.parse(text);
    if (j?.error) return fail(Number(j.error.code) || 500, JSON.stringify(j.error));   // OpenRouter reports upstream errors with HTTP 200
    const out = String(j?.choices?.[0]?.message?.content || '');
    if (!out) return { ok: false as const, error: 'empty answer', kind: 'other' as const, restMs: 0 };
    return { ok: true as const, text: out };
  }
}

// ---------------------------------------------------------------------------
// Runner side: ask the app server to run a request (the keys never leave the server)
// ---------------------------------------------------------------------------
/**
 * scope: "campaign:<id>" | "editor:<jobId>" | "animate:<jobId>" — whichever the runner was started for.
 * The server checks the runner key for that scope before it spends any AI key.
 */
export function createRelay(opts: { appUrl: string; scope: string; runnerKey: string; fetchImpl?: typeof fetch }) {
  const f = opts.fetchImpl || fetch;
  const base = String(opts.appUrl || '').replace(/\/+$/, '');
  const headers = { 'Content-Type': 'application/json', 'X-Animato-Runner-Key': opts.runnerKey, 'X-Animato-Scope': opts.scope };
  return {
    available: !!(base && opts.runnerKey && opts.scope),
    async llm(req: LlmRequest): Promise<RelayAnswer> {
      const res = await f(`${base}/api/ai/relay/llm`, { method: 'POST', headers, body: JSON.stringify({ req }), signal: AbortSignal.timeout(Math.max(120_000, (req.timeoutMs || 60_000) * 2.5)) });
      const txt = await res.text();
      let j: any = {}; try { j = JSON.parse(txt); } catch { /* not JSON */ }
      if (!res.ok || !j?.success) throw new Error(`HTTP ${res.status} ${String(j?.error || txt).slice(0, 200)}`);
      return { text: String(j.text || ''), provider: j.provider, model: j.model, sources: j.sources };
    },
    /** Speech → text (Groq Whisper on the server). Returns words with timings. */
    async stt(audio: Uint8Array, mime: string, opts2: { language?: string; prompt?: string } = {}): Promise<{ text: string; words: { word: string; start: number; end: number }[] }> {
      const q = new URLSearchParams({ ...(opts2.language ? { language: opts2.language } : {}), ...(opts2.prompt ? { prompt: opts2.prompt.slice(0, 600) } : {}) });
      const res = await f(`${base}/api/ai/relay/stt?${q}`, { method: 'POST', headers: { 'Content-Type': mime || 'audio/wav', 'X-Animato-Runner-Key': opts.runnerKey, 'X-Animato-Scope': opts.scope }, body: audio as any, signal: AbortSignal.timeout(240_000) });
      const j: any = await res.json().catch(() => ({}));
      if (!res.ok || !j?.success) throw new Error(`HTTP ${res.status} ${String(j?.error || '').slice(0, 200)}`);
      return { text: String(j.text || ''), words: Array.isArray(j.words) ? j.words : [] };
    },
    /** The music keys (ACE cloud). Short-lived: only for the song of this run. */
    async musicKeys(): Promise<string[]> {
      const res = await f(`${base}/api/ai/relay/music-keys`, { method: 'POST', headers, body: '{}', signal: AbortSignal.timeout(30_000) });
      const j: any = await res.json().catch(() => ({}));
      if (!res.ok || !j?.success) throw new Error(`HTTP ${res.status} ${String(j?.error || '').slice(0, 200)}`);
      return splitKeys(j.keys);
    },
    /** Picture from Pollinations through the server (no key needed). */
  };
}

/** Public search results + the first readable paragraphs of the top pages (no API, no key). */
export async function webContext(query: string, f: typeof fetch): Promise<{ text: string; sources: WebSource[] }> {
  const q = String(query).replace(/\s+/g, ' ').slice(0, 300);
  const ua = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36' };
  const res = await f(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, { headers: ua, signal: AbortSignal.timeout(15000) });
  const html = await res.text();
  const strip = (s: string) => s.replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();
  const items: { title: string; uri: string; snippet: string }[] = [];
  for (const m of html.matchAll(/<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)) {
    let uri = m[1];
    const u = uri.match(/[?&]uddg=([^&]+)/); if (u) uri = decodeURIComponent(u[1]);
    if (!/^https?:/.test(uri)) continue;
    items.push({ title: strip(m[2]), uri, snippet: strip(m[3]) });
    if (items.length >= 6) break;
  }
  const pages = await Promise.all(items.slice(0, 3).map(async (it) => {
    try {
      const r = await f(it.uri, { headers: ua, signal: AbortSignal.timeout(12000) });
      const h = (await r.text()).replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, ' ');
      const paras = [...h.matchAll(/<p[^>]*>([\s\S]*?)<\/p>/gi)].map((p) => strip(p[1])).filter((p) => p.length > 60);
      return paras.join(' ').slice(0, 1800);
    } catch { return ''; }
  }));
  const text = items.map((it, i) => `[${i + 1}] ${it.title} — ${it.uri}\n${it.snippet}${pages[i] ? `\n${pages[i]}` : ''}`).join('\n\n');
  return { text, sources: items.map((it) => ({ title: it.title, uri: it.uri })) };
}

/** First JSON object in a model answer (tolerates code fences and <think> blocks). */
export function extractJsonObject(text: string): any {
  const cleaned = String(text || '').replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/```json/gi, '```').replace(/```/g, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end <= start) throw new Error('no JSON object in model output');
  return JSON.parse(cleaned.slice(start, end + 1));
}
