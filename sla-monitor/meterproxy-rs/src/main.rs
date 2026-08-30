//! Metering reverse proxy for vLLM / LiteLLM — Rust port of meterproxy.py.
//!
//! Measures, per request, the two rates that matter and cannot be derived from
//! Prometheus histograms (TTFT and prompt length are separate distributions, so
//! their per-request ratio is unavailable):
//!
//!     prefill tok/s    = prompt_tokens     / (t_first_token - t_start)
//!     generation tok/s = completion_tokens / (t_end - t_first_token)
//!
//! Attribution is by API key, which neither vLLM nor a downstream gateway
//! exposes per consumer once the key has been swapped for an upstream one.
//!
//! WIRE COMPATIBILITY IS THE POINT OF THIS PORT. Metric names, label sets and
//! histogram bucket edges are byte-identical to the Python original so that the
//! existing Grafana dashboard keeps working across the swap. The one deliberate
//! difference: `# HELP`/`# TYPE` are emitted once per metric family instead of
//! once per series. Repeating them is malformed exposition that Prometheus
//! tolerates; emitting them once is correct and changes no series identity.
//!
//! Design notes carried over from the original:
//!   * Streaming is forwarded chunk-by-chunk, never buffered, so client TTFT is
//!     unaffected by the proxy.
//!   * Token counts come from the usage chunk. If the client did not ask for it
//!     we inject `stream_options.include_usage` and STRIP the extra usage chunk
//!     back out, so the client contract is unchanged.
//!   * Non-streaming requests have no first-token event, so prefill and
//!     generation cannot be separated; they are recorded as e2e-only.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::any,
    Router,
};
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;

// --- bucket edges -----------------------------------------------------------
// Value AND its exact rendering, because `le` label text defines series
// identity: le="1" and le="1.0" are different series to Prometheus.
const RATE_BUCKETS: &[(f64, &str)] = &[
    (1.0, "1"), (2.0, "2"), (5.0, "5"), (10.0, "10"), (20.0, "20"), (35.0, "35"),
    (50.0, "50"), (75.0, "75"), (100.0, "100"), (150.0, "150"), (200.0, "200"),
    (350.0, "350"), (500.0, "500"), (1000.0, "1000"), (2000.0, "2000"),
    (5000.0, "5000"), (10000.0, "10000"), (20000.0, "20000"),
];
const TIME_BUCKETS: &[(f64, &str)] = &[
    (0.05, "0.05"), (0.1, "0.1"), (0.25, "0.25"), (0.5, "0.5"), (1.0, "1"),
    (2.0, "2"), (3.0, "3"), (5.0, "5"), (8.0, "8"), (12.0, "12"), (20.0, "20"),
    (30.0, "30"), (60.0, "60"), (120.0, "120"), (300.0, "300"),
];

const DASHBOARD: &str = include_str!("dashboard.html");

// --- state ------------------------------------------------------------------

#[derive(Clone)]
struct Rec {
    ts: f64,
    key: String,
    model: String,
    stream: bool,
    prompt_tokens: Option<f64>,
    completion_tokens: Option<f64>,
    ttft: Option<f64>,
    e2e: f64,
    prefill_rate: Option<f64>,
    gen_rate: Option<f64>,
}

#[derive(Default, Clone)]
struct Hist {
    counts: Vec<f64>, // len == edges.len() + 1, last is +Inf
    sum: f64,
    n: u64,
}

impl Hist {
    fn obs(&mut self, edges: &[(f64, &str)], v: f64) {
        if self.counts.is_empty() {
            self.counts = vec![0.0; edges.len() + 1];
        }
        self.sum += v;
        self.n += 1;
        for (i, (b, _)) in edges.iter().enumerate() {
            if v <= *b {
                self.counts[i] += 1.0;
            }
        }
        let last = self.counts.len() - 1;
        self.counts[last] += 1.0;
    }
    fn bucket(&self, i: usize) -> f64 {
        self.counts.get(i).copied().unwrap_or(0.0)
    }
    fn inf(&self, edges: usize) -> f64 {
        self.counts.get(edges).copied().unwrap_or(0.0)
    }
}

#[derive(Default, Clone)]
struct Cum {
    requests: u64,
    prompt_tokens: f64,
    completion_tokens: f64,
    // Time denominators for CUMULATIVE (ratio-of-sums) rates. A mean of
    // per-request rates over-weights small requests; tokens/seconds does not.
    prefill_seconds: f64,
    generation_seconds: f64,
    // Numerators MUST cover exactly the same requests as the denominators
    // above, or non-streaming requests contribute tokens with no measured time
    // and the ratio blows up.
    prefill_tokens_measured: f64,
    generation_tokens_measured: f64,
    prefill: Hist,
    gen: Hist,
    ttft: Hist,
}

struct Inner {
    records: std::collections::VecDeque<Rec>,
    cum: HashMap<(String, String), Cum>,
}

struct App {
    inner: Mutex<Inner>,
    routes: Vec<(String, String)>, // url-prefix -> host:port, longest first
    client_keys: HashMap<String, String>,
    upstream_key: String,
    window_s: f64,
    maxrec: usize,
    http: reqwest::Client,
}

fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl App {
    fn observe(&self, rec: &Rec) {
        let mut g = self.inner.lock();
        let e = g
            .cum
            .entry((rec.key.clone(), rec.model.clone()))
            .or_default();
        e.requests += 1;
        e.prompt_tokens += rec.prompt_tokens.unwrap_or(0.0);
        e.completion_tokens += rec.completion_tokens.unwrap_or(0.0);
        if let (Some(t), Some(p)) = (rec.ttft, rec.prompt_tokens) {
            if t > 0.0 && p > 0.0 {
                e.prefill_seconds += t;
                e.prefill_tokens_measured += p;
            }
        }
        if let (Some(t), Some(c)) = (rec.ttft, rec.completion_tokens) {
            if c > 0.0 {
                let gen_s = rec.e2e - t;
                if gen_s > 0.0 {
                    e.generation_seconds += gen_s;
                    e.generation_tokens_measured += c;
                }
            }
        }
        if let Some(v) = rec.prefill_rate {
            e.prefill.obs(RATE_BUCKETS, v);
        }
        if let Some(v) = rec.gen_rate {
            e.gen.obs(RATE_BUCKETS, v);
        }
        if let Some(v) = rec.ttft {
            e.ttft.obs(TIME_BUCKETS, v);
        }
        if g.records.len() >= self.maxrec {
            g.records.pop_front();
        }
        let r = rec.clone();
        g.records.push_back(r);
    }

    fn pick_route(&self, path: &str) -> Option<(String, String)> {
        for (prefix, target) in &self.routes {
            if path.starts_with(prefix.as_str()) {
                let rest = &path[prefix.len()..];
                let rest = if rest.is_empty() { "/" } else { rest };
                return Some((target.clone(), rest.to_string()));
            }
        }
        None
    }
}

/// Render a float the way Prometheus wants it. Rust's f64 Display is
/// shortest-round-trip and never uses exponent form, which the text format
/// accepts; integral values print without a trailing `.0`.
fn g(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "0".to_string()
    }
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

// --- Prometheus exposition ---------------------------------------------------

fn metrics_text(app: &App) -> String {
    let g_ = app.inner.lock();
    let mut keys: Vec<_> = g_.cum.keys().cloned().collect();
    keys.sort();
    let mut out = String::with_capacity(16 * 1024);

    out.push_str("# HELP llmproxy_requests_total Requests proxied, by api key.\n");
    out.push_str("# TYPE llmproxy_requests_total counter\n");
    for k in &keys {
        let e = &g_.cum[k];
        out.push_str(&format!(
            "llmproxy_requests_total{{key=\"{}\",model=\"{}\"}} {}\n",
            esc(&k.0), esc(&k.1), e.requests
        ));
    }

    let fields: &[(fn(&Cum) -> f64, &str, &str)] = &[
        (|c| c.prompt_tokens, "llmproxy_prompt_tokens_total", "Prompt tokens by api key."),
        (|c| c.completion_tokens, "llmproxy_completion_tokens_total", "Completion tokens by api key."),
        (|c| c.prefill_seconds, "llmproxy_prefill_seconds_total",
         "Summed TTFT seconds. Denominator for cumulative prefill tok/s."),
        (|c| c.generation_seconds, "llmproxy_generation_seconds_total",
         "Summed decode seconds (end-TTFT). Denominator for cumulative generation tok/s."),
        (|c| c.prefill_tokens_measured, "llmproxy_prefill_tokens_measured_total",
         "Prompt tokens from requests that have a measured TTFT. Numerator for cumulative prefill tok/s."),
        (|c| c.generation_tokens_measured, "llmproxy_generation_tokens_measured_total",
         "Completion tokens from requests with measured decode time. Numerator for cumulative generation tok/s."),
    ];
    for (get, metric, help) in fields {
        out.push_str(&format!("# HELP {} {}\n# TYPE {} counter\n", metric, help, metric));
        for k in &keys {
            out.push_str(&format!(
                "{}{{key=\"{}\",model=\"{}\"}} {}\n",
                metric, esc(&k.0), esc(&k.1), g(get(&g_.cum[k]))
            ));
        }
    }

    let hists: &[(fn(&Cum) -> &Hist, &str, &str, &[(f64, &str)])] = &[
        (|c| &c.prefill, "llmproxy_prefill_tokens_per_second",
         "Per-request prompt_tokens/TTFT (includes queue wait).", RATE_BUCKETS),
        (|c| &c.gen, "llmproxy_generation_tokens_per_second",
         "Per-request completion_tokens/(end-TTFT).", RATE_BUCKETS),
        (|c| &c.ttft, "llmproxy_ttft_seconds",
         "Per-request time to first token.", TIME_BUCKETS),
    ];
    for (get, metric, help, edges) in hists {
        out.push_str(&format!("# HELP {} {}\n# TYPE {} histogram\n", metric, help, metric));
        for k in &keys {
            let h = get(&g_.cum[k]);
            let lab = format!("key=\"{}\",model=\"{}\"", esc(&k.0), esc(&k.1));
            for (i, (_, txt)) in edges.iter().enumerate() {
                out.push_str(&format!(
                    "{}_bucket{{{},le=\"{}\"}} {}\n", metric, lab, txt, g(h.bucket(i))
                ));
            }
            out.push_str(&format!(
                "{}_bucket{{{},le=\"+Inf\"}} {}\n", metric, lab, g(h.inf(edges.len()))
            ));
            out.push_str(&format!("{}_sum{{{}}} {}\n", metric, lab, g(h.sum)));
            out.push_str(&format!("{}_count{{{}}} {}\n", metric, lab, h.n));
        }
    }
    out
}

// --- sliding-window dashboard data ------------------------------------------

fn pct(vals: &mut Vec<f64>, q: f64) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if vals.len() == 1 {
        return Some(vals[0]);
    }
    let i = q * (vals.len() - 1) as f64;
    let lo = i.floor() as usize;
    let hi = (lo + 1).min(vals.len() - 1);
    Some(vals[lo] + (vals[hi] - vals[lo]) * (i - lo as f64))
}

fn data_json(app: &App) -> Value {
    let now = now_unix();
    let g_ = app.inner.lock();
    let recent: Vec<&Rec> = g_
        .records
        .iter()
        .filter(|r| now - r.ts <= app.window_s)
        .collect();

    let mut groups: HashMap<(String, String), Vec<&Rec>> = HashMap::new();
    for r in &recent {
        let m = if r.model.is_empty() { "?".to_string() } else { r.model.clone() };
        groups.entry((r.key.clone(), m)).or_default().push(r);
    }
    let mut ordered: Vec<_> = groups.into_iter().collect();
    ordered.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

    let mut rows = Vec::new();
    for ((k, m), rs) in ordered {
        let mut pre: Vec<f64> = rs.iter().filter_map(|r| r.prefill_rate).collect();
        let mut gen: Vec<f64> = rs.iter().filter_map(|r| r.gen_rate).collect();
        let mut tt: Vec<f64> = rs.iter().filter_map(|r| r.ttft).collect();
        let (npre, ngen, ntt) = (pre.len(), gen.len(), tt.len());
        rows.push(serde_json::json!({
            "key": k, "label": "", "model": m,
            "requests": rs.len(),
            "streamed": rs.iter().filter(|r| r.stream).count(),
            "measurable": ngen,
            "prompt_tokens": rs.iter().map(|r| r.prompt_tokens.unwrap_or(0.0)).sum::<f64>(),
            "completion_tokens": rs.iter().map(|r| r.completion_tokens.unwrap_or(0.0)).sum::<f64>(),
            "prefill": {"p5": pct(&mut pre, 0.05), "p50": pct(&mut pre, 0.5),
                        "p95": pct(&mut pre, 0.95), "n": npre},
            "gen": {"p5": pct(&mut gen, 0.05), "p50": pct(&mut gen, 0.5),
                    "p95": pct(&mut gen, 0.95), "n": ngen},
            "ttft": {"p50": pct(&mut tt, 0.5), "p95": pct(&mut tt, 0.95),
                     "p99": pct(&mut tt, 0.99), "n": ntt},
        }));
    }
    let routes: HashMap<&str, &str> = app
        .routes
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    serde_json::json!({
        "window_s": app.window_s, "total": recent.len(), "rows": rows, "routes": routes
    })
}

// --- proxying ----------------------------------------------------------------

fn hop_by_hop_req(name: &HeaderName) -> bool {
    matches!(name.as_str(), "host" | "content-length" | "accept-encoding")
}
fn hop_by_hop_resp(name: &HeaderName) -> bool {
    matches!(name.as_str(), "content-length" | "content-encoding" | "transfer-encoding")
}

/// Truthy in the Python sense: a present, non-empty string.
fn nonempty_str(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::String(s)) if !s.is_empty())
}

async fn handle(
    State(app): State<Arc<App>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();
    match path.as_str() {
        "/metrics" => {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                metrics_text(&app),
            )
                .into_response()
        }
        "/data.json" => {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                data_json(&app).to_string(),
            )
                .into_response()
        }
        "/" | "/stats" => {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                DASHBOARD,
            )
                .into_response()
        }
        _ => {}
    }

    let Some((target, upstream_path)) = app.pick_route(&path) else {
        return (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, "application/json")],
            format!("{{\"error\": \"no route for {}\"}}", esc(&path)),
        )
            .into_response();
    };

    // --- identify the consumer ------------------------------------------------
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let tok = auth.split_whitespace().last().unwrap_or("").to_string();
    let label = app.client_keys.get(&tok).cloned();
    let kid = label.clone().unwrap_or_else(|| {
        if auth.is_empty() { "anon".to_string() } else { short_hash(&tok) }
    });

    // --- inspect / rewrite the request body ----------------------------------
    let mut model = String::new();
    let mut streaming = false;
    let mut injected = false;
    let mut send_body = body.clone();
    if !body.is_empty() {
        if let Ok(mut payload) = serde_json::from_slice::<Value>(&body) {
            if let Some(m) = payload.get("model").and_then(|v| v.as_str()) {
                model = m.to_string();
            }
            streaming = payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            if streaming {
                let already = payload
                    .get("stream_options")
                    .and_then(|s| s.get("include_usage"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if !already {
                    if !payload.get("stream_options").map(|s| s.is_object()).unwrap_or(false) {
                        payload["stream_options"] = serde_json::json!({});
                    }
                    payload["stream_options"]["include_usage"] = Value::Bool(true);
                    if let Ok(b) = serde_json::to_vec(&payload) {
                        send_body = Bytes::from(b);
                        injected = true; // we asked for it; strip it on the way out
                    }
                }
            }
        }
    }

    // --- build the upstream request -------------------------------------------
    let q = uri.query().map(|s| format!("?{}", s)).unwrap_or_default();
    let url = format!("http://{}{}{}", target, upstream_path, q);
    let mut req = app.http.request(method, &url);
    for (name, value) in headers.iter() {
        if !hop_by_hop_req(name) {
            req = req.header(name.clone(), value.clone());
        }
    }
    if label.is_some() && !app.upstream_key.is_empty() {
        // consumer key -> engine key
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {}", app.upstream_key)) {
            req = req.header(header::AUTHORIZATION, v);
        }
    }
    req = req.body(send_body);

    let t0 = Instant::now();
    let up = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            let msg = e.to_string();
            let msg: String = msg.chars().take(200).collect();
            return (
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                format!("{{\"error\": \"upstream: {}\"}}", esc(&msg)),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(up.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out_headers = HeaderMap::new();
    for (name, value) in up.headers().iter() {
        if !hop_by_hop_resp(name) {
            out_headers.insert(name.clone(), value.clone());
        }
    }

    if !streaming {
        let raw = up.bytes().await.unwrap_or_default();
        let mut pt = None;
        let mut ct = None;
        if let Ok(d) = serde_json::from_slice::<Value>(&raw) {
            if let Some(u) = d.get("usage") {
                pt = u.get("prompt_tokens").and_then(|v| v.as_f64());
                ct = u.get("completion_tokens").and_then(|v| v.as_f64());
            }
        }
        let e2e = t0.elapsed().as_secs_f64();
        app.observe(&Rec {
            ts: now_unix(), key: kid, model, stream: false,
            prompt_tokens: pt, completion_tokens: ct,
            ttft: None, e2e, prefill_rate: None, gen_rate: None,
        });
        let mut resp = Response::new(Body::from(raw));
        *resp.status_mut() = status;
        *resp.headers_mut() = out_headers;
        return resp;
    }

    // --- streaming: forward chunk by chunk, never buffer ----------------------
    let app2 = app.clone();
    let stream = async_stream::stream! {
        let mut buf = BytesMut::new();
        let mut ttft: Option<f64> = None;
        let mut pt: Option<f64> = None;
        let mut ct: Option<f64> = None;
        let mut upstream = up.bytes_stream();

        while let Some(item) = upstream.next().await {
            let chunk = match item {
                Ok(c) => c,
                Err(_) => break,
            };
            buf.extend_from_slice(&chunk);
            let mut out = BytesMut::new();
            loop {
                let Some(pos) = buf.iter().position(|b| *b == b'\n') else { break };
                let line = buf.split_to(pos + 1);
                let txt = String::from_utf8_lossy(&line[..line.len() - 1]);
                let txt = txt.trim();
                let mut keep = true;
                if let Some(rest) = txt.strip_prefix("data: ") {
                    if rest != "[DONE]" {
                        if let Ok(d) = serde_json::from_str::<Value>(rest) {
                            let has_choices = d
                                .get("choices")
                                .and_then(|c| c.as_array())
                                .map(|a| !a.is_empty())
                                .unwrap_or(false);
                            if let Some(u) = d.get("usage") {
                                if !u.is_null() {
                                    pt = u.get("prompt_tokens").and_then(|v| v.as_f64());
                                    ct = u.get("completion_tokens").and_then(|v| v.as_f64());
                                    if injected && !has_choices {
                                        keep = false; // client never asked for this
                                    }
                                }
                            }
                            if ttft.is_none() {
                                if let Some(chs) = d.get("choices").and_then(|c| c.as_array()) {
                                    'outer: for c in chs {
                                        if let Some(dl) = c.get("delta") {
                                            if nonempty_str(dl.get("content"))
                                                || nonempty_str(dl.get("reasoning"))
                                                || nonempty_str(dl.get("reasoning_content"))
                                            {
                                                ttft = Some(t0.elapsed().as_secs_f64());
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if keep {
                    out.extend_from_slice(&line);
                }
            }
            if !out.is_empty() {
                yield Ok::<Bytes, std::io::Error>(out.freeze());
            }
        }
        if !buf.is_empty() {
            yield Ok::<Bytes, std::io::Error>(buf.split().freeze());
        }

        let e2e = t0.elapsed().as_secs_f64();
        let prefill_rate = match (pt, ttft) {
            (Some(p), Some(t)) if p > 0.0 && t > 0.0 => Some(p / t),
            _ => None,
        };
        let gen_rate = match (ct, ttft) {
            (Some(c), Some(t)) if c > 0.0 && (e2e - t) > 0.01 => Some(c / (e2e - t)),
            _ => None,
        };
        app2.observe(&Rec {
            ts: now_unix(), key: kid, model, stream: true,
            prompt_tokens: pt, completion_tokens: ct,
            ttft, e2e, prefill_rate, gen_rate,
        });
    };

    let mut resp = Response::new(Body::from_stream(stream));
    *resp.status_mut() = status;
    *resp.headers_mut() = out_headers;
    resp
}

/// First 8 hex chars of sha256(token) — matches the Python `key_id` fallback
/// for keys that are not in CLIENT_KEYS.
fn short_hash(tok: &str) -> String {
    let d = sha256(tok.as_bytes());
    d.iter().take(4).map(|b| format!("{:02x}", b)).collect()
}

// Small dependency-free SHA-256 (FIPS 180-4). Only used for anonymous-key
// labelling, so it is not performance-critical.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut gg, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & gg);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = gg; gg = f; f = e;
            e = d.wrapping_add(t1);
            d = c; c = b; b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, gg, hh].iter().enumerate() {
            h[i] = h[i].wrapping_add(*v);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

// --- boot --------------------------------------------------------------------

fn env_map(name: &str) -> HashMap<String, String> {
    let raw = std::env::var(name).unwrap_or_else(|_| "{}".into());
    serde_json::from_str::<HashMap<String, String>>(&raw).unwrap_or_else(|e| {
        eprintln!("{}: not a JSON object of strings ({}), treating as empty", name, e);
        HashMap::new()
    })
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8091);
    let window_s: f64 = std::env::var("WINDOW_S").ok().and_then(|s| s.parse().ok()).unwrap_or(1800.0);
    let maxrec: usize = std::env::var("MAXREC").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let upstream_key = std::env::var("UPSTREAM_KEY").unwrap_or_default();
    let client_keys = env_map("CLIENT_KEYS");

    // Longest prefix first, so /a/b cannot be shadowed by /a.
    let mut routes: Vec<(String, String)> = env_map("ROUTES").into_iter().collect();
    routes.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));

    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(1800))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(64)
        .build()
        .expect("http client");

    let app = Arc::new(App {
        inner: Mutex::new(Inner {
            records: std::collections::VecDeque::new(),
            cum: HashMap::new(),
        }),
        routes: routes.clone(),
        client_keys,
        upstream_key,
        window_s,
        maxrec,
        http,
    });

    let router = Router::new()
        .route("/", any(handle))
        .route("/*tail", any(handle))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024))
        .with_state(app);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!(
        "metering proxy (rust) on :{} routes={:?} window={}s",
        port, routes, window_s as u64
    );
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, router).await.expect("serve");
}
