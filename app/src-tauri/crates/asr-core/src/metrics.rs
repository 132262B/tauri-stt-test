//! 세션 추론 메트릭 (docs/02-architecture.md H). RTF·지연을 run_session 이 기록하고,
//! 자원 모니터 task(app_lib)가 CPU/RAM(sysinfo)과 합쳐 metrics_update 로 emit 한다.

use std::sync::{Arc, Mutex};

/// run_session 과 자원 모니터 task 가 공유하는 메트릭(스레드 안전).
#[derive(Clone, Default)]
pub struct SessionMetrics {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    rtf: f32,
    latencies_ms: Vec<f32>,
}

/// snapshot() 결과.
pub struct MetricsCompute {
    pub rtf: f32,
    pub latency_p50: f32,
    pub latency_p95: f32,
}

impl SessionMetrics {
    /// 1회 추론의 소요시간/오디오길이를 기록(RTF 는 EMA, 지연은 최근 100개 유지).
    pub fn record_iter(&self, infer_ms: f32, audio_sec: f32) {
        if let Ok(mut g) = self.inner.lock() {
            if audio_sec > 0.0 {
                let inst = (infer_ms / 1000.0) / audio_sec;
                g.rtf = if g.rtf == 0.0 {
                    inst
                } else {
                    0.7 * g.rtf + 0.3 * inst
                };
            }
            g.latencies_ms.push(infer_ms);
            let len = g.latencies_ms.len();
            if len > 100 {
                g.latencies_ms.drain(0..len - 100);
            }
        }
    }

    pub fn snapshot(&self) -> MetricsCompute {
        let g = self.inner.lock().expect("metrics lock");
        let mut lat = g.latencies_ms.clone();
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |p: f32| -> f32 {
            if lat.is_empty() {
                0.0
            } else {
                let idx = ((p * (lat.len() as f32 - 1.0)).round() as usize).min(lat.len() - 1);
                lat[idx]
            }
        };
        MetricsCompute {
            rtf: g.rtf,
            latency_p50: pct(0.5),
            latency_p95: pct(0.95),
        }
    }
}
