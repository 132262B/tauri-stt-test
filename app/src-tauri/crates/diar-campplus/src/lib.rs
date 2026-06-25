//! Rust 온라인 화자 분리 — sherpa-onnx 화자 임베딩(CAM++) + 온라인 클러스터링.
//! Python resemblyzer 대체. 세그먼트 임베딩을 코사인 임계로 활성 트랙에 매칭/신규 생성.

use std::path::Path;

use sherpa_rs::embedding_manager::EmbeddingManager;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use asr_core::diar::Diarizer;

/// sherpa-onnx CAM++ (zh_en, 한·영 적합) 화자 임베딩 모델.
const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";

pub struct OnlineDiarizer {
    extractor: EmbeddingExtractor,
    manager: EmbeddingManager,
    threshold: f32,
    next_id: u32,
}

impl OnlineDiarizer {
    /// model: 화자 임베딩 onnx(CAM++). threshold: 동일 화자 코사인 임계(작을수록 관대).
    pub fn new(model: impl AsRef<Path>, threshold: f32) -> Result<Self, String> {
        let config = ExtractorConfig {
            model: model.as_ref().to_string_lossy().into_owned(),
            ..Default::default()
        };
        let extractor = EmbeddingExtractor::new(config).map_err(|e| format!("화자 임베딩 적재 실패: {e}"))?;
        let manager = EmbeddingManager::new(extractor.embedding_size as i32);
        Ok(Self {
            extractor,
            manager,
            threshold,
            next_id: 0,
        })
    }

    /// models_dir/campplus.onnx 가 없으면 sherpa-onnx 릴리스에서 다운로드 후 적재.
    pub fn with_download(models_dir: impl AsRef<Path>, threshold: f32) -> Result<Self, String> {
        let dir = models_dir.as_ref();
        std::fs::create_dir_all(dir).ok();
        let path = dir.join("campplus.onnx");
        if !path.exists() {
            let resp = ureq::get(MODEL_URL)
                .call()
                .map_err(|e| format!("화자모델 다운로드 실패: {e}"))?;
            let tmp = path.with_extension("part");
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            std::io::copy(&mut resp.into_reader(), &mut f).map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        }
        Self::new(path, threshold)
    }

    /// 16kHz mono 세그먼트 → 화자 트랙 id. 너무 짧으면 None.
    pub fn assign(&mut self, samples: &[f32]) -> Option<u32> {
        if samples.len() < 16000 / 2 {
            return None; // < 0.5s
        }
        let mut emb = self
            .extractor
            .compute_speaker_embedding(samples.to_vec(), 16000)
            .ok()?;
        if let Some(name) = self.manager.search(&emb, self.threshold) {
            return name.parse::<u32>().ok();
        }
        let id = self.next_id;
        self.next_id += 1;
        let _ = self.manager.add(id.to_string(), &mut emb);
        Some(id)
    }
}

impl Diarizer for OnlineDiarizer {
    fn assign(&mut self, samples: &[f32]) -> Option<u32> {
        OnlineDiarizer::assign(self, samples)
    }
}
