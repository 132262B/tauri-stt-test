//! 화자분리: pyannote-3.0 segmentation + CAM++ 임베딩 + 글로벌 클러스터링(sherpa-onnx).
//! sherpa-rs 고수준 API 는 num_threads 를 1 로 박아둬서, sherpa_rs_sys 를 직접 호출해
//! **멀티스레드(num_threads=코어수)** 로 돌린다. + int8 segmentation 모델 사용(더 빠름).

use std::ffi::CString;
use std::path::Path;
use std::ptr::null_mut;

use asr_core::diar::{DiarSegment, Diarizer};
use sherpa_rs_sys as sys;

pub struct PyannoteDiarizer {
    sd: *const sys::SherpaOnnxOfflineSpeakerDiarization,
}

// 단일 세션 스레드에서만 사용. C 핸들은 thread-safe 하지 않으나 우리 사용 패턴상 안전.
unsafe impl Send for PyannoteDiarizer {}

impl PyannoteDiarizer {
    pub fn new(
        seg_model: impl AsRef<Path>,
        emb_model: impl AsRef<Path>,
        num_speakers: Option<i32>,
        threshold: f32,
        num_threads: i32,
    ) -> Result<Self, String> {
        let debug = if std::env::var("DIAR_DEBUG").is_ok() {
            1
        } else {
            0
        };
        // CString 들은 Create 호출 동안 살아있어야 함(config 가 포인터를 참조).
        let seg = CString::new(seg_model.as_ref().to_string_lossy().as_bytes())
            .map_err(|e| e.to_string())?;
        let emb = CString::new(emb_model.as_ref().to_string_lossy().as_bytes())
            .map_err(|e| e.to_string())?;
        let provider = CString::new("cpu").unwrap();

        let config = sys::SherpaOnnxOfflineSpeakerDiarizationConfig {
            embedding: sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                model: emb.as_ptr(),
                num_threads,
                debug,
                provider: provider.as_ptr(),
            },
            clustering: sys::SherpaOnnxFastClusteringConfig {
                num_clusters: num_speakers.unwrap_or(-1),
                threshold,
            },
            min_duration_off: 0.5,
            min_duration_on: 0.3,
            segmentation: sys::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                pyannote: sys::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                    model: seg.as_ptr(),
                },
                num_threads,
                debug,
                provider: provider.as_ptr(),
            },
        };
        let sd = unsafe { sys::SherpaOnnxCreateOfflineSpeakerDiarization(&config) };
        if sd.is_null() {
            return Err("화자분리 초기화 실패".into());
        }
        Ok(Self { sd })
    }

    /// models_dir 기준 기본 경로 + int8 segmentation + 코어수 스레드.
    pub fn with_paths(
        models_dir: impl AsRef<Path>,
        num_speakers: Option<i32>,
    ) -> Result<Self, String> {
        let d = models_dir.as_ref();
        // int8 segmentation(있으면) 우선 → 더 빠름.
        let seg_dir = d.join("diar/sherpa-onnx-pyannote-segmentation-3-0");
        let seg = {
            let int8 = seg_dir.join("model.int8.onnx");
            if int8.exists() {
                int8
            } else {
                seg_dir.join("model.onnx")
            }
        };
        let emb = d.join("speaker/campplus.onnx");
        if !seg.exists() {
            return Err(format!("pyannote segmentation 모델 없음: {seg:?}"));
        }
        if !emb.exists() {
            return Err(format!("임베딩 모델 없음: {emb:?}"));
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(2, 8);
        eprintln!(
            "[diar] pyannote: seg={:?}, threads={threads}",
            seg.file_name().unwrap()
        );
        Self::new(seg, emb, num_speakers, 0.5, threads)
    }
}

impl Drop for PyannoteDiarizer {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.sd) };
    }
}

impl Diarizer for PyannoteDiarizer {
    fn diarize(&mut self, samples: &[f32]) -> Vec<DiarSegment> {
        let mut s = samples.to_vec();
        let mut out = Vec::new();
        unsafe {
            let result = sys::SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.sd,
                s.as_mut_ptr(),
                s.len() as i32,
                None,
                null_mut(),
            );
            if result.is_null() {
                return out;
            }
            let n = sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            let segs = sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);
            if !segs.is_null() && n > 0 {
                let slice = std::slice::from_raw_parts(segs, n as usize);
                for seg in slice {
                    out.push(DiarSegment {
                        start: seg.start as f64,
                        end: seg.end as f64,
                        speaker: seg.speaker.max(0) as u32,
                    });
                }
            }
            sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segs);
            sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
        }
        out
    }
}
