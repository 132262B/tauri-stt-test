//! antirez/qwen-asr (C, MIT) 를 in-process 정적 라이브러리로 컴파일한다.
//! mac/iOS 는 Apple Accelerate(BLAS), 그 외는 제네릭/NEON 커널. Python/외부 프로세스 0.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let files = [
        "qwen_asr.c",
        "qwen_asr_kernels.c",
        "qwen_asr_kernels_generic.c",
        "qwen_asr_kernels_neon.c", // arm64 에서만 활성(__ARM_NEON 가드)
        "qwen_asr_kernels_avx.c",  // x86 에서만 활성(__AVX2__ 가드) — arm 에선 빈 오브젝트
        "qwen_asr_audio.c",
        "qwen_asr_encoder.c",
        "qwen_asr_decoder.c",
        "qwen_asr_tokenizer.c",
        "qwen_asr_safetensors.c",
    ];

    let mut build = cc::Build::new();
    build.include("vendor");
    for f in files {
        build.file(format!("vendor/{f}"));
    }
    // 디버그 빌드여도 C 커널은 항상 최적화(추론 속도) + fast-math.
    build.opt_level(3);
    build.flag_if_supported("-ffast-math");
    build.warnings(false);

    // Apple 플랫폼: Accelerate BLAS 사용.
    if target_os == "macos" || target_os == "ios" {
        build.define("USE_BLAS", None);
        build.define("ACCELERATE_NEW_LAPACK", None);
    }

    build.compile("qwen_asr");

    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rerun-if-changed=vendor");
}
