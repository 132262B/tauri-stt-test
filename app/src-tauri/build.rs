fn main() {
    // screencapturekit(및 apple-cf/apple-metal)는 Swift 코드를 링크하므로 실행 시
    // libswift_Concurrency.dylib(=/usr/lib/swift) 가 rpath 에 있어야 한다. 없으면 앱이
    // dyld "Library not loaded" 로 아예 실행되지 않는다 (C7 시스템오디오 도입).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
    tauri_build::build()
}
