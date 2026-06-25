mod app_state;
mod capture;
mod commands;
mod events;
mod export;
mod session;

use app_state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init());

    // shell 플러그인은 사이드카(externalBin) 실행에 쓰이며 iOS 에서는 불가하다.
    // 데스크톱에서만 등록한다 (docs/02-architecture.md 결정3·C.1).
    #[cfg(desktop)]
    {
        builder = builder.plugin(tauri_plugin_shell::init());
    }

    builder
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::ping,
            commands::list_inputs,
            commands::start_session,
            commands::stop_session,
            commands::export_transcript
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
