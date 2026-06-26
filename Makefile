.PHONY: dev dev-coreml dev-gpu install build

dev:
	./run-app

dev-coreml:
	./run-app --coreml

dev-gpu:
	./run-app --gpu

install:
	cd app && pnpm install

build:
	cd app && pnpm tauri build
