# img2md

`in/` 폴더의 이미지를 읽어서 `out/` 폴더에 Markdown 파일로 저장하는 배치 변환기입니다. 기본 모델은 `Qwen/Qwen2.5-VL-7B-Instruct`이며, CLI에서 다른 VLM preset으로 바꿔 실행할 수 있습니다.

## 구조

```text
img2md/
├─ app/
│  └─ img2md.py
├─ in/
├─ out/
├─ hf_cache/
├─ torch_cache/
├─ compose.yaml
├─ Dockerfile
└─ requirements.txt
```

## 지원 입력 형식

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.bmp`
- `.tif`
- `.tiff`
- `.gif`

## 기본 동작

- `in/` 안의 이미지를 이름순으로 순회합니다.
- 각 이미지마다 `out/<원본파일명>.md`를 생성합니다. 예: `slide01.png -> slide01.png.md`
- 기본값은 기존 결과가 있어도 덮어쓰기입니다.
- `out/manifest.json`에 처리 결과를 남깁니다.
- `--combine-output` 옵션이 켜지면 `out/combined.md`도 함께 생성합니다.

## 모델 선택

현재 내장된 preset:

- `qwen7b`: `Qwen/Qwen2.5-VL-7B-Instruct`
- `qwen3b`: `Qwen/Qwen2.5-VL-3B-Instruct`
- `paddleocr_vl`: `PaddlePaddle/PaddleOCR-VL`
- `mineru2_5`: `opendatalab/MinerU2.5-2509-1.2B`

preset 목록 확인:

```bash
python -m app.img2md --list-model-presets
```

기본 preset은 `qwen7b`입니다.

## Docker 실행

첫 실행 시 모델이 Hugging Face에서 다운로드되므로 네트워크가 필요합니다. 실제 추론은 GPU가 있는 환경을 기준으로 구성했습니다.

`HF_TOKEN`이 있으면 다운로드 제한이 완화됩니다.

```bash
HF_TOKEN=hf_xxx docker compose up --build
```

### 1. 폴더 준비

```bash
mkdir -p in out hf_cache torch_cache
```

### 2. 입력 이미지 넣기

이미지 파일들을 `in/` 폴더에 복사합니다.

### 3. 실행

```bash
docker compose up --build
```

또는 `docker run`으로 직접 실행할 수 있습니다.

```bash
docker build -t img2md-qwen25vl .
docker run --rm --gpus all \
  -v "$(pwd)/in:/workspace/in" \
  -v "$(pwd)/out:/workspace/out" \
  -v "$(pwd)/hf_cache:/models/hf" \
  -v "$(pwd)/torch_cache:/models/torch" \
  img2md-qwen25vl
```

## 옵션 예시

기본값은 `qwen7b` preset입니다. Compose에서 preset을 바꾸려면:

```bash
MODEL_PRESET=mineru2_5 docker compose up --build
```

`docker run`에서 직접 preset을 고를 수도 있습니다.

```bash
docker run --rm --gpus all \
  -v "$(pwd)/in:/workspace/in" \
  -v "$(pwd)/out:/workspace/out" \
  -v "$(pwd)/hf_cache:/models/hf" \
  -v "$(pwd)/torch_cache:/models/torch" \
  img2md-qwen25vl \
  python -m app.img2md \
  --input-dir /workspace/in \
  --output-dir /workspace/out \
  --model-preset paddleocr_vl \
  --max-new-tokens 3072 \
  --no-overwrite \
  --combine-output
```

원격 모델 ID를 직접 덮어쓰는 것도 가능합니다.

```bash
docker run --rm --gpus all \
  -v "$(pwd)/in:/workspace/in" \
  -v "$(pwd)/out:/workspace/out" \
  -v "$(pwd)/hf_cache:/models/hf" \
  -v "$(pwd)/torch_cache:/models/torch" \
  img2md-qwen25vl \
  python -m app.img2md \
  --input-dir /workspace/in \
  --output-dir /workspace/out \
  --model-preset qwen7b \
  --model-id Qwen/Qwen2.5-VL-72B-Instruct
```

프롬프트를 직접 바꾸고 싶으면:

```bash
python -m app.img2md \
  --input-dir ./in \
  --output-dir ./out \
  --model-preset paddleocr_vl \
  --prompt "OCR:"
```

(임시)
```bash
cd /mnt/c/study/graduation/img2md
mkdir -p .docker-tmp
printf '{}' > .docker-tmp/config.json

DOCKER_CONFIG="$(pwd)/.docker-tmp" docker build -t img2md-qwen25vl .

docker run --rm --gpus all \
  -v "$(pwd)/in:/workspace/in" \
  -v "$(pwd)/out:/workspace/out" \
  -v "$(pwd)/hf_cache:/models/hf" \
  -v "$(pwd)/torch_cache:/models/torch" \
  img2md-qwen25vl
```

## 로컬 Python 실행

Docker 없이도 실행할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.img2md --input-dir ./in --output-dir ./out --model-preset qwen7b --combine-output
```

## 참고

- Hugging Face Transformers의 범용 `AutoProcessor`와 `AutoModelForImageTextToText` 흐름으로 구성했습니다.
- Qwen preset에는 기본 해상도 토큰 범위 `min_pixels=256*28*28`, `max_pixels=1280*28*28`를 적용했습니다.
- `PaddleOCR-VL`, `MinerU2.5`는 문서 파싱 성향이 강하므로, Markdown 결과가 마음에 안 들면 `--prompt` 또는 `--prompt-file`로 별도 튜닝하는 편이 낫습니다.
- CPU에서도 실행은 가능하지만 매우 느릴 수 있습니다.
