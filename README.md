# img2md

`Qwen2.5-VL`로 `in/` 폴더의 이미지를 읽어서 `out/` 폴더에 Markdown 파일로 저장하는 배치 변환기입니다.

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
- `out/manifest.json`에 처리 결과를 남깁니다.
- `--combine-output` 옵션이 켜지면 `out/combined.md`도 함께 생성합니다.

## Docker 실행

첫 실행 시 모델이 Hugging Face에서 다운로드되므로 네트워크가 필요합니다. 실제 추론은 GPU가 있는 환경을 기준으로 구성했습니다.

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

기본 모델은 `Qwen/Qwen2.5-VL-3B-Instruct`입니다. 더 큰 모델을 쓰려면 `MODEL_ID`를 바꾸면 됩니다.

```bash
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct docker compose up --build
```

직접 CLI 옵션을 넘기는 것도 가능합니다.

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
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --max-new-tokens 3072 \
  --overwrite \
  --combine-output
```

## 로컬 Python 실행

Docker 없이도 실행할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.img2md --input-dir ./in --output-dir ./out --combine-output
```

## 참고

- Hugging Face Transformers의 `Qwen2.5-VL` 문서 기준으로 `AutoProcessor`와 `Qwen2_5_VLForConditionalGeneration` 흐름에 맞춰 구성했습니다.
- 기본 해상도 토큰 범위는 VRAM 사용량을 줄이기 위해 `min_pixels=256*28*28`, `max_pixels=1280*28*28`로 잡았습니다.
- CPU에서도 실행은 가능하지만 매우 느릴 수 있습니다.
