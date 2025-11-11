# ☁️ 클라우드에서 학습하기 가이드

현재 환경에는 GPU가 없으므로, 클라우드 GPU를 사용하는 방법을 안내합니다.

---

## 🎯 추천 옵션 (비용 순)

### **Option 1: Google Colab (무료 ~ 저렴)** ⭐ 추천

**무료 티어:**
- GPU: Tesla T4 (16GB)
- 시간: 최대 12시간 연속
- 비용: 무료
- 적합: Qwen2.5-7B LoRA fine-tuning

**Pro 티어 ($10/월):**
- GPU: A100 (40GB) 선택 가능
- 시간: 더 길게 사용 가능
- 비용: 월 $10
- 적합: Qwen2.5-14B까지 가능

#### Colab에서 바로 시작:

```python
# 1. Colab 노트북에서 실행

# GitHub에서 코드 가져오기
!git clone https://github.com/ho0n-SNUH/snomed-ct-entity-linking.git
%cd snomed-ct-entity-linking

# 브랜치 체크아웃
!git checkout claude/incomplete-description-011CV1AmjEgryVZLyTuDTwE2

# 의존성 설치
!pip install transformers accelerate peft bitsandbytes

# Fine-tuning 시작
!python "3rd Place/Finetuning-Entity-Recognition.py" \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir /content/drive/MyDrive/models/qwen-entity \
    --lora_r 32 \
    --lora_alpha 64
```

**예상 시간:** 4-6시간 (Colab T4)

---

### **Option 2: Kaggle Notebooks (무료)** ⭐⭐

**특징:**
- GPU: P100 (16GB) 또는 T4 (16GB)
- 시간: 주당 30시간 무료
- 디스크: 100GB
- 비용: 완전 무료

**장점:**
- Colab보다 안정적
- 더 긴 연속 실행 시간
- Datasets 기본 제공

#### Kaggle에서 시작:

1. https://www.kaggle.com/code 접속
2. New Notebook 생성
3. Settings → Accelerator → GPU T4 x2 선택
4. 위의 Colab 코드 동일하게 실행

---

### **Option 3: RunPod (시간당 과금)** 💰

**특징:**
- GPU: RTX 4090 (24GB) ~ H100 (80GB)
- 비용: $0.39/hr (4090) ~ $2.49/hr (H100)
- 원하는 만큼만 사용

**예상 비용:**
- Qwen2.5-7B fine-tuning: $2-3 (RTX 4090, 6시간)
- Qwen2.5-14B fine-tuning: $5-8 (A6000, 8시간)

#### RunPod 사용법:

```bash
# 1. RunPod.io 가입 및 GPU 선택

# 2. Jupyter 환경에서:
git clone https://github.com/ho0n-SNUH/snomed-ct-entity-linking.git
cd snomed-ct-entity-linking

# 3. 자동 스크립트 실행
./upgrade_to_latest_models.sh

# 4. Fine-tuning
cd "3rd Place"
./finetune_qwen_entity_recognition.sh Qwen/Qwen2.5-7B-Instruct
```

---

### **Option 4: AWS SageMaker / Azure ML** 💼

**특징:**
- 엔터프라이즈급
- 자동 스케일링
- MLOps 통합

**비용:**
- ml.g5.xlarge (A10G): $1.01/hr
- ml.g5.2xlarge (A10G): $1.52/hr

**적합 대상:** 기업용, 대규모 실험

---

## 🚀 가장 빠른 시작 (Colab 무료)

### Step 1: Colab 노트북 만들기

**Colab 노트북 템플릿** (복붙해서 사용):

```python
# ================================
# Qwen2.5 SNOMED CT Fine-tuning
# Google Colab용
# ================================

# GPU 확인
!nvidia-smi

# 저장소 클론
!git clone https://github.com/ho0n-SNUH/snomed-ct-entity-linking.git
%cd snomed-ct-entity-linking
!git checkout claude/incomplete-description-011CV1AmjEgryVZLyTuDTwE2

# 의존성 설치
!pip install -q transformers[torch] accelerate peft bitsandbytes datasets trl

# Google Drive 마운트 (모델 저장용)
from google.colab import drive
drive.mount('/content/drive')

# 학습 데이터 확인
import os
data_path = "3rd Place/data"  # 실제 경로로 수정
if os.path.exists(data_path):
    print(f"✓ Data found: {os.listdir(data_path)}")
else:
    print("⚠️ Upload your training data to Colab")

# Fine-tuning 실행
!python "3rd Place/Finetuning-Entity-Recognition.py" \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --train_data "3rd Place/data/train_notes.csv" \
    --output_dir /content/drive/MyDrive/snomed-models/qwen-entity-v1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_length 512 \
    --use_8bit

print("\n✅ Fine-tuning completed!")
print("Model saved to: /content/drive/MyDrive/snomed-models/qwen-entity-v1")
```

### Step 2: 학습 실행

1. https://colab.research.google.com 접속
2. 새 노트북 생성
3. 런타임 → 런타임 유형 변경 → **GPU (T4)** 선택
4. 위 코드 붙여넣기
5. 실행 (Shift + Enter)

**예상 시간:**
- Setup: 5분
- Fine-tuning: 4-6시간
- 총: ~6시간

---

## 💾 학습 후 모델 가져오기

### Colab/Kaggle → 로컬

```python
# Colab에서 모델 다운로드
from google.colab import files

# 모델을 압축
!cd /content/drive/MyDrive/snomed-models && \
 tar -czf qwen-entity-v1.tar.gz qwen-entity-v1/

# 다운로드
files.download('/content/drive/MyDrive/snomed-models/qwen-entity-v1.tar.gz')
```

### 로컬에서 사용

```bash
# 압축 해제
tar -xzf qwen-entity-v1.tar.gz -C "3rd Place/models/"

# 추론 실행
python "3rd Place/main.py" \
    --base_model_path Qwen/Qwen2.5-7B-Instruct \
    --model_path_peft "3rd Place/models/qwen-entity-v1"
```

---

## 📊 비용 비교

| 옵션 | GPU | 시간 | 비용 | 총 비용 (6hr) |
|------|-----|------|------|---------------|
| **Colab 무료** | T4 | 12hr/day | $0 | **$0** ⭐ |
| **Kaggle** | P100 | 30hr/week | $0 | **$0** ⭐ |
| Colab Pro | A100 | 무제한 | $10/월 | **$10/월** |
| RunPod 4090 | RTX 4090 | 무제한 | $0.39/hr | **$2.34** |
| RunPod A6000 | A6000 | 무제한 | $0.79/hr | **$4.74** |
| AWS g5.xlarge | A10G | 무제한 | $1.01/hr | **$6.06** |

**추천:** 일단 **Colab 무료**로 시작 → 부족하면 Colab Pro ($10/월)

---

## 🎓 학습 팁

### 1. Colab 세션 끊김 방지

```javascript
// Colab에서 F12 (개발자 도구) → Console에 입력
function KeepClicking(){
   console.log("Clicking");
   document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)  // 1분마다 클릭
```

### 2. Checkpoint 자주 저장

```python
# Fine-tuning 스크립트에서
--save_steps 500  # 500 step마다 저장
```

### 3. 학습 중단 후 재개

```python
# 이전 checkpoint에서 재시작
!python Finetuning-Entity-Recognition.py \
    --resume_from_checkpoint /content/drive/MyDrive/models/qwen-entity-v1/checkpoint-1500
```

---

## ⚡ 빠른 실험 (경량 모델)

GPU가 약하거나 빠른 테스트를 원한다면:

```bash
# 3B 모델로 먼저 테스트 (2-3시간)
python Finetuning-Entity-Recognition.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_r 16 \
    --num_train_epochs 1
```

---

## 🔗 유용한 링크

- [Google Colab](https://colab.research.google.com)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [RunPod](https://www.runpod.io)
- [Hugging Face Spaces (무료 GPU)](https://huggingface.co/spaces)

---

## 📞 추가 도움

학습 중 문제가 생기면:
1. Colab/Kaggle 에러 로그 확인
2. GitHub Issues에 질문
3. Hugging Face Forums 검색

**행운을 빕니다! 🚀**
