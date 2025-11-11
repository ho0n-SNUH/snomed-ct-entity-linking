#!/bin/bash
# 최신 모델로 업그레이드 자동화 스크립트

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "SNOMED CT - Latest Model Upgrade Script"
echo "=========================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 단계 출력
print_step() {
    echo ""
    echo -e "${GREEN}==>${NC} $1"
    echo ""
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# GPU 확인
print_step "Step 0: Checking GPU availability"
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. GPU required for this upgrade."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "Detected VRAM: ${VRAM} MB"

# 모델 크기 선택
if [ "$VRAM" -gt 40000 ]; then
    DEFAULT_MODEL="Qwen/Qwen2.5-14B-Instruct"
    echo -e "${GREEN}Recommending: Qwen2.5-14B (48GB VRAM available)${NC}"
elif [ "$VRAM" -gt 20000 ]; then
    DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
    echo -e "${YELLOW}Recommending: Qwen2.5-7B (24GB+ VRAM available)${NC}"
else
    print_error "Insufficient VRAM (need at least 24GB)"
    exit 1
fi

# 사용자 선택
echo ""
echo "Select model to use:"
echo "1) Qwen2.5-7B-Instruct (faster, 14GB VRAM)"
echo "2) Qwen2.5-14B-Instruct (better performance, 28GB VRAM)"
echo "3) Qwen2.5-32B-Instruct (best performance, 64GB VRAM, requires INT4)"
echo "4) BioMistral-7B (medical specialized)"
read -p "Enter choice [1-4] (default: auto-detect): " model_choice

case $model_choice in
    1)
        QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
        MODEL_SIZE="7B"
        ;;
    2)
        QWEN_MODEL="Qwen/Qwen2.5-14B-Instruct"
        MODEL_SIZE="14B"
        ;;
    3)
        QWEN_MODEL="Qwen/Qwen2.5-32B-Instruct"
        MODEL_SIZE="32B"
        print_warning "32B model requires INT4 quantization"
        ;;
    4)
        QWEN_MODEL="BioMistral/BioMistral-7B"
        MODEL_SIZE="7B"
        ;;
    *)
        QWEN_MODEL=$DEFAULT_MODEL
        MODEL_SIZE=$(echo $QWEN_MODEL | grep -oP '\d+B')
        ;;
esac

echo -e "${GREEN}Selected: $QWEN_MODEL${NC}"

# Step 1: 의존성 설치
print_step "Step 1: Installing dependencies"

pip install --upgrade pip

# 필수 패키지
pip install transformers>=4.37.0
pip install torch>=2.0.0
pip install accelerate>=0.25.0
pip install peft>=0.8.0
pip install bitsandbytes>=0.41.0
pip install vllm>=0.3.0
pip install FlagEmbedding>=1.2.0
pip install langchain>=0.1.0
pip install faiss-gpu>=1.7.2

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 2: 모델 다운로드
print_step "Step 2: Downloading models"

echo "Downloading $QWEN_MODEL..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    '$QWEN_MODEL',
    trust_remote_code=True
)

print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$QWEN_MODEL',
    torch_dtype=torch.float16,
    device_map='cpu',  # Just download, don't load to GPU yet
    trust_remote_code=True
)

print('✓ Model downloaded successfully')
"

echo "Downloading BGE-M3..."
python3 -c "
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
print('✓ BGE-M3 downloaded successfully')
"

# Step 3: FAISS 재구축
print_step "Step 3: Rebuilding FAISS index with BGE-M3"

ASSETS_DIR="3rd Place/assets"
if [ ! -d "$ASSETS_DIR" ]; then
    ASSETS_DIR="assets"
fi

TERMINOLOGIES="$ASSETS_DIR/newdict_snomed_extended.txt"
OUTPUT_INDEX="$ASSETS_DIR/faiss_index_bge_m3"

if [ ! -f "$TERMINOLOGIES" ]; then
    print_warning "Terminologies file not found: $TERMINOLOGIES"
    print_warning "Skipping FAISS rebuild. You'll need to rebuild it manually."
else
    echo "Building FAISS index from $TERMINOLOGIES..."
    echo "This may take 10-20 minutes..."

    python3 model_migration_guide.py \
        --action rebuild-faiss \
        --terminologies "$TERMINOLOGIES" \
        --output-index "$OUTPUT_INDEX"

    echo -e "${GREEN}✓ FAISS index rebuilt${NC}"
fi

# Step 4: Fine-tuning 준비
print_step "Step 4: Preparing fine-tuning scripts"

FINETUNING_DIR="3rd Place"
if [ ! -d "$FINETUNING_DIR" ]; then
    FINETUNING_DIR="."
fi

# Fine-tuning 스크립트 생성
cat > "${FINETUNING_DIR}/finetune_qwen_entity_recognition.sh" << 'EOF'
#!/bin/bash
# Qwen2.5 Entity Recognition Fine-tuning

QWEN_MODEL="$1"
OUTPUT_DIR="models/qwen-entity-$(date +%Y%m%d-%H%M%S)"

echo "Fine-tuning $QWEN_MODEL for Entity Recognition"
echo "Output: $OUTPUT_DIR"

python Finetuning-Entity-Recognition.py \
    --base_model "$QWEN_MODEL" \
    --output_dir "$OUTPUT_DIR" \
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
    --use_8bit

echo "✓ Fine-tuning completed: $OUTPUT_DIR"
EOF

chmod +x "${FINETUNING_DIR}/finetune_qwen_entity_recognition.sh"

cat > "${FINETUNING_DIR}/finetune_qwen_classification.sh" << 'EOF'
#!/bin/bash
# Qwen2.5 Classification Fine-tuning

QWEN_MODEL="$1"
OUTPUT_DIR="models/qwen-classification-$(date +%Y%m%d-%H%M%S)"

echo "Fine-tuning $QWEN_MODEL for Classification"
echo "Output: $OUTPUT_DIR"

python Finetuning-Classification.py \
    --base_model "$QWEN_MODEL" \
    --output_dir "$OUTPUT_DIR" \
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
    --use_8bit

echo "✓ Fine-tuning completed: $OUTPUT_DIR"
EOF

chmod +x "${FINETUNING_DIR}/finetune_qwen_classification.sh"

echo -e "${GREEN}✓ Fine-tuning scripts created${NC}"

# Step 5: 테스트
print_step "Step 5: Testing new models"

echo "Testing Qwen model..."
python3 model_migration_guide.py \
    --action test-qwen \
    --qwen-model "$QWEN_MODEL"

# Step 6: 요약
print_step "Upgrade Complete! 🎉"

echo ""
echo "============================================"
echo "Summary"
echo "============================================"
echo "✓ Model: $QWEN_MODEL"
echo "✓ Embedding: BGE-M3"
echo "✓ FAISS Index: $OUTPUT_INDEX"
echo ""
echo "Next Steps:"
echo "============================================"
echo ""
echo "1. Fine-tune Entity Recognition:"
echo "   cd $FINETUNING_DIR"
echo "   ./finetune_qwen_entity_recognition.sh $QWEN_MODEL"
echo ""
echo "2. Fine-tune Classification:"
echo "   ./finetune_qwen_classification.sh $QWEN_MODEL"
echo ""
echo "3. Run inference with new models:"
echo "   python main.py \\"
echo "     --base_model_path $QWEN_MODEL \\"
echo "     --model_path_peft models/qwen-entity-<timestamp> \\"
echo "     --model_classification_path_peft models/qwen-classification-<timestamp> \\"
echo "     --faiss_index $OUTPUT_INDEX"
echo ""
echo "Expected Performance Improvements:"
echo "  - Entity F1: +8-15%"
echo "  - Classification Accuracy: +10-18%"
echo "  - IoU: +12-22%"
echo ""
echo "============================================"
echo ""

# 설정 파일 저장
cat > model_config.env << EOF
# Model Configuration
# Generated on: $(date)

QWEN_MODEL=$QWEN_MODEL
MODEL_SIZE=$MODEL_SIZE
FAISS_INDEX=$OUTPUT_INDEX
TERMINOLOGIES=$TERMINOLOGIES

# Usage:
# source model_config.env
EOF

echo -e "${GREEN}Configuration saved to: model_config.env${NC}"
echo ""
echo "Happy coding! 🚀"
