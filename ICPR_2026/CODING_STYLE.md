# 🎯 Coding Style Guide

A clean, scalable, and maintainable coding standard for Python ML/AI projects.

---

## 1. Registry Pattern

### ✅ Khi nào DÙNG
| Domain | Use Case | Lý do |
|--------|----------|-------|
| **LLM** | Model types (`qwen`, `llama`, `gemma`) | Dễ switch model qua config |
| **LLM** | Templates (chat format) | Mỗi model có format prompt khác nhau |
| **CV** | Backbone (`resnet`, `vit`, `swin`) | Benchmark nhiều kiến trúc |
| **CV** | Augmentation pipeline | Thử nghiệm các chiến thuật aug |
| **ML** | Feature transformers | Đổi scaler/encoder linh hoạt |
| **RecSys** | Embedding layers | Thử collaborative vs content-based |

### ❌ Khi nào KHÔNG dùng
| Use Case | Lý do |
|----------|-------|
| Script inference 1 model cố định | Cần code tường minh, dễ debug |
| Export ONNX/TensorRT | Là pipeline chạy 1 lần |
| Code visualize/plot | Logic duy nhất, không thay thế |

### Code mẫu
```python
MODELS = {}

def register_model(name: str):
    def wrapper(cls):
        MODELS[name] = cls
        return cls
    return wrapper

@register_model('qwen2-vl')
class Qwen2VL:
    ...
```

---

## 2. Dataclass

### ✅ Khi nào DÙNG
| Domain | Use Case |
|--------|----------|
| **LLM** | `ModelMeta` (model_type, template, requires) |
| **CV** | `DatasetInfo` (num_classes, mean, std) |
| **RecSys** | `UserProfile`, `ItemMeta` |

### ❌ Khi nào KHÔNG dùng
| Use Case | Dùng thay thế |
|----------|---------------|
| Config động từ YAML | `dict` hoặc `OmegaConf` |
| Object cần method phức tạp | Class thường |

```python
@dataclass
class ModelMeta:
    model_type: str
    template: str
    architectures: List[str] = field(default_factory=list)
```

---

## 3. Type Hints

### ✅ Bắt buộc cho
- Public functions/methods
- API endpoints
- Class `__init__`

### ❌ Có thể bỏ qua cho
- Lambda expressions
- List comprehensions nội bộ

```python
# ✅ Good
def encode(text: str, max_length: int = 512) -> torch.Tensor: ...

# ❌ Bad
def encode(text, max_length=512): ...
```

---

## 4. Project Structure by Domain

### LLM Project
```
llm_project/
├── model/
│   ├── constant.py    # LLMModelType.qwen = 'qwen'
│   ├── register.py    # MODEL_MAPPING
│   └── qwen.py
├── template/          # Chat templates
├── dataset/           # Dataset loaders
└── pipeline/
    ├── train.py
    └── infer.py
```

### CV Project
```
cv_project/
├── backbone/
│   ├── register.py
│   ├── resnet.py
│   └── vit.py
├── augmentation/
│   └── register.py
├── dataset/
│   ├── coco.py
│   └── custom.py
└── train.py
```

### RecSys Project
```
recsys_project/
├── model/
│   ├── collaborative.py
│   └── content_based.py
├── feature/
│   ├── user_encoder.py
│   └── item_encoder.py
├── retrieval/
└── ranking/
```

---

## 5. Config-Driven Design

### ✅ Dùng config cho
| Thành phần | Ví dụ |
|------------|-------|
| Hyperparameters | `lr`, `batch_size`, `epochs` |
| Model selection | `model_type: qwen2-7b` |
| Data paths | `train_data`, `val_data` |

### ❌ Không dùng config cho
| Thành phần | Lý do |
|------------|-------|
| Logic nghiệp vụ | Khó debug, dễ sai |
| Constants cố định | Đặt trong `constant.py` |

```yaml
# configs/train_v1.yaml
model:
  type: qwen2-7b
  lora_rank: 16
training:
  lr: 1e-4
  epochs: 3
```

---

## 6. Logging Rules

| Level | Khi nào dùng |
|-------|--------------|
| `DEBUG` | Chi tiết internal, chỉ bật khi debug |
| `INFO` | Tiến trình chính: load model, start training |
| `WARNING` | Deprecated, fallback behavior |
| `ERROR` | Exception đã catch và xử lý |

```python
logger.info(f'Loaded {len(dataset)} samples')
logger.warning('Flash attention not available, using SDPA')
```

---

## 7. Domain-Specific Patterns

### LLM: Template Pattern
```python
@dataclass
class Template:
    system: str
    user_prefix: str
    assistant_prefix: str
    
    def format(self, messages: List[dict]) -> str: ...
```

### CV: Transform Pipeline
```python
train_transforms = Compose([
    build_aug({'type': 'RandomCrop', 'size': 224}),
    build_aug({'type': 'Normalize', 'mean': [0.485, 0.456, 0.406]}),
])
```

### RecSys: Two-Tower Pattern
```python
class TwoTowerModel:
    def __init__(self, user_encoder, item_encoder): ...
    def forward(self, user_features, item_features): ...
```

---

## Quick Decision Table

| Câu hỏi | Có → Dùng | Không → Bỏ qua |
|---------|-----------|----------------|
| Component có thể thay thế? | Registry | Direct import |
| Cần validate nhiều field? | Dataclass | Dict |
| Function là public API? | Type hints | Optional |
| Giá trị thay đổi theo thí nghiệm? | Config YAML | Hardcode |
| Cần trace lại lúc debug? | Logging | Print |
