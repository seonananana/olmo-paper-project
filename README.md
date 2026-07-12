# OLMo Transparency Experiment
**"투명성이 과학적 연구를 가능하게 한다"** — OLMo 논문 주장 실증 실험

## 논문 정보
- **논문**: OLMo: Accelerating the Science of Language Models (Groeneveld et al., 2024)
- **기관**: Allen Institute for AI (Ai2)
- **핵심 주장**: 학습 데이터·코드·체크포인트 전체 공개 → 과학적 재현·확장 가능

## 실험 구조
| 실험 | 내용 | 논문 근거 |
|---|---|---|
| 1 | 공개 체크포인트 268개 목록 확인 | 연구 가능성 |
| 2 | Base→SFT→Instruct 영어 4태스크 패턴 비교 | 재현 가능성 + 인과 추적 |
| 3 | TÜLU v2 LoRA SFT 재현 vs 공식 SFT | 재현 가능성 |
| 4 | 발달장애 직장 도메인 500개 LoRA SFT+DPO | 도메인 확장 |

## 모델
- **Base**: allenai/OLMo-2-0425-1B
- **SFT**: allenai/OLMo-2-0425-1B-SFT
- **Instruct**: allenai/OLMo-2-0425-1B-Instruct

## 폴더 구조
***
olmo-transparency-experiment/
├── README.md
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── baseline.py
│   └── evaluate.py
├── data/
│   ├── sample_cases.jsonl
│   └── domain_500_en.jsonl
├── results/
│   ├── olmo2_english_results.csv
│   ├── olmo2_sft_comparison.csv
│   └── olmo2_domain_comparison.csv
└── docs/
├── paper_card.md
└── project_canvas.md
***
## 실행 환경
- Python 3.11
- PyTorch 2.6+
- transformers, peft, trl 1.7.1, accelerate, datasets
- GPU: RTX 4090 (PART 3 학습 필요) / PART 1,2,4 추론은 그 이하 가능

## 실행 순서
```bash
pip install transformers peft trl==1.7.1 accelerate datasets bitsandbytes huggingface_hub pandas
jupyter notebook notebooks/demo.ipynb
```

## 핵심 결론
> OLMo가 가중치·데이터·코드를 전부 공개했기 때문에  
> Base→SFT→Instruct 인과 추적, LoRA 재현, 도메인 확장이 모두 가능했다.  
> 닫힌 모델에서는 이 실험 전체가 불가능하다.

## 프로젝트 활용
발달장애 근로자 직장 내 의사소통 훈련 LLM 서비스  
공개 가중치 기반 도메인 파인튜닝 → 상황별 언어 피드백 + 표현 선택지 제공
