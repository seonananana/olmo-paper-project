# ============================================================
# baseline.py
# PART 2: Base → SFT → Instruct 단계별 추론 함수
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_IDS = {
    'Base'    : 'allenai/OLMo-2-0425-1B',
    'SFT'     : 'allenai/OLMo-2-0425-1B-SFT',
    'Instruct': 'allenai/OLMo-2-0425-1B-Instruct'
}


def load_model(model_id: str):
    """모델 및 토크나이저 로드"""
    print(f'[로딩] {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map='auto'
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, stage: str, text: str) -> str:
    """
    stage별 프롬프트 구성
    - Base: raw 텍스트
    - SFT/Instruct: chat template 또는 ### Instruction 포맷
    """
    if stage == 'Base':
        return text
    if tokenizer.chat_template is None:
        return f"### Instruction:\n{text}\n\n### Response:\n"
    try:
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        return f"### Instruction:\n{text}\n\n### Response:\n"


def generate(tokenizer, model, stage: str, text: str,
             max_new_tokens: int = 150) -> str:
    """단일 프롬프트 추론 (greedy decoding)"""
    prompt = build_prompt(tokenizer, stage, text)
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        out[0][input_len:], skip_special_tokens=True
    ).strip()


def run_stage(tokenizer, model, stage: str, tasks: dict) -> list:
    """태스크 전체 실행"""
    rows = []
    for task, prompts in tasks.items():
        for i, p in enumerate(prompts):
            resp = generate(tokenizer, model, stage, p)
            rows.append({
                'stage'   : stage,
                'task'    : task,
                'idx'     : i + 1,
                'prompt'  : p,
                'response': resp,
                'resp_len': len(resp)
            })
            print(f'  [{task}] {i+1} 완료')
    return rows


if __name__ == '__main__':
    import pandas as pd
    import json

    # 프롬프트 로드
    with open('../data/sample_cases.jsonl', encoding='utf-8') as f:
        cases = [json.loads(line) for line in f]

    tasks = {}
    for c in cases:
        tasks.setdefault(c['task'], []).append(c['prompt'])

    all_rows = []
    for stage, model_id in MODEL_IDS.items():
        tokenizer, model = load_model(model_id)
        print(f'\n=== [{stage}] 추론 시작 ===')
        all_rows.extend(run_stage(tokenizer, model, stage, tasks))
        del model
        torch.cuda.empty_cache()
        print(f'[{stage}] 완료 + 메모리 해제')

    df = pd.DataFrame(all_rows)
    df.to_csv('../results/olmo2_english_results.csv',
              index=False, encoding='utf-8-sig')
    print('\n저장 완료: results/olmo2_english_results.csv')