# ============================================================
# evaluate.py
# 응답 길이 분석 및 단계별 패턴 비교
# ============================================================

import pandas as pd
import json


def load_results(csv_path: str) -> pd.DataFrame:
    """결과 CSV 로드"""
    return pd.read_csv(csv_path, encoding='utf-8-sig')


def response_length_summary(df: pd.DataFrame,
                             stage_order: list = None) -> pd.DataFrame:
    """단계별 태스크별 평균 응답 길이"""
    if stage_order is None:
        stage_order = ['Base', 'SFT', 'Instruct']

    pivot = (
        df.groupby(['stage', 'task'])['resp_len']
        .mean()
        .unstack()
        .reindex(stage_order)
        .round(1)
    )
    return pivot


def stage_mean(df: pd.DataFrame,
               stage_order: list = None) -> pd.Series:
    """단계별 전체 평균 응답 길이"""
    if stage_order is None:
        stage_order = ['Base', 'SFT', 'Instruct']
    return df.groupby('stage')['resp_len'].mean().reindex(stage_order).round(1)


def print_comparison(df: pd.DataFrame,
                     stage_order: list = None,
                     max_len: int = 120):
    """태스크 × 프롬프트 × 단계별 응답 나란히 출력"""
    if stage_order is None:
        stage_order = ['Base', 'SFT', 'Instruct']

    for task in df['task'].unique():
        print(f'\n{"="*60}\n  Task: {task}\n{"="*60}')
        sub = df[df['task'] == task]
        for idx in sub['idx'].unique():
            p = sub[sub['idx'] == idx].iloc[0]['prompt']
            print(f'\n  [Q{idx}] {p[:80]}')
            for stage in stage_order:
                r = sub[(sub['idx'] == idx) & (sub['stage'] == stage)]
                if len(r):
                    resp = r.iloc[0]['response']
                    preview = resp[:max_len] + '...' if len(resp) > max_len else resp
                    print(f'  [{stage:8s}] {preview}')


def pattern_check(df: pd.DataFrame,
                  stage_order: list = None) -> dict:
    """
    Base → SFT → Instruct 응답 길이 증가 패턴 확인
    반환: {'pattern_valid': bool, 'stage_means': dict}
    """
    if stage_order is None:
        stage_order = ['Base', 'SFT', 'Instruct']

    means = stage_mean(df, stage_order)
    values = [means[s] for s in stage_order]
    pattern_valid = all(values[i] <= values[i+1]
                        for i in range(len(values)-1))
    return {
        'pattern_valid': pattern_valid,
        'stage_means': means.to_dict()
    }


def domain_comparison_summary(csv_path: str,
                               model_order: list = None):
    """PART 4 도메인 비교 결과 요약"""
    if model_order is None:
        model_order = [
            'Base', 'Our_SFT', 'Domain_SFT', 'Domain_DPO', 'Official_SFT'
        ]
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print('\n=== 모델별 평균 응답 길이 ===')
    print(df.groupby('model')['resp_len'].mean()
          .reindex(model_order).round(1))
    return df


if __name__ == '__main__':
    # PART 2 결과 분석
    print('=== PART 2: Base→SFT→Instruct 패턴 분석 ===')
    df2 = load_results('../results/olmo2_english_results.csv')

    print('\n--- 평균 응답 길이 (태스크별) ---')
    print(response_length_summary(df2))

    print('\n--- 단계별 전체 평균 ---')
    print(stage_mean(df2))

    print('\n--- 패턴 유효성 확인 ---')
    result = pattern_check(df2)
    print(f'패턴 재현 성공: {result["pattern_valid"]}')
    print(f'단계별 평균: {result["stage_means"]}')

    print_comparison(df2)

    # PART 3 결과 분석
    print('\n\n=== PART 3: 우리SFT vs 공식SFT ===')
    df3 = load_results('../results/olmo2_sft_comparison.csv')
    print(df3[['prompt', 'our_sft', 'official_sft']].to_string())

    # PART 4 결과 분석
    print('\n\n=== PART 4: 도메인 비교 ===')
    domain_comparison_summary('../results/olmo2_domain_comparison.csv')