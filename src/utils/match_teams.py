from rapidfuzz import process, fuzz
from utils.logger import logger

def fuzzy_match_teams(source, target, start_threshold=80, step=20):
    """
    Summary :
        threshold를 낮춰가며 fuzzy match를 시도하고,
        unmatched가 없거나, threshold가 0이되면 종료.
    Args:    
        source: 기준이 되는 팀명 리스트 (예: A DataFrame의 팀명)
        target: 비교 대상이 되는 팀명 리스트 (예: B DataFrame의 팀명)
        threshold: 유사도 기준 (0~100). 높을수록 정확히 일치하는 경우만 매칭됨
        step: 
    
    Returns: dict 형태의 매핑 딕셔너리
    """
    current_threshold = start_threshold
    remaining_sources = source
    available_targets = target
    final_mapping = {}
    
    while current_threshold >= 0 and remaining_sources:
        new_mapping = {}
        
        for team in remaining_sources:
            match, score, _ = process.extractOne(team, available_targets, scorer=fuzz.ratio)
            if score >= current_threshold:
                new_mapping[team] = match
            
        final_mapping.update(new_mapping)
        remaining_sources -= set(new_mapping.keys())
        available_targets -= set(new_mapping.values())

        logger.info(
            f"[fuzzy_match] Threshold: {current_threshold} | 매핑 완료: {len(new_mapping)} | 남은 항목: {len(remaining_sources)}"
        )

        current_threshold -= step
    
    if not remaining_sources:
        logger.info('[fuzzy_match] 모든 항목이 성공적으로 매핑되었습니다.')
        return final_mapping, None
    
    else:
        logger.warning(f'[fuzzy_match] {sorted(remaining_sources)} 항목이 매핑되지 않았습니다.')
        return final_mapping, remaining_sources