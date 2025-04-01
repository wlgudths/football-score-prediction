from rapidfuzz import process, fuzz

def fuzzy_match_teams(source_list, target_list, threshold=85):
    """
    source_list: 기준이 되는 팀명 리스트 (예: A DataFrame의 팀명)
    target_list: 비교 대상이 되는 팀명 리스트 (예: B DataFrame의 팀명)
    threshold: 유사도 기준 (0~100). 높을수록 정확히 일치하는 경우만 매칭됨.
    
    Returns: dict 형태의 매핑 딕셔너리
    """
    mapping = {}
    
    for team in source_list:
        match, score, _ = process.extractOne(team, target_list, scorer=fuzz.ratio)
        if score >= threshold:
            mapping[team] = match
    
    return mapping