"""추천 모델 모음. 모든 모델은 recommend(user, ctx) -> pd.Series 인터페이스를 따른다."""

from . import popularity, content, user_cf, item_cf, svd, mab

# 평가에 넣을 모델 딕셔너리 (이름 -> recommend 함수)
MODELS = {
    "popularity": popularity.recommend,
    "content": content.recommend,
    "user_cf": user_cf.recommend,
    "item_cf": item_cf.recommend,
    "svd": svd.recommend,
    "mab": mab.recommend,
}
