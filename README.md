# 10th-toy-team6
6조 토이프로젝트

모델 구조
인기 게임 top500에서 가지고 오기

수정하기 

# ① Steam 데이터 수집

여기서 **데이터를 두 종류로 나눠서 생각하는 게 제일 중요해.**

## A. 게임 데이터

게임 자체가 어떤 게임인지 설명하는 데이터.

예를 들어 GTA V가 있다면:

| Feature | GTA V |
| --- | --- |
| Action | 1 |
| RPG | 0 |
| Open World | 1 |
| Story Rich | 1 |
| Multiplayer | 1 |
| Exploration | 1 |
| Horror | 0 |
| 가격 | 29.99 |
| 긍정 비율 | 0.89 |
| 평균 플레이타임 | 35h |
| 출시연도 | 2015 |

이런 식으로 **"이 게임은 어떤 특성을 가지고 있는가?"**를 나타내는 데이터야.

너희가 지금 수집하고 있는 코드에서도 게임별로 `genre`, `tags`, `steam_categories`, `playtime_hours` 등을 모으고 있지.

여기에 가능하다면

- 가격
- 리뷰 수
- 긍정 리뷰 비율
- 출시일
- 개발사
- 플랫폼
- 싱글/멀티
- 평균 플레이타임

등을 추가할 수 있어.

---

## B. 사용자 데이터

이건 **"이 사람이 어떤 게임을 얼마나 했는가?"**야.

예를 들어 가상의 User A가:

| 게임 | 플레이 시간 |
| --- | --- |
| GTA V | 120h |
| RDR2 | 80h |
| Witcher 3 | 60h |
| Stardew Valley | 40h |
| Hades | 10h |

를 플레이했다고 해보자.

Steam API에서 `playtime_forever`를 이용해서 이런 데이터를 얻을 수 있어. 현재 코드에서도 이 값을 시간으로 변환하고 있어.

---

# ② 게임 Feature Engineering

여기가 **게임 데이터를 ML이 계산할 수 있는 형태로 바꾸는 과정**이야.

원래 데이터가

```
GTA V
장르: Action, Adventure
태그: Open World, Story Rich, Crime, Multiplayer
가격: $29.99
평점: 89%
```

이렇게 되어 있잖아.

컴퓨터가 이걸 그대로 비교하기는 어려우니까 숫자로 바꿔.

### 장르/태그

예를 들어 우리가 사용할 feature를

```
Action
RPG
Strategy
Open World
Story Rich
Multiplayer
Horror
Exploration
```

이라고 정했다고 해보자.

GTA V는

```
Action       1
RPG          0
Strategy     0
Open World   1
Story Rich   1
Multiplayer  1
Horror       0
Exploration  1
```

이렇게 표현할 수 있어.

즉,

```
GTA V
→ [1, 0, 0, 1, 1, 1, 0, 1]
```

이게 **게임 Feature Vector**야.

---

## 수치형 Feature도 넣는다

예를 들어:

```
가격        = 29.99
평점        = 0.89
플레이타임  = 35
출시연도    = 2015
```

이런 것도 넣을 수 있어.

다만 가격이 29.99이고 평점이 0.89라고 그대로 넣으면 **feature마다 숫자의 스케일이 다르기 때문에** 정규화/표준화를 해주는 게 좋음.

결국 한 게임이:

```
GTA V
→ [Action, RPG, OpenWorld, StoryRich, Multiplayer, ..., 가격, 평점, 플레이타임]
```

이라는 **하나의 숫자 벡터**가 되는 거야.

---

# ③ 사용자 취향 벡터 생성

**여기가 진짜 너희 프로젝트의 핵심이야.**

아까 User A가

```
GTA V       120h
RDR2         80h
Witcher 3    60h
Stardew      40h
Hades        10h
```

를 했다고 했지.

이제 각각의 게임 Feature Vector를 가져와.

예를 들어 아주 단순화해서:

```
              Action  OpenWorld  Story  RPG  Cozy
GTA V            1        1        1    0     0
RDR2             1        1        1    1     0
Witcher 3        1        1        1    1     0
Stardew          0        0        1    1     1
Hades            1        0        1    1     0
```

그리고 **플레이 시간에 따라 가중치를 줘.**

120시간 플레이한 GTA V는 사용자 취향을 많이 반영하고,

10시간 플레이한 Hades는 상대적으로 적게 반영하는 거지.

그러면 최종적으로:

```
User A 취향

Action       0.83
Open World   0.71
Story        0.95
RPG          0.68
Cozy         0.21
```

같은 결과가 나와.

즉 이 사람을 한 문장으로 표현하면:

> **스토리와 액션을 좋아하고, 오픈월드/RPG 성향도 강하지만 Cozy 게임 선호도는 낮은 사용자**
> 

가 되는 거야.

이게 바로 **User Preference Vector**야.

---

# ④ 사용자 ↔ 미플레이 게임 유사도 계산

인기 게임 top500에서 가지고 오기

수정하기 

이제 추천할 차례야.

Steam에는 사용자가 아직 안 해본 게임이 엄청 많잖아.

예를 들어 후보가:

```
Cyberpunk 2077
Baldur's Gate 3
Elden Ring
Hades II
Stardew Valley
```

라고 해보자.

각 게임도 이미 Feature Vector를 가지고 있음.

그러면:

```
User A 취향 벡터
        ↕
   유사도 계산
        ↕
게임 Feature Vector
```

를 하는 거야.

보통 여기서는 **Cosine Similarity**를 사용하기 좋음.

예를 들어:

| 게임 | 사용자 취향과 유사도 |
| --- | --- |
| Cyberpunk 2077 | 0.94 |
| Baldur's Gate 3 | 0.87 |
| Elden Ring | 0.82 |
| Hades II | 0.65 |
| Stardew Valley | 0.31 |

그러면 Cyberpunk가 가장 취향에 가까운 게임이라는 뜻.

---

# ⑤ 추천 점수 산출

여기서 한 단계 더 생각할 수 있어.

**유사도 = 추천 점수**로 바로 사용해도 되지만, 다른 요소도 반영할 수 있음.

예를 들어:

```
추천 점수
= 취향 유사도
+ 게임 평점
```

같은 식.

예를 들어:

| 게임 | 취향 유사도 | 평점 | 최종 추천점수 |
| --- | --- | --- | --- |
| Cyberpunk | 0.94 | 0.89 | 0.93 |
| BG3 | 0.87 | 0.96 | 0.89 |
| Elden Ring | 0.82 | 0.92 | 0.84 |

이런 식으로 만들 수 있음.

**단, 가중치는 임의로 정하지 말고 실험으로 결정하는 게 좋아.**

예를 들어

```
Model A
Similarity 100%

Model B
Similarity 80%
+ Rating 20%

Model C
Similarity 70%
+ Rating 20%
+ Recency 10%
```

를 비교해서 실제 추천 성능이 가장 좋은 걸 선택할 수 있어.

---

# ⑥ Top-N 추천

이제 점수 순으로 정렬하면 끝.

```
1위 Cyberpunk 2077     93%
2위 Baldur's Gate 3    89%
3위 Elden Ring         84%
4위 Kingdom Come 2     82%
5위 Hades II           76%
```

이게 **Top-N Recommendation**이야.

여기서 N은 5개, 10개 등으로 정하면 됨.

---

단순히

> "Cyberpunk 추천합니다."
> 

하고 끝내지 말고 **왜 추천했는지를 보여주는 거야.**

예를 들어:

### 🎮 Cyberpunk 2077

**추천도 93%**

> 당신이 플레이한 게임에서 나타난 **Open World, Story Rich, Action** 선호도가 높아 이 게임을 추천합니다.
> 

그리고:

```
당신의 선호              게임의 특성

Open World  █████████  →  █████████
Story Rich  █████████  →  █████████
Action      ████████   →  ████████
RPG         ███████    →  ███████
```

이렇게 보여줄 수 있음.

그러면 **추천 근거가 설명 가능한 추천 시스템**이 되는 거야.

---

# 데이터 흐름

가장 중요한 부분이라 이걸 이해하면 됨.
             [Steam API]
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   [게임 정보]           [사용자 정보]
        │                   │
        │              GTA V 120h
        │              RDR2 80h
        │              Witcher 60h
        │                   │
        ↓                   ↓
 [게임 Feature]       [플레이 이력]
        │                   │
        │                   ↓
        │            [취향 벡터 생성]
        │                   │
        │                   ↓
        │             User Vector
        │                   │
        └─────────┬─────────┘
                  ↓
             [유사도 계산]
                  ↓
       아직 안 한 게임들 중에서
                  ↓
            [추천 점수]
                  ↓
              Top 10
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
   추천 게임   취향 프로필   추천 이유
