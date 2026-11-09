# Log

One entry for each recipe, selection or Track B model you train, newest at
the bottom. The numbers come from `experiments.csv`. "Why" and "what it means"
are in your own words: your agent asks you for them and does not make them up.
End the log with your plan for what to try next.

<!-- The form of an entry:

## <date> <tag> (Track A or B)
- changed: what is different from the run before
- why: your reason
- validation: 0.000 (before: 0.000)
- what it means: what you think the result says
-->

## 2026-10-29 Step 1: what I see in the training photos (not a run, so no score)

Looked at the contact sheets in `results/curate/`: Busan every 20th photo
(`look_busan_00` to `look_busan_05`), Seoul, Tokyo, Osaka and Taipei every
60th photo (`look_seoul_00` to `look_tokyo_04`).

- Busan: 바다, 유적지
- Osaka: 일본의 자연환경, 전통적인 거리
- Seoul: 궁궐 사진, 먹을 것 사진
- Taipei: 궁궐 사진, 유물, 건물
- Tokyo: 로드뷰 지하철, 일본 건물
- All five: 다섯 사진 모두 거리, 전체 항공뷰, 여행하면서 찍은 동물, 음식
  사진 등이 섞여 있어서 별 차이가 느껴지지 않는다.

## 2026-10-30 baseline (Track A)

- changed: nothing; the starter recipe as it is, run twice with seeds 0 and 1
  (both on this computer's CPU: MPS fails on this Mac and Colab's GPU quota
  was used up). Same recipe, same photos (all training photos, 249,984
  photos, batch 64, 3,906 steps).
- why: 두 점수가 얼마나 차이날지, 흔들리는지 보고 싶어서
- validation: 0.4987 (seed 0) and 0.4971 (seed 1); the gap is 0.0016, 0.16
  percentage points. Per city, seed 0 / seed 1: seoul 0.600/0.578, busan
  0.378/0.402, tokyo 0.629/0.627, osaka 0.351/0.369, taipei 0.535/0.509.
- what it means: 아주 작은 차이라 잘 모르겠지만, 왜 seed 1에서 더 떨어진
  건지 의문이다.

## 2026-11-05 Step 3: the baseline's mistakes (not a run, so no score)

`src/curate.py mistakes results/baseline_seed0/model.onnx` ran the baseline on
the 12,077 validation photos. Wrong: seoul 974 of 2435, busan 864 of 1390,
tokyo 1440 of 3881, osaka 1382 of 2128, taipei 1042 of 2243. I looked at the
48 most confident mistakes of each city
(`results/curate/mistakes_baseline_seed0_<city>_00.png`).

Two things the mistakes have in common, in my words:

1. 사람이 아주 많은 거리
2. 차의 로드뷰

## 2026-11-05 busan20 (Track A)

- changed: a Busan photo counts as 1.88, so Busan gets 20.0% of the 250,000
  training photos instead of 11.7% (each Busan photo is seen 9.6 times
  instead of 5.7). Seoul, Tokyo, Osaka and Taipei keep their present
  proportions. Nothing else changed.
- why: 부산 사진이 도쿄의 절반도 안 되는 게 걸려서
- validation: 0.4889 (before: 0.4987 seed 0 and 0.4971 seed 1; the seed
  difference is 0.0016, so this is below both)
- what it means: 부산의 자료를 높인 결과 부산의 정확도가 높아졌지만, 부산을
  제외한 다른 지역들을 모델이 신경쓰지 못해서 전반적인 정확도가 떨어졌고
  전체 정확도는 이전보다 떨어지게 되었다. (busan 0.378/0.402 -> 0.437;
  seoul 0.600/0.578 -> 0.544; tokyo 0.629/0.627 -> 0.623; osaka
  0.351/0.369 -> 0.337; taipei 0.535/0.509 -> 0.503)

## 2026-11-09 noflip (Track A)

- changed: deleted T.RandomHorizontalFlip() from train_transform. Everything
  else is the starter recipe again, so the Busan weighting from busan20 is
  gone too. This is one change against the baseline.
- why: Tokyo and Osaka drive on the left and Seoul, Busan and Taipei on the
  right, so a flipped photo shows lanes, signs and traffic that cannot exist
  in that city.
- validation: 0.4889 (before: 0.4987 seed 0 and 0.4971 seed 1; the seed
  difference is 0.0016, so this is below both)
- what it means: 좌우 반전된것을 삭제했는데, 그 사진들이 모델에 혼란을 일으키는
  요인은 아니였던것 같다. (seoul 0.600/0.578 -> 0.568; busan 0.378/0.402 ->
  0.374; tokyo 0.629/0.627 -> 0.624; osaka 0.351/0.369 -> 0.351; taipei
  0.535/0.509 -> 0.527)

## 2026-11-10 ema (Track A)

- changed: nothing in the photos and nothing in the schedule; only the model
  that is handed in. A copy of the weights is kept and updated after every
  step as 0.999 of itself plus 0.001 of the current weights, and at the end
  that average is loaded back and exported. It costs about 2% of the step
  time and nothing else.
- why: \fillin{waiting for the student's reason for the moving average}
- validation: 0.5052 (before: 0.4987 seed 0 and 0.4971 seed 1)
- what it means: \fillin{waiting for the student's reading of this result}

## 2026-11-10 noosv (Track A)

- changed: the ema recipe exactly as it is, trained on `track_a/selection.csv`
  instead of all training photos: the 3,110 `osv5m` photos (cameras on cars
  and bikes) are left out, so 41,147 of the 44,257. `scripts/make_selection.py`
  writes the selection; `uv run python src/curate.py count` shows what is left.
  Nothing in the recipe changed.
- why: 내가 본 '차의 로드뷰'(osv5m) 3,110장을 빼고
- validation: 0.4911 (before: 0.5052, the ema recipe on all photos)
- what it means: 로드뷰를 뺌으로써 오히려 부산을 제외한 나머지 도시가 정확도를
  잃었다. 내 생각에는 로드뷰 사진은 모델이 5개 요소를 구별하는 데에 별 영향이
  없는 것으로 판단된다. 그럼에도 부산은 정확도가 올랐는데, 그 이유는 로드뷰를
  빼면서 부산 사진의 비율이 높아졌기 때문이라고 생각한다. 그래도 전반적인
  정확도가 낮아진 이유는, 로드뷰를 통해 다양한 배경이 감소했기 때문이라고
  판단했다. (seoul 0.6074 -> 0.5758; busan 0.382 -> 0.4108; tokyo 0.6478 ->
  0.6243; osaka 0.3689 -> 0.3656; taipei 0.5198 -> 0.4788)
