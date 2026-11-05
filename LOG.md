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
