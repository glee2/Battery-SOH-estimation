# CNN을 활용한 리튬-이온 배터리 건강 상태 예측 방법론 개발 </br> A convolutional neural network model for SOH estimation of Li-ion batteries with physical interpretability

### 연구의 필요성
- <b>리튬-이온 배터리(Li-ion battery)</b>의 충전 용량은 충・방전을 반복함에 따라 점차 감소하며, <b>건강 상태(SOH: State-of-health)</b>은 이러한 배터리의 수명을 측정하는 지표임
  - 배터리의 노화는 내부 저항의 불일치, 내부 열 불균형 등 다양한 원인으로 인해 비선형적으로 나타남
- 배터리 제조 업체는 가속화된 실험 환경을 구성하여 배터리 건강 상태의 변화를 관찰함으로써 배터리의 내구성 평가를 진행하여 배터리 성능 검증 및 품질 개선에 활용함
  - 그러나 가속화된 실험 환경 하에서도 최소 3개월의 시간이 소요되는 등 <em>효율성 제고가 필수적</em> 임
- 이에 따라 배터리의 충・방전 주기에 따른 수명 열화 데이터를 바탕으로 미래 SOH 값을 예측하는 데이터 기반 모델링 방법들이 제안되었으며, 리튬-이온 배터리의 비선형적인 수명 열화 패턴을 모델링할 수 있다는 강점으로 인해 각광받고 있음
  - SVM (Support vector machine), LOF (Local outlier factor), Particle filter, ELM (Extreme learning machine), GRU (Gated recurrent unit), LSTM (Long-short term memory) 등 다양한 데이터 마이닝 및 머신러닝 기법들이 사용됨
- 그러나 이러한 데이터 기반 모델링 방법들은 통계적 특징에 대한 의존성, 적은 데이터 샘플 수, 결과 해석의 어려움 등으로 인해 <em>실제로 적용되기에 한계</em> 가 있음
  - 배터리 수명 열화 패턴으로부터 자동으로 특징을 추출하고, 이를 바탕으로 미래의 배터리 건강 상태를 예측하며, 예측 결과를 자동으로 해석할 수 있는 데이터 기반 방법론의 개발이 필요함

### 데이터
- 배터리 수명 열화 데이터
  - 내구성 평가를 위해 배터리의 충・방전을 반복함에 따라 변화하는 SOH 값이 기록된 데이터
  - SOH 값은 다음과 같이 리튬-이온 배터리의 최초 충전 용량 대비 특정 시점에서 남아있는 용량의 비율로 정의됨

$$ SOH(t)= {{C(t)} \over {C(0)}},\ C(0): 최초\ 충전\ 용량, C(t): t시점의\ 충전\ 용량 $$

- 전기차에 사용되는 리튬-이온 배터리 <b>379</b>개의 수명 열화 데이터를 활용함(현대자동차 제공)
  - 0~1,000 충・방전 주기 동안의 SOH 값이 기록되어 있음

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/Figure1.jpg?raw=true" width="80%" height="80%"></p>
<p align="center"><u><b> 리튬-이온 배터리 수명 열화 데이터 예시 </b></u></p>

### 방법론
- RP (Recurrence plot), GAF (Gramian angular fields)와 같은 <b>시계열-이미지 변환 방법(time-series imaging methods)</b>을 활용하여 시계열 형태인 배터리 수명 열화 데이터를 이미지 형태의 데이터로 변환함
  - RP: 시계열 데이터 내 값의 변화를 공간 궤적으로 표현하고, 각 공간 궤적에 위치하는 점 사이 거리를 바탕으로 2차원 행렬을 구성하여 이미지 형태로 변환하는 방법
  - GAF: 시계열 데이터의 각 시점의 값 사이의 상관 관계를 극좌표를 기준으로 표현하여 이미지 형태로 변환하는 방법. 시계열 데이터의 값을 각도의 합 또는 차를 이용하여 나타내며, 이에 따라 Gramian angular summation field (GASF)와 Gramian angular difference field (GADF)의 두 가지 방식으로 구분됨
  - 본 연구에서는 RP와 GAF 방법을 통해 배터리 수명 열화 데이터를 다음과 같이 충・방전 주기에 따른 2차원 이미지로 변환함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/Figure2.png?raw=true" width="80%" height="80%"></p>

- 이미지 데이터 처리에 강점이 있는 <b>CNN (Convolutional neural networks)</b>을 활용하여 배터리 수명 열화 데이터의 초기 충・방전 주기(예: 100주기)의 SOH 값을 입력으로, 후기 충・방전 주기(예: 700주기)의 SOH 값을 출력으로 하는 회귀 예측 모델을 구축함
  - CNN 모델에 RP와 GAF 이미지를 병렬로 입력함으로써 미래 SOH 값 예측에 배터리 수명 열화 패턴에 대한 풍부한 정보를 반영하도록 함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/Figure3.png?raw=true" width="80%" height="80%"></p>
<p align="center"><u><b> 배터리 건강 상태 예측을 위한 CNN 모델 구조 </b></u></p>

- CAM (Class activation map) 기법을 도입하여 미래 SOH 값에 영향을 미치는 배터리의 초기 충・방전 주기의 주요한 시계열적 특징을 포착함
  
### 실험 설계
- 배터리 건강 상태 예측 범위
  - 다양한 맥락에서의 배터리 건강 상태 예측에 대한 시사점을 제공하기 위해, 입력과 출력 시점을 달리하여 다양한 예측 범위를 설정하였음

|예측 범위|주기|
|:-----:|:-----:|
|입력 시점|50|
||100|
||150|
||200|
||250|
|출력 시점|300|
||500|
||700|

- 5-fold 교차 검증
  - 전체 데이터셋을 5개의 부분집합으로 나누어 CNN의 학습 및 평가를 여러 번 수행하여, 모든 데이터 샘플을 최소 한 번 이상 성능 평가에 활용함

- 예측 성능 평가
  - 다음과 같이 회귀 예측을 위한 성능 평가 지표를 도입하여, 배터리 건강 상태 예측에 대한 성능 평가를 수행함

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/MAPE.png?raw=true" width="60%" height="60%"></p>

### 연구 결과
- 기존 데이터 기반 배터리 건강 상태 예측 방법론과의 비교 분석을 실시하여 다음의 성능 평가 결과를 얻었으며, 본 연구에서 제안한 방법론이 대부분의 예측 범위에서 가장 높은 성능을 달성하였음

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/Figure4.png?raw=true" width="80%" height="80%"></p>

- CAM 기법을 활용하여 미래 SOH 값에 대한 초기 충・방전 주기의 SOH 값의 영향을 다음과 같이 활성화 맵의 형태로 나타냄
  - 100주기 동안의 SOH 값을 입력으로 하여 700주기 시점의 SOH 값을 예측하는 CNN 모델에 대해 적용한 결과임
  - 최종 시점(700주기)의 SOH 값에 따라 0 \~ 0.7, 0.7 \~ 0.75, 0.75 \~ 0.8, 0.8 \~ 0.85, 0.85 \~ 1.0의 5가지 배터리 샘플에 대해 활성화 맵을 생성함
- 배터리 건강 상태가 정상인 경우 극초기 충・방전 주기(25 \~ 50)의 SOH 값의 영향이 큰 반면, 불량인 경우 중간 시점 충・방전 주기(50 \~ 100)의 SOH 값의 영향이 큰 것으로 나타남

<p align="center"><img src="https://github.com/glee2/Markdown-practice/blob/main/2_SOH_estimation/Figure5.png?raw=true" width="80%" height="80%"></p>
