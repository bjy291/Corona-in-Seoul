# Corona-in-Seoul


## 인하공업전문대학 3학년 빅데이터 수업 프로젝트

### 서울시 코로나 현황 시각화

데이터 출처 - 공공데이터포털 서울시 지역별 코로나 현황 
![image](https://user-images.githubusercontent.com/71078707/125554975-abd514f6-ebd6-427a-b1a1-95722d2604cc.png)
- 시각화 데이터 : 서울시 최초 코로나 감염자 발생일로 부터 2020.11.30까지의 데이터
- 회귀분석 : 11.30 데이터와 12.14까지의 데이터 사용.


### 사용환경
- 사용언어 : Python
- 개발환경 : Jupyter Notebook(Ana
- 라이브러리 : Pandas, Matplotlib, numpy, seaborn, sklearn

#### 서울시의 코로나 19 확진자 수가 확 늘어남에 따라 위험도를 알기 위해 프로젝트를 계획했습니다. 
#### 프로젝트는 11월30일까지의 데이터만 수집하고 처리했습니다.
#### 서울시의 11월까지의 전체 확진자 수, 감염률, 감염비률, 지역별 감염자 수, 지역별 감염자 수를 통한 위치 정보 시각화를 시각화 했습니다.
#### 서울시는 할로윈데이 이후로 확진자 수가 엄청나게 증가했습니다.
#### 할로윈데이 이후 10월 31일 이후 확진자 증가량을 확인할 수 있도록 했습니다.
#### 그 후 사회적 거리두기 단계가 올라감에 따라 확진자의 증가량, 감소량을 확인하도록 했습니다. 
#### 11월에 서울시 확진자가 확 증가함에 따라 공공시절 확진자 통계를 확인할 수 있도록 했습니다.
#### 회귀 분석을 통해 12월의 확진자를 예측해보았습니다.
- sklearn : 머신러닝 기반 회귀분석 예측시도
- - 결과로는 학습데이터(Attribute) 부족으로 실제 데이터와 예측 데이터 오차 큼

### 결과
![image](https://user-images.githubusercontent.com/71078707/125555663-199c9aa3-ec27-411a-9fd3-340eb2f57fb7.png)
![image](https://user-images.githubusercontent.com/71078707/125555681-55079e95-735f-490e-899a-f7fe0ca0b1bd.png)
![image](https://user-images.githubusercontent.com/71078707/125555706-6035539e-ec55-4203-a995-6c21abe1e6df.png)
#### - 서울시의 데이터만 존재하기에 다른 지역은 0
![image](https://user-images.githubusercontent.com/71078707/125555744-478935be-011d-427a-b71e-aa0e92a815d2.png)
![image](https://user-images.githubusercontent.com/71078707/125555762-80b2985d-8f14-4312-ad4d-6ae7f32e11f6.png)
#### 회귀분석
