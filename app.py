# =====================================================================
# ⚽ AI 스카우터 웹페이지 (Streamlit)
# 실행: streamlit run app.py   /  배포: Streamlit Community Cloud
# =====================================================================
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title='AI 스카우터', page_icon='⚽')
st.title('⚽ AI 스카우터')
st.write('선수의 능력치를 입력하면 선형회귀 AI가 **종합능력치**를 예측합니다.')

# 학습한 모델과 독립변수 이름 불러오기 (cache로 한 번만 로드)
@st.cache_resource
def load():
    return joblib.load('model.pkl')
saved = load()
model, feature = saved['model'], saved['feature']

# [웹-조건1] 독립변수 값 입력
# 설명: 슬라이더로 1~99 사이의 능력치를 입력받는다. 게임 능력치 범위와 같게 제한해
# 잘못된 값이 들어가지 않게 했고, 라벨에 실제 사용하는 독립변수 이름을 표시한다.
value = st.slider(f'선수의 {feature} 능력치', min_value=1, max_value=99, value=70)

# [웹-조건3] 인공지능 분석 버튼
# 설명: st.button은 클릭된 순간 True를 반환한다. if문으로 감싸 버튼을 눌렀을 때만
# 예측 코드가 실행되도록 했다.
if st.button('인공지능 분석'):
    # [웹-조건2] 예측 결과 출력
    # 설명: 입력값을 학습 때와 같은 형태(컬럼명이 있는 표)로 만들어 model.predict()에 넣는다.
    # 예측값은 1~99로 제한하고 소수 첫째 자리로 반올림해 st.success로 보여준다.
    # 점수 구간에 따라 등급도 함께 출력한다.
    pred = model.predict(pd.DataFrame({feature: [value]}))[0]
    pred = round(float(np.clip(pred, 1, 99)), 1)
    grade = ('🌟 월드클래스' if pred >= 85 else '🔥 주전급' if pred >= 75
             else '🌱 유망주' if pred >= 65 else '⚽ 아마추어')
    st.success(f'예상 종합능력치: {pred}점')
    st.info(f'등급: {grade}')
