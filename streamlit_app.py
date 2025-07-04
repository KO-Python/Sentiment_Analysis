import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import dropbox
import io
from datetime import datetime

# ✅ KoTE 감정 탐지 파이프라인
pipe = pipeline(
    "text-classification",
    model="searle-j/kote_for_easygoing_people",
    tokenizer="searle-j/kote_for_easygoing_people",
    function_to_apply="sigmoid",
    top_k=None
)

# ✅ Dropbox API 세팅
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
DROPBOX_PATH = "/gender_conflict_sentiment.xlsx"

def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

st.set_page_config(page_title="KoTE 젠더 감정 설문", page_icon="🧑‍🤝‍🧑")
st.title("🧑‍🤝‍🧑 20–30대 성별 간 감정 조사")

# ✅ 지시문: 목적+방법
st.write('''
이 조사는 상대 성별(남성/여성)에 대한 귀하의 생각과 경험을 바탕으로 **감정을 분석하기 위해** 진행됩니다.  
작성하신 내용은 연구 목적으로만 사용되며, 귀하의 익명성은 철저히 보장됩니다.

상대 성별에 대해 평소에 생각했던 점, 좋았던 점, 아쉬웠던 점 등을 자유롭게 적어주세요.  
**긍정적인 생각과 부정적인 생각 모두 환영합니다!**

* 작성하신 내용을 바탕으로 귀하의 감정을 자동으로 분석해 드립니다.  
✅ 최소 3~5줄 이상 솔직하게 작성해주시면 큰 도움이 됩니다.
''')

# ✅ 상태 변수 초기화
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# ✅ 연령 입력
age = st.text_input(
    "당신의 연령은?",
    placeholder="예: 25",
    key="age"
)

# ✅ 성별 선택
gender = st.radio(
    "당신의 성별은?",
    ["여성", "남성"],
    index=None,
    horizontal=True,
    key="gender"
)

# ✅ 대상 설명
if gender:
    target_group = "20–30대 남성" if gender == "여성" else "20–30대 여성"
    st.info(f"✍️ {target_group}에 대해 솔직하게 작성해주세요.")
else:
    target_group = None

# ✅ 메시지 입력
text = st.text_area(
    "상대 성별에 대한 메시지:" if target_group else "먼저 성별을 선택해주세요!",
    key="text",
    height=250
)

# ✅ 감정 분석 버튼
if st.button("감정 분석하기"):
    if not age.strip():
        st.warning("연령을 입력해주세요!")
    elif not age.strip().isdigit():
        st.warning("연령은 숫자로만 입력해주세요!")
    elif not gender:
        st.warning("성별을 선택해주세요!")
    elif not text.strip():
        st.warning("메시지를 작성해주세요!")
    elif len(text.strip()) < 10:
        st.warning("최소 10자 이상 작성해주세요!")
    else:
        results = analyze_emotion(text)
        if results:
            st.session_state["analyzed"] = True
            st.session_state["results"] = results
        else:
            st.info("분석 결과가 없습니다. 다른 내용을 시도해주세요.")

# ✅ 분석 결과 표시
if st.session_state["analyzed"] and st.session_state["results"]:
    results = st.session_state["results"]

    labels, scores = zip(*results)
    fig = px.bar(
        x=scores, y=labels,
        orientation='h',
        labels={'x': 'Score', 'y': 'Emotion'},
        title="감정 탐지 결과"
    )
    fig.update_layout(xaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 전체 감정 점수")
    st.table(results)

    # ✅ 신뢰도 질문
    st.subheader("🔍 감정 분석 결과 신뢰도 평가")
    trust_score = st.radio(
        "이 감정 분석 결과를 얼마나 신뢰하시나요? (5점 척도)",
        [
            "1점 (전혀 신뢰하지 않음)",
            "2점",
            "3점 (보통)",
            "4점",
            "5점 (매우 신뢰함)"
        ],
        index=None,
        key="trust_score"
    )

    # ✅ 결과 저장 버튼
    if st.button("결과 저장하기"):
        if not trust_score:
            st.warning("감정 분석 신뢰도를 선택해주세요!")
        elif not age.strip().isdigit():
            st.warning("연령은 숫자로 입력해주세요!")
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            age_value = int(age.strip())

            df_new = pd.DataFrame([{
                "timestamp": now,
                "respondent_age": age_value,
                "respondent_gender": gender,
                "target_group": target_group,
                "message": text,
                "top_emotions": ", ".join([f"{label}({score})" for label, score in results]),
                "trust_score": trust_score
            }])

            try:
                md, res = dbx.files_download(DROPBOX_PATH)
                with io.BytesIO(res.content) as f:
                    df_existing = pd.read_excel(f)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except dropbox.exceptions.ApiError:
                df = df_new

            with io.BytesIO() as output:
                df.to_excel(output, index=False)
                output.seek(0)
                dbx.files_upload(output.read(), DROPBOX_PATH, mode=dropbox.files.WriteMode.overwrite)

            st.success("✅ 결과가 성공적으로 저장되었습니다. 새로 고침 후 추가 참여가 가능합니다.")
            st.session_state.clear()