# ============================================================
# streamlit_app.py
# KoTE 설문폼 UX 예시 (성별 체크 → 상대 성별 메시지 → 감정 탐지 → Dropbox 무기한 저장)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import dropbox
import io
from datetime import datetime

# ✅ KoTE 모델 파이프라인
pipe = pipeline(
    "text-classification",
    model="searle-j/kote_for_easygoing_people",
    tokenizer="searle-j/kote_for_easygoing_people",
    function_to_apply="sigmoid",
    top_k=None
)

# ✅ Dropbox Access Token (Secrets에서 안전하게 가져오기!)
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
DROPBOX_PATH = "/gender_conflict_sentiment.xlsx"

# ✅ 감정 분석 함수
def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

# ✅ Streamlit UI
st.set_page_config(page_title="KoTE 젠더 감정 설문", page_icon="🧑‍🤝‍🧑")
st.title("🧑‍🤝‍🧑 20–30대 젠더 상호집단 감정 설문")
st.write("아래에서 성별을 선택하고, 상대 성별에 대한 솔직한 메시지를 작성해주세요.")

# ✅ 1️⃣ 성별 선택
gender = st.radio(
    "당신의 성별은?",
    ["여성", "남성"],
    horizontal=True
)

# ✅ 2️⃣ 대상 설명 안내
if gender == "여성":
    target_group = "20–30대 남성"
else:
    target_group = "20–30대 여성"

st.info(f"✍️ {target_group}에 대해 솔직하게 느끼는 점을 자유롭게 작성해주세요.")

# ✅ 3️⃣ 메시지 입력창
text = st.text_area(f"{target_group}에 대한 솔직한 메시지")

# ✅ 4️⃣ 버튼
if st.button("감정 분석 & 저장하기"):
    if not text.strip():
        st.warning("메시지를 작성해주세요!")
    else:
        results = analyze_emotion(text)

        if results:
            # ✅ 시각화
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

            # ✅ Dropbox에 저장 (성별 + 대상 + 메시지 + 감정 결과)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new = pd.DataFrame([{
                "timestamp": now,
                "respondent_gender": gender,
                "target_group": target_group,
                "message": text,
                "top_emotions": ", ".join([f"{label}({score})" for label, score in results])
            }])

            try:
                md, res = dbx.files_download(DROPBOX_PATH)
                with io.BytesIO(res.content) as f:
                    df_existing = pd.read_excel(f)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except dropbox.exceptions.ApiError:
                # 파일이 없으면 새로 생성
                df = df_new

            with io.BytesIO() as output:
                df.to_excel(output, index=False)
                output.seek(0)
                dbx.files_upload(output.read(), DROPBOX_PATH, mode=dropbox.files.WriteMode.overwrite)

            st.success("✅ 감정 분석 완료! 데이터가 안전하게 Dropbox에 저장되었습니다.")
        else:
            st.info("분석 결과가 없습니다.")