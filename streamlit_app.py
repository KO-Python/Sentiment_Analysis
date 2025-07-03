# ============================================================
# streamlit_app.py
# KoTE 감정 탐지기 (Streamlit Cloud + Dropbox 무기한 저장)
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

# ✅ Dropbox Access Token (안전하게 Secrets로 불러옴)
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
DROPBOX_PATH = "/sentiment_logs.xlsx"

# ✅ 감정 분석 함수
def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

# ✅ Streamlit UI
st.set_page_config(page_title="KoTE 감정 탐지기", page_icon="🧐")
st.title("🧐 KoTE 감정 탐지기 (무기한 Dropbox 저장)")
st.write("문장을 입력하면 KoTE 모델로 감정 분석 후 Dropbox에 무기한 저장됩니다.")

# ✅ 사용자 입력
text = st.text_area("문장을 입력하세요:")

if st.button("분석하기"):
    if not text.strip():
        st.warning("문장을 입력해주세요!")
    else:
        results = analyze_emotion(text)

        if results:
            # ✅ 시각화 (plotly)
            labels, scores = zip(*results)
            fig = px.bar(
                x=scores, y=labels,
                orientation='h',
                labels={'x': 'Score', 'y': 'Emotion'},
                title="감정 탐지 결과"
            )
            fig.update_layout(xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

            # ✅ 전체 감정 점수 표
            st.subheader("📄 전체 감정 점수")
            st.table(results)

            # ✅ Dropbox 저장: 기존 파일 다운로드 → merge → overwrite
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new = pd.DataFrame([{
                "timestamp": now,
                "input_text": text,
                "top_emotions": ", ".join([f"{label}({score})" for label, score in results])
            }])

            try:
                md, res = dbx.files_download(DROPBOX_PATH)
                with io.BytesIO(res.content) as f:
                    df_existing = pd.read_excel(f)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except dropbox.exceptions.ApiError:
                # 첫 실행이라 파일 없으면 새로 생성
                df = df_new

            with io.BytesIO() as output:
                df.to_excel(output, index=False)
                output.seek(0)
                dbx.files_upload(output.read(), DROPBOX_PATH, mode=dropbox.files.WriteMode.overwrite)

            st.success("✅ Dropbox에 무기한 저장 완료!")
        else:
            st.info("분석 결과가 없습니다.")