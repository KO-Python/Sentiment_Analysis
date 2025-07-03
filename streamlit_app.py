import streamlit as st
import plotly.express as px
from transformers import pipeline

# ✅ KoTE 파이프라인
pipe = pipeline(
    "text-classification",
    model="searle-j/kote_for_easygoing_people",
    tokenizer="searle-j/kote_for_easygoing_people",
    function_to_apply="sigmoid",
    top_k=None
)

def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

st.set_page_config(page_title="KoTE 감정 분석기", page_icon="🧐")
st.title("🧐 KoTE 온라인 감정 탐지기")
st.write("문장이나 댓글을 입력하면 KoTE 모델로 다중 감정 분석!")

text = st.text_area("분석할 문장을 입력하세요:")

if st.button("분석하기"):
    if not text.strip():
        st.warning("❗️ 문장을 입력해주세요!")
    else:
        results = analyze_emotion(text)
        st.write("✅ 상위 감정:", results)

        if results:
            labels, scores = zip(*results)
            fig = px.bar(
                x=scores,
                y=labels,
                orientation='h',
                labels={'x': 'Score', 'y': 'Emotion'},
                title="감정 탐지 결과"
            )
            fig.update_layout(xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

            st.write("📄 전체 감정 점수")
            st.table(results)