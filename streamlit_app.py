import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import dropbox
import io
from datetime import datetime

# ✅ KoTE 모델 로드
pipe = pipeline(
    "text-classification",
    model="searle-j/kote_for_easygoing_people",
    tokenizer="searle-j/kote_for_easygoing_people",
    function_to_apply="sigmoid",
    top_k=None
)

# ✅ Dropbox API 준비
DROPBOX_TOKEN = st.secrets["sl.u.AF3Zb4DDKEgptbCUg-le1tdLgVBUlEBkanTiodfH0eWvKRC43ltVeQju-SSt0wVghxgzlqC-X5gp4nCIDfVLltn_Tw4dj4eA6eBpYBBj0zlaG2BP3yYRWGVEC56CMlvqWaPcYFVKOYSC_BIVH_RVPrSjnaDsRG0KlF88PUiZ5KSgj9DFz505m_IHdr_-D4gTILY-0SVJ5iUeGMANwWCneTVom7-Y0wUx073Qu8k5s_DmQ61ibBrhyPfJJPw-ilUvnksxXHUi8GWJiIgCBfwS40VR01xyv5qgc2Rjqev5ORQFcZvdxvH9vLqQ9AM-L2eW-qjDfCFcn2qEY-IxiU0nIGpToREDp3_X5LqeVROwqOaCHvCf8UjJpQ3Qfxn6Z-QO-gsNnwt74kR_Ws_3LlLn4ZN0lZBrGN4hgX2udLausWQf4YCnse3UiuidOsuQLA5C9NgYxz0FXut3VtHAcTFPoQICHNUt3s6WtMmSGvJGBqo3_HXIulXk23YWfh6ptsYSlKb-JeFh_b_vuT-TbOw14_7crLEZ3_4bNazyz8sqYJoFXtZ-SGAzn0OEDj-cYzmnYWMejDml91JDtdJ97_Ym-XewO19yKyINcyzSLDzK9jwL5myaNU6W3v4VJe08aw6yuDUKrT6xBI6CHOLFLEN3HLu8TablBc-EMRaN49OFE37vDfGCLh6Pjet_HTBwMjHxQ9nn93ejVxBNqLNS4ffIOVmTuZSh0ZVjsuJrnmZAJfrMFghJQZJlbhnBOCT2zg85ouSfN05oAZbDxOealJEiX-etmXxIZaXbHzx3zUB0N6vkSC1SiIFIFjdxw2a2sPPupt9Iht8PANi-DuSqGedpl-vVe7_Ihq_rPtA3-QL8aP-fgglOV2OJlLGekmnScdbRBE3_uU-k4lD25N7uEM-kztE6cxrCU1tMTvtIt3l8h8Tu93TB59r7QZ3BccAuP6zrh4vANH5DhS4JbeIKsGsoqrLGByHDwoRYA62MQjKgV8TkAzlXRGSszA8SJ0Hi07epsLDp5PZ3db-FJuIRDMVAjvBhIhgAWmAqDZBOxhYsFa6fk-kc2ZSlahWbtJqiduOuuXG7vfy2skASz3ybJvnfyMZCRuga0BWuHLL08GZihyFykjk5U9mrJrN9Eyqei-guQlgP2K1W_g0LJewJyq0nJDnSU5LAzOBGiNippHUNxg28S-uYfMdsKOAFeM0BCNA-kB_ZouVCQbOMV-K84r4ZQgfgqYFfTb_wemwJtPLlMiqpYqNDFMTYfDdchBKVS1P5KEagLdDzNya04ViDmA8gyFTp"]
dbx = dropbox.Dropbox(DROPBOX_TOKEN)
DROPBOX_PATH = "/sentiment_logs.xlsx"

def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

# ✅ Streamlit 앱
st.title("🧐 KoTE 감정 탐지기 (Dropbox 무기한 저장)")
text = st.text_area("문장을 입력하세요:")

if st.button("분석하기"):
    if text.strip():
        results = analyze_emotion(text)
        
        # 시각화
        if results:
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

            # ✅ 새 데이터 만들기
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new = pd.DataFrame([{
                "timestamp": now,
                "input_text": text,
                "top_emotions": ", ".join([f"{label}({score})" for label, score in results])
            }])

            # ✅ Dropbox에서 최신 파일 다운로드 & merge
            try:
                md, res = dbx.files_download(DROPBOX_PATH)
                with io.BytesIO(res.content) as f:
                    df_existing = pd.read_excel(f)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            except dropbox.exceptions.ApiError:
                # 첫 실행 시 파일이 없으면 새로 생성
                df = df_new

            # ✅ 다시 Dropbox에 overwrite
            with io.BytesIO() as output:
                df.to_excel(output, index=False)
                output.seek(0)
                dbx.files_upload(
                    output.read(),
                    DROPBOX_PATH,
                    mode=dropbox.files.WriteMode.overwrite
                )

            st.success("✅ 감사합니다!")