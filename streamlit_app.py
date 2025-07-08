import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import dropbox
import io
from datetime import datetime

# ============================================
# ✅ 1) KoTE 감정 탐지 파이프라인 (CPU 강제)
# ============================================
pipe = pipeline(
    "text-classification",
    model="searle-j/kote_for_easygoing_people",
    tokenizer="searle-j/kote_for_easygoing_people",
    function_to_apply="sigmoid",
    top_k=None,
    device=-1  # ✅ CPU-only 환경
)

# ============================================
# ✅ 2) Dropbox API 세팅
# ============================================
dbx = dropbox.Dropbox(
    oauth2_access_token=st.secrets["dropbox"]["ACCESS_TOKEN"],
    oauth2_refresh_token=st.secrets["dropbox"]["REFRESH_TOKEN"],
    app_key=st.secrets["dropbox"]["APP_KEY"],
    app_secret=st.secrets["dropbox"]["APP_SECRET"]
)
DROPBOX_PATH = "/gender_conflict_sentiment.xlsx"

# ============================================
# ✅ 3) 상태 변수 초기화
# ============================================
if "page" not in st.session_state:
    st.session_state["page"] = "intro"
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

# ============================================
# ✅ 4) 감정 분석 함수
# ============================================
def analyze_emotion(text):
    outputs = pipe(text)[0]
    results = [(o["label"], round(o["score"], 3)) for o in outputs if o["score"] > 0.3]
    return sorted(results, key=lambda x: x[1], reverse=True)

# ============================================
# ✅ 5) 인트로 페이지
# ============================================
if st.session_state["page"] == "intro":
    st.set_page_config(page_title="2030 성별 인식 조사", page_icon="🧑‍🤝‍🧑")
    st.title("🧑‍🤝‍🧑 2030 성별 인식 조사")

    st.write('''
    안녕하세요.  
    이 조사는 귀하께서 일상 속에서 느낀 경험과 생각을 자유롭게 표현하는 설문 조사입니다.  
    정해진 답변은 없으며, 귀하께서 평소 느끼신 생각과 느낌을 적어주시는 것이 가장 중요합니다.  
    응답해주신 내용은 연구 목적으로만 활용되며, 익명성은 철저히 보장되오니 안심하고 솔직하게 작성해주시기 바랍니다.
    ''')

    st.subheader("📌 기본 정보 입력")
    st.text_input("당신의 연령은?", placeholder="예: 25", key="age")
    st.radio("당신의 성별은?", ["여성", "남성"], index=None, horizontal=True, key="gender")

    if st.button("다음 창으로"):
        if not st.session_state["age"].strip():
            st.warning("⚠️ 연령을 입력해주세요!")
        elif not st.session_state["age"].strip().isdigit():
            st.warning("⚠️ 연령은 숫자로만 입력해주세요!")
        elif not st.session_state["gender"]:
            st.warning("⚠️ 성별을 선택해주세요!")
        else:
            st.session_state["page"] = "survey"
            st.rerun()

# ============================================
# ✅ 6) 설문 페이지
# ============================================
elif st.session_state["page"] == "survey":
    age = st.session_state.get("age", "")
    gender = st.session_state.get("gender", "")

    if not gender or not age:
        st.session_state["page"] = "intro"
        st.rerun()

    # ✅ 동일 key 유지 (숨김)
    st.text_input("당신의 연령", value=age, key="age", disabled=True)
    st.radio("당신의 성별", ["여성", "남성"], index=["여성", "남성"].index(gender), key="gender", disabled=True)

    st.subheader("✍️ 설문에 응답해주세요")

    user_gender = gender
    opposite_gender = "남성" if user_gender == "여성" else "여성"

    st.write(f'''
    다음은 귀하의 일상 속 경험에서 사람들과의 관계에서 느낀 생각이나 감정에 대해 묻는 질문입니다.  
    귀하께서 속한 [20–30대 {user_gender}]에 대해 평소에 생각했던 점, 좋았던 점, 아쉬웠던 점, 느낀 점 등을 자유롭게 3–5줄 이상 적어주세요.
    ''')

    own_group_text = st.text_area(
        f"{user_gender} 집단에 대한 생각",
        key="own_group_text",
        height=200
    )

    st.write(f'''
    계속해서 귀하가 속하지 않은 상대 성별(집단) [20–30대 {opposite_gender}]에 대해 평소에 생각했던 점, 좋았던 점, 아쉬웠던 점, 느낀 점 등을 자유롭게 3–5줄 이상 적어주세요.
    ''')

    other_group_text = st.text_area(
        f"{opposite_gender} 집단에 대한 생각",
        key="other_group_text",
        height=200
    )

    if st.button("감정 분석하기"):
        if not own_group_text.strip() or not other_group_text.strip():
            st.warning("⚠️ 모든 질문에 답변을 작성해주세요!")
        else:
            st.session_state["own_results"] = analyze_emotion(own_group_text)
            st.session_state["other_results"] = analyze_emotion(other_group_text)
            st.session_state["page"] = "result"
            st.session_state["analyzed"] = True
            st.rerun()

# ============================================
# ✅ 7) 결과 페이지
# ============================================
elif st.session_state["page"] == "result":
    age = st.session_state.get("age", "")
    gender = st.session_state.get("gender", "")
    own_group_text = st.session_state.get("own_group_text", "")
    other_group_text = st.session_state.get("other_group_text", "")

    if not age or not gender or not own_group_text or not other_group_text:
        st.session_state["page"] = "intro"
        st.rerun()

    # ✅ 숨김 위젯으로 상태 유지
    st.text_input("당신의 연령", value=age, key="age", disabled=True)
    st.radio("당신의 성별", ["여성", "남성"], index=["여성", "남성"].index(gender), key="gender", disabled=True)
    st.text_area("귀하의 생각(자기 집단)", value=own_group_text, key="own_group_text", height=100, disabled=True)
    st.text_area("귀하의 생각(상대 집단)", value=other_group_text, key="other_group_text", height=100, disabled=True)

    st.subheader("🎉 연구에 참여해주셔서 감사합니다!")

    st.write('''
    연구에 참여해주셔서 진심으로 감사드립니다.  
    이번 조사는 20–30대 성인 남녀 간의 감정 및 인식 차이를 탐색하기 위한 연구의 일환으로 진행되었습니다.  
    설문 초반에는 연구의 구체적인 목적을 밝히지 않았으나, 이는 참여자의 보다 자연스럽고 솔직한 응답을 유도하기 위한 방식이었습니다.  
    귀하께서 작성해주신 서술형 응답은, 상대 성별에 대한 감정과 사회적 인식이 어떤 방식으로 형성되고 표현되는지를 분석하는 데 사용됩니다.  
    응답 내용은 익명으로 처리되며, 오직 학술 연구 목적으로만 활용됩니다.

    귀하께서 제출해주신 내용의 분석 결과는 아래와 같습니다.
    ''')

    df_own = pd.DataFrame(st.session_state["own_results"], columns=["label", "score"]).sort_values(by="score", ascending=False)
    df_other = pd.DataFrame(st.session_state["other_results"], columns=["label", "score"]).sort_values(by="score", ascending=False)

    st.write(f"### ✅ 귀하의 집단({gender})에 대한 감정 분석 결과")
    fig1 = px.bar(
        df_own,
        x="score",
        y="label",
        orientation='h',
        category_orders={"label": df_own["label"].tolist()}
    )
    fig1.update_layout(xaxis_range=[0, 1])
    st.plotly_chart(fig1)
    st.table(df_own)

    st.write(f"### ✅ 상대 집단({ '남성' if gender=='여성' else '여성' })에 대한 감정 분석 결과")
    fig2 = px.bar(
        df_other,
        x="score",
        y="label",
        orientation='h',
        category_orders={"label": df_other["label"].tolist()}
    )
    fig2.update_layout(xaxis_range=[0, 1])
    st.plotly_chart(fig2)
    st.table(df_other)

    st.subheader("🔍 감정 분석 결과 신뢰도 평가")
    trust_score = st.radio(
        "감정 분석 결과를 얼마나 신뢰하시나요? (1점 = 전혀 신뢰하지 않음, 5점 = 매우 신뢰함)",
        ["1점", "2점", "3점", "4점", "5점"],
        index=None,
        key="trust_score"
    )

    if st.button("결과 저장하기"):
        if not trust_score:
            st.warning("⚠️ 신뢰도를 선택해주세요!")
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            age_value = int(age.strip())

            df_new = pd.DataFrame([{
                "timestamp": now,
                "respondent_age": age_value,
                "respondent_gender": gender,
                "own_group_text": own_group_text,
                "own_results": ", ".join([f"{label}({score})" for label, score in st.session_state["own_results"]]),
                "other_group_text": other_group_text,
                "other_results": ", ".join([f"{label}({score})" for label, score in st.session_state["other_results"]]),
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

            st.success("✅ 새로고침 후 추가 참여가 가능합니다.")
            st.session_state.clear()