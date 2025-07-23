import os
import streamlit as st
import random
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

# LangChainの基本的な部品
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChainでOpenAIのモデルを使うための部品
from langchain_openai import ChatOpenAI

# LLMの初期化
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# LLMに質問して回答を得る関数
def get_ai_response(input_text: str, mode: str) -> str:
    """
    入力テキストとモード（選択値）を受け取り、LLMからの回答を返す関数
    
    Args:
        input_text (str): ユーザーの入力テキスト
        mode (str): 選択されたモード（"BMI計算" or "今日の運勢占い"）
    
    Returns:
        str: LLMからの回答
    """
    if mode == "BMI計算":
        # BMI計算の専門家としてのプロンプト
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは経験豊富な健康・栄養の専門家です。BMIや体重管理、健康的な生活習慣について専門的な知識を持っています。
            ユーザーから体重と身長の情報、またはBMIに関する質問を受け取ります。
            科学的根拠に基づいた実践的なアドバイスを提供してください。
            医学的な診断は行わず、一般的な健康情報として回答してください。"""),
            ("human", "{input}")
        ])
    elif mode == "今日の運勢占い":
        # 占い師としてのプロンプト
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは経験豊富で知識豊富な占い師です。星座占いや運勢について深い知識を持っています。
            ユーザーから星座の情報や運勢に関する質問を受け取ります。
            神秘的で希望に満ちた、でも実用的なアドバイスを提供してください。
            占いの雰囲気を大切にしながら、前向きで建設的な回答をしてください。"""),
            ("human", "{input}")
        ])
    else:
        # デフォルトの汎用プロンプト
        prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは親切なAIアシスタントです。質問に丁寧に答えてください。"),
            ("human", "{input}")
        ])
    
    # チェインを作成して実行
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": input_text})
    return response
# --- Streamlit UIの構築 ---
st.header("体重診断と今日の運勢のAIアシスタント")
st.write("診断後、フォローアップの質問が可能です")

# 動作モードの選択
selected_mode = st.radio(
    "何がしたいですか？",
    ("BMI計算", "今日の運勢占い")
)

# 各モードに応じた入力フォームの表示
if selected_mode == "BMI計算":
    st.subheader("健康診断・相談")
    
    # BMI計算用の数値入力
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input("身長 (cm)", min_value=0.0, max_value=250.0, step=0.1)
    with col2:
        weight_kg = st.number_input("体重 (kg)", min_value=0.0, max_value=300.0, step=0.1)
    
    # 健康に関する悩み
    health_concern = st.text_area(
        "健康に関して悩んでいることがあれば教えてください",
        placeholder="例：最近体重が増えてきて心配です\n例：運動不足で疲れやすいです\n例：食生活を改善したいです"
    )
    
    # 診断ボタン
    if st.button("健康診断を受ける"):
        if height_cm > 0 and weight_kg > 0:
            # BMI計算
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            
            # BMI判定
            if bmi < 18.5:
                bmi_category = "低体重"
            elif 18.5 <= bmi < 25:
                bmi_category = "普通体重"
            elif 25 <= bmi < 30:
                bmi_category = "肥満（1度）"
            else:
                bmi_category = "肥満（2度以上）"
            
            # 専門家への質問文を作成
            input_text = f"""
            【患者情報】
            身長: {height_cm}cm
            体重: {weight_kg}kg
            BMI: {bmi:.2f} ({bmi_category})
            
            【相談内容】
            {health_concern if health_concern.strip() else '特に悩みはありません'}
            
            上記の情報を踏まえて、健康状態の評価とアドバイスをお願いします。
            """
            
            with st.spinner("健康専門家が診断中..."):
                try:
                    # get_ai_response関数を呼び出して回答を取得
                    response = get_ai_response(input_text, selected_mode)
                    
                    # 回答を表示
                    st.success("🩺 健康専門家からの診断結果:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("身長と体重を正しく入力してください。")

elif selected_mode == "今日の運勢占い":
    st.subheader("今日の運勢占い")
    
    # 星座選択
    constellation_options = [
        "選択してください", "おひつじ座", "おうし座", "ふたご座", "かに座", 
        "しし座", "おとめ座", "てんびん座", "さそり座", 
        "いて座", "やぎ座", "みずがめ座", "うお座"
    ]
    constellation = st.selectbox("あなたの星座を選択してください", constellation_options)
    
    # 最近の悩みや叶えたいこと
    personal_concern = st.text_area(
        "最近の悩みや、叶えたいことがあれば教えてください",
        placeholder="例：恋愛がうまくいかない\n例：仕事で成功したい\n例：人間関係を良くしたい\n例：健康になりたい"
    )
    
    # 占いボタン
    if st.button("今日の運勢を占う"):
        if constellation != "選択してください":
            # 占い師への質問文を作成
            input_text = f"""
            【相談者情報】
            星座: {constellation}
            日付: 2025年7月23日
            
            【相談内容】
            {personal_concern if personal_concern.strip() else '特に相談したいことはありません'}
            
            上記の情報を踏まえて、今日の運勢と運気を上げるためのアドバイスをお願いします。
            """
            
            with st.spinner("占い師が運勢を占っています..."):
                try:
                    # get_ai_response関数を呼び出して回答を取得
                    response = get_ai_response(input_text, selected_mode)
                    
                    # 回答を表示
                    st.success("🔮 占い師からの運勢診断:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("星座を選択してください。")
