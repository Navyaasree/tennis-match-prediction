import plotly.graph_objects as go
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00ffcc;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("tennis_model.pkl")

# ---------------- TITLE ----------------
st.title("🎾 AI Tennis Match Predictor")

st.markdown("### Compare two players and predict the winner using Machine Learning")

# Store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PLAYER DATABASE ----------------
players = {
    "Roger Federer": [5, 8, 2, 60, 35, 20, 4],
    "Rafael Nadal": [3, 6, 2, 65, 38, 22, 5],
    "Novak Djokovic": [1, 7, 1, 70, 40, 25, 6],
    "Andy Murray": [15, 5, 3, 55, 30, 18, 3],
    "Carlos Alcaraz": [2, 9, 2, 68, 37, 24, 5]
}

# ---------------- PLAYER SELECTION ----------------
col1, col2 = st.columns(2)

with col1:
    playerA_name = st.selectbox("Select Player A", list(players.keys()))

with col2:
    playerB_name = st.selectbox("Select Player B", list(players.keys()))

# ---------------- AUTO-FILL STATS ----------------
a_stats = players[playerA_name]
b_stats = players[playerB_name]

st.subheader("📊 Player Stats")

col1, col2 = st.columns(2)

with col1:
    st.write(f"### {playerA_name}")
    a_rank = st.number_input("Rank A", value=a_stats[0])
    a_ace = st.number_input("Aces A", value=a_stats[1])
    a_df = st.number_input("Double Faults A", value=a_stats[2])
    a_first_in = st.number_input("First Serve In A", value=a_stats[3])
    a_first_won = st.number_input("First Serve Won A", value=a_stats[4])
    a_second_won = st.number_input("Second Serve Won A", value=a_stats[5])
    a_bp = st.number_input("Break Points Saved A", value=a_stats[6])

with col2:
    st.write(f"### {playerB_name}")
    b_rank = st.number_input("Rank B", value=b_stats[0])
    b_ace = st.number_input("Aces B", value=b_stats[1])
    b_df = st.number_input("Double Faults B", value=b_stats[2])
    b_first_in = st.number_input("First Serve In B", value=b_stats[3])
    b_first_won = st.number_input("First Serve Won B", value=b_stats[4])
    b_second_won = st.number_input("Second Serve Won B", value=b_stats[5])
    b_bp = st.number_input("Break Points Saved B", value=b_stats[6])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Match Result"):

    # Feature differences
    rank_diff = a_rank - b_rank
    ace_diff = a_ace - b_ace
    df_diff = a_df - b_df
    first_in_diff = a_first_in - b_first_in
    first_won_diff = a_first_won - b_first_won
    second_won_diff = a_second_won - b_second_won
    bp_diff = a_bp - b_bp

    data = np.array([[rank_diff, ace_diff, df_diff,
                      first_in_diff, first_won_diff,
                      second_won_diff, bp_diff]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)

    win_prob_A = prob[0][1] * 100
    win_prob_B = prob[0][0] * 100

    st.subheader("🏆 Prediction Result")

    if prediction[0] == 1:
        st.success(f"{playerA_name} is likely to WIN 🏆")
    else:
        st.success(f"{playerB_name} is likely to WIN 🏆")

    st.write(f"🎯 {playerA_name}: {win_prob_A:.2f}%")
    st.write(f"🎯 {playerB_name}: {win_prob_B:.2f}%")

    st.progress(int(win_prob_A))
    st.progress(int(win_prob_B))

    # Save history
    result = playerA_name if prediction[0] == 1 else playerB_name

    st.session_state.history.append({
        "Player A": playerA_name,
        "Player B": playerB_name,
        "Winner": result,
        "Prob A (%)": round(win_prob_A, 2),
        "Prob B (%)": round(win_prob_B, 2)
    })
 
         # ---------------- ADVANCED INTERACTIVE GRAPH ----------------
    st.subheader("📊 Player Comparison")

    labels = ["Aces", "Double Faults", "1st Serve In", "1st Won", "2nd Won", "BP Saved"]

    playerA_stats = [a_ace, a_df, a_first_in, a_first_won, a_second_won, a_bp]
    playerB_stats = [b_ace, b_df, b_first_in, b_first_won, b_second_won, b_bp]

   

        # ---------------- PROFESSIONAL BAR GRAPH ----------------
    st.subheader("📊 Player Performance Comparison")

    labels = ["Aces", "Double Faults", "1st Serve In", "1st Won", "2nd Won", "BP Saved"]

    playerA_stats = [a_ace, a_df, a_first_in, a_first_won, a_second_won, a_bp]
    playerB_stats = [b_ace, b_df, b_first_in, b_first_won, b_second_won, b_bp]

    import plotly.graph_objects as go

    fig = go.Figure()

    # Player A bars
    fig.add_trace(go.Bar(
        x=labels,
        y=playerA_stats,
        name=playerA_name,
        marker_color='#00ffcc'
    ))

    # Player B bars
    fig.add_trace(go.Bar(
        x=labels,
        y=playerB_stats,
        name=playerB_name,
        marker_color='#ff4b4b'
    ))

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        barmode='group',
        height=400,
        title="Player Stats Comparison",
        xaxis_title="Performance Metrics",
        yaxis_title="Values",
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🎾 Project Overview")

st.sidebar.markdown("""
**Tennis Match Predictor using AI**

---

### ⚙️ Key Features
- Player vs Player prediction  
- Win probability (%)  
- Interactive graphs  

---

### 🤖 Model
- Random Forest  
- Accuracy: ~75%  

---

### 📊 Inputs
Rank, Aces, Double Faults,  
Serve Stats, Break Points  

---

### 👩‍💻 Developer
Navyasree (CSE - AI & DS)
""")

#----------history-----------------------------------

st.subheader("📜 Prediction History")

if len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)
else:
    st.write("No predictions yet.")


