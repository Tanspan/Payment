pip install streamlit pandas matplotlib seaborn scikit-learn joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set config
st.set_page_config(page_title="SmartPay Recommender", layout="wide")
st.title("💳 SmartPay: Personalized Payment Recommendation Viewer")

# Load predicted results
try:
    df = pd.read_csv("Final_User_Recommendations.csv")
except:
    st.error("❌ Could not load 'Final_User_Recommendations.csv'. Please ensure it exists.")
    st.stop()

# ===============================
# 📊 Pie Charts - Category & Method
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛍 Transaction Categories")
    cat_counts = df['Category'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(cat_counts, labels=cat_counts.index, autopct="%1.1f%%", startangle=140, shadow=True)
    st.pyplot(fig1)

with col2:
    st.subheader("💳 Recommended Payment Methods")
    method_counts = df['Recommended_Payment_Method'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(method_counts, labels=method_counts.index, autopct="%1.1f%%", startangle=140, shadow=True)
    st.pyplot(fig2)

# ===============================
# 📊 Side-by-Side: Bar & Line Plot
# ===============================
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔍 Avg Confidence per Method")
    conf_plot = df.groupby("Recommended_Payment_Method")["Confidence (%)"].mean().sort_values()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=conf_plot.values, y=conf_plot.index, palette="viridis", ax=ax3)
    ax3.set_xlabel("Average Confidence (%)")
    ax3.set_title("Model Confidence per Method")
    st.pyplot(fig3)

with col4:
    st.subheader("🔁 Recommendations Across Transactions")
    method_mapping = {m: i for i, m in enumerate(df["Recommended_Payment_Method"].unique())}
    df["Method_ID"] = df["Recommended_Payment_Method"].map(method_mapping)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(df.index, df["Method_ID"], marker='o', linestyle='-', color='steelblue', label="Recommended")
    ax4.scatter(df.index[-1], df["Method_ID"].iloc[-1], color='red', s=100, label="Next Recommendation")
    ax4.set_yticks(list(method_mapping.values()))
    ax4.set_yticklabels(list(method_mapping.keys()))
    ax4.set_xlabel("Transaction Index")
    ax4.set_ylabel("Recommended Method")
    ax4.set_title("Transaction-wise Recommendations")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

# ===============================
# ✅ Final Recommendation
# ===============================
final_rec = df["Recommended_Payment_Method"].iloc[-1]
final_conf = df["Confidence (%)"].iloc[-1]
st.markdown(f"""
### ✅ Final Recommendation for Upcoming Transaction:
- Recommended Method: 🟢 {final_rec}
- Confidence: {final_conf}%
""")

# Footer
st.markdown("---")
st.caption("© 2025 SmartPay")
