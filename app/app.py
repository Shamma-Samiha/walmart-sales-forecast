import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from datetime import datetime

# -----------------------------
# 🎨 Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecast",
    page_icon="🛒",
    layout="wide",
)

# -----------------------------
# 🧠 Load Trained Model
# -----------------------------
model = joblib.load("src/random_forest_model.pkl")

# -----------------------------
# 🏷️ App Header
# -----------------------------
st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>🛍️ Walmart Weekly Sales Forecast</h1>
    <p style='text-align: center; font-size: 18px;'>
        Predict <b>weekly sales</b> for Walmart stores using temperature, fuel price, CPI, unemployment, and seasonal factors.<br>
        Designed for business users to gain quick, data-driven insights.
    </p>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# 📥 Sidebar Inputs
# -----------------------------
st.sidebar.header("🧩 Input Parameters")
store = st.sidebar.number_input("🏪 Store ID", 1, 50, 1)
holiday_flag = st.sidebar.selectbox("🎉 Is it a Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x else "No")
temperature = st.sidebar.slider("🌡️ Temperature (°F)", 20.0, 120.0, 60.0)
fuel_price = st.sidebar.slider("⛽ Fuel Price ($)", 2.0, 5.0, 3.5)
cpi = st.sidebar.number_input("📈 CPI", 100.0, 250.0, 180.0)
unemployment = st.sidebar.slider("💼 Unemployment Rate", 0.0, 15.0, 7.5)
month = st.sidebar.slider("🗓️ Month", 1, 12, 5)
year = st.sidebar.selectbox("📅 Year", [2010, 2011, 2012])
week = st.sidebar.slider("📆 Week of Year", 1, 52, 25)
store_avg_sales = st.sidebar.number_input("💰 Store Average Sales ($)", 0.0, 3_000_000.0, 1_000_000.0)

# -----------------------------
# 🧮 Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    "Store": store,
    "Holiday_Flag": holiday_flag,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
    "Month": month,
    "Year": year,
    "Week": week,
    "Store_Avg_Sales": store_avg_sales
}])

# -----------------------------
# 🔮 Predict
# -----------------------------
if st.button("🚀 Predict Weekly Sales", use_container_width=True):
    prediction = model.predict(input_data)[0]
    delta = ((prediction - store_avg_sales) / store_avg_sales) * 100

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Predicted Weekly Sales", f"${prediction:,.2f}", delta=f"{delta:+.2f}%")
    with col2:
        st.metric("🏪 Store Average Sales", f"${store_avg_sales:,.2f}")

    st.divider()
    st.markdown("## 📊 Visual Insights")

    # -----------------------------
    # 📈 Chart 1: Predicted vs Average
    # -----------------------------
    colA, colB = st.columns(2)
    with colA:
        chart_data = pd.DataFrame({
            'Type': ['Predicted', 'Average'],
            'Sales': [prediction, store_avg_sales]
        })
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(data=chart_data, x='Type', y='Sales', palette=['#00BFFF', '#FFD700'], ax=ax1)
        ax1.set_title("Predicted vs Average Weekly Sales")
        ax1.set_ylabel("Sales ($)")
        st.pyplot(fig1)
        st.caption("💡 Comparison between predicted sales and store average.")

    # -----------------------------
    # 📈 Chart 2: Simulated Monthly Trend
    # -----------------------------
    with colB:
        months = list(range(1, 13))
        simulated_sales = [store_avg_sales * (0.9 + np.sin(m/2)/10) for m in months]
        simulated_sales[-1] = prediction
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(months, simulated_sales, marker='o', color='#32CD32')
        ax2.set_title("Simulated Monthly Sales Trend")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Sales ($)")
        st.pyplot(fig2)
        st.caption("📈 Simulated yearly sales pattern for the store.")

    # -----------------------------
    # 🔥 Chart 3: Simulated Feature Heatmap
    # -----------------------------
    st.markdown("### 🔥 Feature Relationships (Simulated Heatmap)")
    simulated_data = pd.DataFrame({
        'Temperature': np.random.uniform(40, 90, 10),
        'Fuel_Price': np.random.uniform(2.0, 4.0, 10),
        'CPI': np.random.uniform(150, 230, 10),
        'Unemployment': np.random.uniform(5, 10, 10),
        'Weekly_Sales': np.random.uniform(500000, 2000000, 10)
    })
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.heatmap(simulated_data.corr(), annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)
    st.caption("🧠 Relationships between sales and economic indicators (simulated).")

    # -----------------------------
    # 🧾 Business Summary
    # -----------------------------
    st.markdown("## 🧾 Business Summary")

    if delta > 10:
        summary = f"Sales are expected to increase by **{delta:.2f}%**, suggesting a strong performance week for Store {store}."
    elif delta < -10:
        summary = f"Sales are projected to fall by **{abs(delta):.2f}%**, indicating a potential slowdown in weekly revenue."
    else:
        summary = f"Sales are likely to remain steady with a **{delta:.2f}%** change compared to the average."

    if holiday_flag == 1:
        summary += " Since it’s a **holiday week**, increased foot traffic and higher purchase volumes are likely."
    if temperature > 90:
        summary += " Hot weather conditions could influence consumer activity or logistics."
    if unemployment > 10:
        summary += " A higher unemployment rate might slightly dampen overall sales."

    st.success(summary)

    # -----------------------------
    # 🪄 Create Branded PDF with Cover Page
    # -----------------------------
    def create_pdf_with_cover(store, year, week, prediction, store_avg_sales, delta, summary):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)

        # COVER PAGE
        c.setFillColorRGB(0.0, 0.45, 0.85)
        c.rect(0, 0, 8.5 * inch, 11 * inch, fill=1)

        # Logo-like yellow squares
        c.setFillColorRGB(1.0, 0.84, 0.0)
        c.rect(1.4 * inch, 8.6 * inch, 0.3 * inch, 0.3 * inch, fill=1)
        c.rect(1.9 * inch, 8.6 * inch, 0.3 * inch, 0.3 * inch, fill=1)
        c.rect(2.4 * inch, 8.6 * inch, 0.3 * inch, 0.3 * inch, fill=1)

        # Title
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(4.25 * inch, 7.9 * inch, "Walmart Weekly Sales Forecast")

        c.setFont("Helvetica", 16)
        c.drawCentredString(4.25 * inch, 7.2 * inch, "Data Science Project by Shamma Samiha")
        c.setFont("Helvetica", 12)
        c.drawCentredString(4.25 * inch, 6.6 * inch, "Generated on " + datetime.now().strftime("%B %d, %Y"))
        c.linkURL("https://www.linkedin.com/in/shamma-samiha", (2.5 * inch, 6.45 * inch, 6 * inch, 6.7 * inch))
        c.setFont("Helvetica", 10)
        c.drawCentredString(4.25 * inch, 0.7 * inch, "Walmart | Data Science Portfolio Report")
        c.showPage()

        # REPORT PAGE
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(4.25 * inch, 10.5 * inch, "Walmart Weekly Sales Forecast Report")

        y = 10.0 * inch
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.0, 0.45, 0.85)
        c.drawString(1 * inch, y - 0.4 * inch, "Store Information")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
        c.drawString(1.2 * inch, y - 0.7 * inch, f"Store ID: {store}")
        c.drawString(1.2 * inch, y - 0.9 * inch, f"Year: {year}")
        c.drawString(1.2 * inch, y - 1.1 * inch, f"Week: {week}")

        c.line(1 * inch, y - 1.25 * inch, 7.5 * inch, y - 1.25 * inch)

        c.setFillColorRGB(0.0, 0.45, 0.85)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, y - 1.6 * inch, "Sales Prediction Results")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
        c.drawString(1.2 * inch, y - 1.9 * inch, f"Predicted Weekly Sales: ${prediction:,.2f}")
        c.drawString(1.2 * inch, y - 2.1 * inch, f"Store Average Sales: ${store_avg_sales:,.2f}")
        c.drawString(1.2 * inch, y - 2.3 * inch, f"Change vs. Average: {delta:+.2f}%")

        c.line(1 * inch, y - 2.45 * inch, 7.5 * inch, y - 2.45 * inch)

        c.setFillColorRGB(0.0, 0.45, 0.85)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, y - 2.8 * inch, "Business Summary")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 11)
        text_obj = c.beginText(1.2 * inch, y - 3.1 * inch)
        text_obj.setLeading(14)
        for line in summary.split(". "):
            text_obj.textLine(line.strip() + ".")
        c.drawText(text_obj)

        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(4.25 * inch, 0.7 * inch, "Generated by Shamma Samiha | Data Science Portfolio Project")
        c.linkURL("https://www.linkedin.com/in/shamma-samiha", (2.5 * inch, 0.65 * inch, 6 * inch, 0.8 * inch))

        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf_with_cover(store, year, week, prediction, store_avg_sales, delta, summary)
    st.download_button(
        label="📥 Download Branded Report (PDF)",
        data=pdf_buffer,
        file_name=f"Walmart_Sales_Report_Store{store}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

else:
    st.info("⬅️ Adjust sidebar inputs and click **Predict Weekly Sales** to generate forecasts and insights.")

# -----------------------------
# 💬 Footer
# -----------------------------
st.markdown("""
    <hr>
    <p style='text-align: center; color: grey;'>
        Built with ❤️ using <b>Streamlit</b> | Created by 
        <a href="https://www.linkedin.com/in/shamma-samiha" target="_blank" style="color:#00BFFF; text-decoration:none;">
        <b>Shamma Samiha</b></a>
    </p>
""", unsafe_allow_html=True)
