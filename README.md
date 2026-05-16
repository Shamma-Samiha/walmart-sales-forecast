## 🛒 Walmart Weekly Sales Forecast

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project%20Stage-Production%20Ready-brightgreen)

A data science project predicting **weekly sales for Walmart stores** using economic, seasonal, and store-level factors such as temperature, fuel price, CPI, unemployment, and holidays.  

Built with 💙 **Python, Streamlit, and Machine Learning (Random Forest)**.

---

## 📊 Project Overview

This project analyzes Walmart’s historical sales data and builds a **predictive model** that estimates future weekly sales for each store.  
It also includes a **Streamlit dashboard** for real-time forecasting and a **branded PDF report generator** for decision-makers.

## Live Demo : https://walmart-sales-forecast.streamlit.app/
---

## 🧠 Key Features

| Feature | Description |
|----------|-------------|
| 🔮 **Sales Forecasting** | Predict weekly sales using trained Random Forest Regressor |
| 📊 **Interactive Dashboard** | Streamlit-based web app for real-time prediction |
| 🧾 **Auto-Generated Report** | Professionally branded PDF with business summary |
| 📈 **Visual Insights** | Monthly trends, feature correlations, predicted vs average charts |
| 🧩 **Business Explanation** | AI-generated summary written for non-technical users |

---

## 🧰 Tech Stack

- **Python 3.13**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Streamlit**
- **ReportLab** (for PDF generation)

---

## 📂 Project Structure

```

retail-sales-forecast/
│
├── app/
│   └── app.py               # Streamlit web app
│
├── data/
│   ├── raw/                 # Original Walmart dataset
│   └── processed/           # Cleaned dataset
│
├── notebook/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
│
├── src/
│   ├── data_prep.py         # Data preprocessing
│   ├── modeling.py          # Model training and evaluation
│   └── random_forest_model.pkl  # Trained model file
│
├── README.md
└── requirements.txt

````

---

## 🖥️ How to Run the App

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Shamma-Samiha/walmart-sales-forecast.git
cd walmart-sales-forecast
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch Streamlit App

```bash
streamlit run app/app.py
```

### 4️⃣ Use the App

* Enter store information and parameters.
* Click **Predict Weekly Sales**.
* View interactive charts and download the **PDF report**.

---

---

## 📸 Website View 

### 🌐 Streamlit Dashboard  
Here’s how the interactive Walmart Weekly Sales Prediction app looks in action 👇  

![Dashboard View 1](assets/dashboard%2001.png)
![Dashboard View 2](assets/Dashboard%2002.png)
![Dashboard View 3](assets/Dashboard%2003.png)

---

### 🧾 Generated PDF Report  
This automatically generated report summarizes key predictions and insights for decision-makers 👇  

![Report Page 1](assets/Report%201.png)
![Report Page 2](assets/report%202.png)


---

## 🧩 Machine Learning Performance

| Model                 | MAE       | RMSE       | R²    |
| --------------------- | --------- | ---------- | ----- |
| **Linear Regression** | 96,000.51 | 154,733.99 | 0.926 |
| **Random Forest**     | 50,124.47 | 79,608.10  | 0.980 |

✅ Random Forest Model achieved an **R² = 0.98**, demonstrating excellent prediction accuracy.

---

## 📈 Example Business Insight

> “Sales are expected to increase by **+5.85 %** next week, indicating steady store performance.
> As this is a holiday period, higher customer turnout and sales volume are expected.”

---

## 👩‍💻 Author

**Shamma Samiha**
📍 Daffodil International University, Bangladesh
🔗 [LinkedIn Profile](https://www.linkedin.com/in/shamma-samiha)

---

## 🏆 Project Highlights

* Real-world case study: **Retail Demand Forecasting**
* Focused on **Explainability**, **Usability**, and **Business Communication**

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you found this project helpful, please consider **starring the repository 🌟** on GitHub — it helps others discover this project!

