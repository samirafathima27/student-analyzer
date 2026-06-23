# 🎓 EduLens — AI Powered Student Performance Analyzer

An intelligent web application that analyzes student performance data, predicts at-risk students using Machine Learning, and generates professional reports using AI.

Live Demo: [your-streamlit-url-here]


# 🚀 What it does


📊 Overview Dashboard — KPI cards, subject averages, class insights
📈 Interactive Charts — Score distribution, pass/fail breakdown, gender comparison
🔍 Deep Analysis — Gender, test prep, parental education impact
🤖 ML Prediction — Random Forest model predicts if a student will pass or fail
📝 AI Report — Groq LLM (Llama 3.3 70B) writes a full professional class report
📥 Download Report — Save the AI report as a text file

# Tech Stack

Python Core                 programming language
Pandas Data                 cleaning and analysis
Scikit-learn                Random Forest ML model
Plotly                      Interactive charts 
Groq API (Llama 3.3 70B)    AI report generationStreamlit


# 📁 Project Structure

student-analyzer/

├── app.py                      # Streamlit UI
├── analyzer.py                 # Pandas + ML logic
├── report.py                   # Groq LLM report generation
├── StudentsPerformance.csv     # Dataset
├── requirements.txt            # Dependencies
└── README.md


# ⚙️ Setup & Run Locally

1. Clone the repo

bashgit clone https://github.com/samirafathima27/student-analyzer.git
cd student-analyzer

2. Install dependencies

bashpip install -r requirements.txt

3. Add API key

Create a .env file:

GROQ_API_KEY=your-groq-key-here

Get your free key at: https://console.groq.com

4. Run

bashstreamlit run app.py

Opens at http://localhost:8501


# 📊 Dataset

Uses the Students Performance in Exams dataset from Kaggle.


1000 student records
Features: gender, race/ethnicity, parental education, lunch, test preparation
Scores: math, reading, writing



# 🔑 Key Insights from Data


Students who completed test preparation scored significantly higher
Female students outperform male students across all subjects
Parental education level positively correlates with student performance
Standard lunch students perform better than free/reduced lunch students
