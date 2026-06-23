import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_report(overview, gender_analysis, test_prep_analysis, accuracy):
    """Generate AI written report using Groq"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are an educational data analyst. Based on the following student performance data, write a clear, professional, and insightful report in simple English.

CLASS OVERVIEW:
- Total Students: {overview['total_students']}
- Pass Count: {overview['pass_count']}
- Fail Count: {overview['fail_count']}
- Average Math Score: {overview['avg_math']}
- Average Reading Score: {overview['avg_reading']}
- Average Writing Score: {overview['avg_writing']}
- Overall Average: {overview['avg_total']}

GENDER ANALYSIS:
{gender_analysis.to_string()}

TEST PREPARATION IMPACT:
{test_prep_analysis.to_string()}

ML MODEL ACCURACY: {accuracy}%

Write a report with these sections:
1. Executive Summary (3-4 lines)
2. Key Findings (5 bullet points)
3. Areas of Concern
4. Recommendations for Teachers
5. Conclusion

Keep it clear, professional and actionable.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1500,
    )

    return response.choices[0].message.content