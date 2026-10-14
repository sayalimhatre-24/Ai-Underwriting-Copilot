"""
AI-Powered Term-Insurance Underwriting Automation (Prototype)
Author: Anurag Kumar | HDFC Life Insurance Co. Ltd. Internship Project

Demonstrates a two-step AI pipeline for underwriting support:
  Step 1: Extraction  - pulls structured applicant data out of raw
                         "document text" (stands in for OCR output).
  Step 2: Risk Scoring - feeds the structured data to Groq acting as
                         an underwriter, producing a risk category and
                         recommended action.

This is a Level 1 (demonstration in context) prototype using synthetic
applicant scenarios - no real customer data is used or required.

Setup:
    pip install groq pandas openpyxl
    Set your API key below or as an environment variable GROQ_API_KEY.
    Get a free key at https://console.groq.com/keys
"""

import os
import time
import json
import webbrowser
import pandas as pd
from groq import Groq

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key")
MODEL = "llama-3.3-70b-versatile"
OUTPUT_FILE = "Underwriting_Recommendations_Output.xlsx"
OUTPUT_HTML = "Underwriting_Dashboard.html"

client = Groq(api_key=API_KEY)

# ---------------------------------------------------------------------------
# 2. SYNTHETIC TEST CASES
#    Stand-ins for KYC / income proof / medical report document text.
#    In a fuller build these would come from real OCR output; here they
#    represent the text an OCR step would have extracted from documents.
# ---------------------------------------------------------------------------

SYNTHETIC_CASES = [
    {
        "applicant_id": "APP-001",
        "raw_document_text": """
            KYC: Name - Rohit Sharma, DOB - 14/03/1988, Address - Pune, Maharashtra.
            Salary Slip: Employer - TCS Ltd, Monthly Gross Salary - Rs. 1,45,000.
            Medical Report: BMI 23.4, Blood Pressure 118/76, No pre-existing conditions declared.
            Sum Assured Requested: Rs. 1,00,00,000
        """,
    },
    {
        "applicant_id": "APP-002",
        "raw_document_text": """
            KYC: Name - Anita Deshmukh, DOB - 02/11/1975, Address - Nagpur, Maharashtra.
            Bank Statement: Average Monthly Balance - Rs. 38,000, irregular deposits noted.
            Medical Report: BMI 31.2, Blood Pressure 148/94, Type 2 Diabetes (diagnosed 2019), smoker.
            Sum Assured Requested: Rs. 75,00,000
        """,
    },
    {
        "applicant_id": "APP-003",
        "raw_document_text": """
            KYC: Name - Vikram Nair, DOB - 22/07/1995, Address - Kochi, Kerala.
            ITR: Declared Annual Income - Rs. 9,60,000 (FY24).
            Medical Report: BMI 21.8, Blood Pressure 112/72, No pre-existing conditions.
            Sum Assured Requested: Rs. 50,00,000
        """,
    },
    {
        "applicant_id": "APP-004",
        "raw_document_text": """
            KYC: Name - Sunita Rao, DOB - 09/01/1968, Address - Bengaluru, Karnataka.
            Salary Slip: Employer - Self-employed consultant, declared monthly income Rs. 2,10,000.
            Medical Report: BMI 27.9, Blood Pressure 136/88, Hypertension (controlled with medication).
            Sum Assured Requested: Rs. 1,50,00,000
        """,
    },
    {
        "applicant_id": "APP-005",
        "raw_document_text": """
            KYC: Name - Farhan Ali, DOB - 30/05/2000, Address - Lucknow, Uttar Pradesh.
            Salary Slip: Employer - Infosys, Monthly Gross Salary - Rs. 62,000.
            Medical Report: BMI 22.0, Blood Pressure 120/78, No pre-existing conditions.
            Sum Assured Requested: Rs. 40,00,000
        """,
    },
]

# ---------------------------------------------------------------------------
# 3. PROMPT TEMPLATES
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_TEMPLATE = """You are an OCR/data-extraction assistant for an insurance underwriting system.

Extract the following fields from the raw document text below and return ONLY a valid JSON
object with these exact keys (use "Not Available" if a field is missing):
- name
- age_estimate
- declared_monthly_income
- bmi
- blood_pressure
- existing_conditions
- lifestyle_flags
- sum_assured_requested

RAW DOCUMENT TEXT:
{raw_text}

Return ONLY the JSON object, no other text, no markdown formatting.
"""

RISK_PROMPT_TEMPLATE = """You are a senior underwriter at a life insurance company evaluating a term
insurance application.

APPLICANT PROFILE (extracted from submitted documents):
{profile_json}

TASK:
Evaluate this applicant and produce your assessment in exactly this three-part plain text
format, with these exact headers and no markdown formatting (no asterisks, no bold, no
headings, no bullet points):

RISK CATEGORY: <LOW / MEDIUM / HIGH>

RECOMMENDED ACTION: <APPROVE / REFER TO HUMAN UNDERWRITER / DECLINE>

REASONING: <100-150 words explaining the risk category and recommended action, covering
income-to-sum-assured ratio reasonableness, BMI/blood pressure/medical flags, and any
lifestyle risk factors. Flag any case with significant health risk factors or income/sum
assured mismatch for human underwriter review rather than auto-approving or auto-declining.>
"""

# ---------------------------------------------------------------------------
# 4. CORE FUNCTIONS
# ---------------------------------------------------------------------------

def _call_with_retry(prompt, label, max_retries=5):
    """Calls Groq with retry-with-backoff on 429 rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content, None
        except Exception as e:
            error_text = str(e)
            is_rate_limit = "429" in error_text or "rate_limit" in error_text.lower()
            if is_rate_limit and attempt < max_retries:
                wait_time = 15 * attempt
                print(f"  Rate limit hit for {label}, waiting {wait_time}s before retry "
                      f"({attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            return None, error_text


def extract_fields(raw_text, label):
    """Step 1: turns raw 'document text' into structured JSON fields."""
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(raw_text=raw_text)
    text, error = _call_with_retry(prompt, label)
    if error:
        return {"error": f"Extraction failed - {error}"}
    try:
        cleaned = text.strip().strip("```").replace("json", "", 1).strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"error": f"Could not parse JSON - {e}", "raw_text": text}


def assess_risk(profile, label):
    """Step 2: feeds structured profile to Gemini acting as an underwriter."""
    prompt = RISK_PROMPT_TEMPLATE.format(profile_json=json.dumps(profile, indent=2))
    text, error = _call_with_retry(prompt, label)
    if error:
        return f"RISK CATEGORY: ERROR\nRECOMMENDED ACTION: N/A\nREASONING: API call failed - {error}"
    return text


def parse_risk_response(raw_text):
    """Tolerant of markdown formatting (e.g. **RISK CATEGORY:**)."""
    risk_category, action, reasoning = "", "", ""
    current_section = None
    for line in raw_text.splitlines():
        stripped = line.strip().lstrip("#").strip().strip("*").strip()
        upper = stripped.upper()
        if upper.startswith("RISK CATEGORY:"):
            risk_category = stripped.split(":", 1)[1].strip()
            current_section = "risk"
        elif upper.startswith("RECOMMENDED ACTION:"):
            action = stripped.split(":", 1)[1].strip()
            current_section = "action"
        elif upper.startswith("REASONING:"):
            reasoning = stripped.split(":", 1)[1].strip()
            current_section = "reasoning"
        elif stripped and current_section == "reasoning":
            reasoning += " " + stripped
    if not risk_category and not action and not reasoning:
        reasoning = raw_text.strip().replace("\n", " ")[:2000]
        risk_category = "UNPARSED"
    return risk_category.upper().strip("*").strip(), action.upper().strip("*").strip(), reasoning.strip()


# ---------------------------------------------------------------------------
# 5. DASHBOARD GENERATION
# ---------------------------------------------------------------------------

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Underwriting — AI Recommendations Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#F5F6FA; --card:#FFFFFF; --border:#E4E7EC;
    --text:#101828; --text-dim:#667085;
    --indigo:#4F46E5; --green:#16A34A; --amber:#D97706; --red:#DC2626;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:'Segoe UI',Inter,system-ui,-apple-system,sans-serif;}
  .wrap{max-width:1200px;margin:0 auto;padding:40px 28px 60px;}
  header{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px;}
  h1{font-size:24px;font-weight:600;margin:0;}
  .sub{color:var(--text-dim);font-size:13px;margin-top:4px;}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
  .kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:20px;box-shadow:0 1px 2px rgba(16,24,40,0.04);}
  .kpi .label{color:var(--text-dim);font-size:12px;text-transform:uppercase;letter-spacing:0.4px;}
  .kpi .num{font-size:28px;font-weight:700;margin-top:6px;}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
  @media (max-width:900px){.grid{grid-template-columns:1fr;} .kpis{grid-template-columns:1fr 1fr;}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:22px;box-shadow:0 1px 2px rgba(16,24,40,0.04);}
  .card h3{margin:0 0 14px;font-size:14px;font-weight:600;color:var(--text);}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  thead th{text-align:left;color:var(--text-dim);font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:0.3px;padding:10px 12px;border-bottom:1px solid var(--border);}
  tbody td{padding:12px;border-bottom:1px solid var(--border);vertical-align:top;}
  tbody tr:hover{background:#FAFBFF;}
  .tbl-wrap{max-height:480px;overflow:auto;}
  .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
  .badge.low,.badge.approve{background:#DCFCE7;color:var(--green);}
  .badge.medium,.badge.refer{background:#FEF3C7;color:var(--amber);}
  .badge.high,.badge.decline,.badge.error{background:#FEE2E2;color:var(--red);}
  .badge.na{background:#F2F4F7;color:var(--text-dim);}
  .reasoning{max-width:360px;color:var(--text-dim);font-size:12.5px;line-height:1.5;}
  canvas{max-height:230px;}
  .generated{color:var(--text-dim);font-size:11px;margin-top:24px;text-align:center;}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Underwriting — AI Recommendations</h1>
      <div class="sub">Auto-generated from Underwriting_Recommendations_Output.xlsx</div>
    </div>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="label">Applicants Processed</div><div class="num" id="k-total">0</div></div>
    <div class="kpi"><div class="label">Approve</div><div class="num" id="k-approve">0</div></div>
    <div class="kpi"><div class="label">Refer</div><div class="num" id="k-refer">0</div></div>
    <div class="kpi"><div class="label">Decline</div><div class="num" id="k-decline">0</div></div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Risk Category Split</h3>
      <canvas id="chart-risk"></canvas>
    </div>
    <div class="card">
      <h3>Recommended Action Split</h3>
      <canvas id="chart-action"></canvas>
    </div>
  </div>

  <div class="card">
    <h3>Applicant-by-Applicant Results</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Applicant</th><th>Risk Category</th><th>Recommended Action</th><th>Reasoning</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="generated">Generated automatically after script run · __TIMESTAMP__</div>
</div>

<script>
const rows = __DATA_JSON__;

function riskClass(v){
  v=(v||'').toUpperCase();
  if(v==='LOW') return 'low'; if(v==='MEDIUM') return 'medium';
  if(v==='HIGH') return 'high'; if(v==='ERROR'||v==='UNPARSED') return 'error';
  return 'na';
}
function actionClass(v){
  v=(v||'').toUpperCase();
  if(v.includes('APPROVE')) return 'approve';
  if(v.includes('REFER')) return 'refer';
  if(v.includes('DECLINE')) return 'decline';
  return 'na';
}

const tbody = document.getElementById('tbody');
const riskCounts = {LOW:0,MEDIUM:0,HIGH:0,OTHER:0};
let approve=0, refer=0, decline=0;

rows.forEach(r=>{
  const risk = (r.Risk_Category||'').toUpperCase();
  if(riskCounts[risk]!==undefined) riskCounts[risk]++; else if(risk) riskCounts.OTHER++;
  const action = (r.Recommended_Action||'').toUpperCase();
  if(action.includes('APPROVE')) approve++;
  else if(action.includes('REFER')) refer++;
  else if(action.includes('DECLINE')) decline++;

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><strong>${r.Applicant_ID||''}</strong></td>
    <td><span class="badge ${riskClass(r.Risk_Category)}">${r.Risk_Category||'—'}</span></td>
    <td><span class="badge ${actionClass(r.Recommended_Action)}">${r.Recommended_Action||'—'}</span></td>
    <td class="reasoning">${r.Reasoning||''}</td>
  `;
  tbody.appendChild(tr);
});

document.getElementById('k-total').textContent = rows.length;
document.getElementById('k-approve').textContent = approve;
document.getElementById('k-refer').textContent = refer;
document.getElementById('k-decline').textContent = decline;

new Chart(document.getElementById('chart-risk'), {
  type:'doughnut',
  data:{labels:['Low','Medium','High','Other'],
    datasets:[{data:[riskCounts.LOW,riskCounts.MEDIUM,riskCounts.HIGH,riskCounts.OTHER],
      backgroundColor:['#16A34A','#D97706','#DC2626','#94A3B8']}]},
  options:{plugins:{legend:{position:'bottom'}}}
});

new Chart(document.getElementById('chart-action'), {
  type:'bar',
  data:{labels:['Approve','Refer','Decline'],
    datasets:[{data:[approve,refer,decline],backgroundColor:['#16A34A','#D97706','#DC2626'],borderRadius:6}]},
  options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,ticks:{precision:0}}}}
});
</script>
</body>
</html>
"""


def generate_dashboard(results, timestamp):
    """Builds a self-contained HTML dashboard with the results embedded directly
    (no manual file upload needed) and opens it automatically in the browser."""
    html = DASHBOARD_TEMPLATE.replace("__DATA_JSON__", json.dumps(results))
    html = html.replace("__TIMESTAMP__", timestamp)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {OUTPUT_HTML} — opening in browser...")
    webbrowser.open("file://" + os.path.abspath(OUTPUT_HTML))


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    results = []
    for i, case in enumerate(SYNTHETIC_CASES):
        print(f"Processing {case['applicant_id']} ({i + 1}/{len(SYNTHETIC_CASES)})...")

        # Step 1: Extraction
        profile = extract_fields(case["raw_document_text"], case["applicant_id"])

        # Step 2: Risk scoring
        raw_output = assess_risk(profile, case["applicant_id"])
        risk_category, action, reasoning = parse_risk_response(raw_output)

        results.append({
            "Applicant_ID": case["applicant_id"],
            "Extracted_Profile": json.dumps(profile),
            "Risk_Category": risk_category,
            "Recommended_Action": action,
            "Reasoning": reasoning,
        })

        time.sleep(5)  # delay between calls to stay within free-tier rate limits

    output_df = pd.DataFrame(results)
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(results)} applicants processed. Output saved to {OUTPUT_FILE}")

    print("\nSummary:")
    print(output_df["Risk_Category"].value_counts().to_string())
    print()
    print(output_df["Recommended_Action"].value_counts().to_string())

    generate_dashboard(results, time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
