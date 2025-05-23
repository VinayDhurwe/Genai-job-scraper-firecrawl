import streamlit as st
import pandas as pd
from io import BytesIO
import time
from bs4 import BeautifulSoup
from groq import Groq
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
from tavily import TavilyClient
import json
from urllib.parse import urlencode

# Firecrawl SDK
from firecrawl import FirecrawlApp

# ------------------------- JOB STATE DEFINITION -------------------------
class JobState(TypedDict):
    Title: str
    Company: str
    Experience: str
    Description: str
    is_relevant: Optional[str]
    is_competitor: Optional[str]
    job_tier: Optional[str]

# Field to search keywords map
FIELD_KEYWORDS = {
    "Data Science": "data scientist",
    "Human Resources": "human resources",
    "Digital Transformation": "digital transformation",
    "Cyber Security": "cyber security",
    "FinTech": "fintech",
    "Project Management": "project management",
    "Strategic Management": "strategic management",
    "Business Management": "business management",
    "General Management": "general management",
    "Product Management": "product management"
}
GLOBAL_FIELD: str = ""

# ------------------------- Tavily Helper Functions -------------------------
def search_with_tavily(query: str, tavily_client: TavilyClient) -> str:
    try:
        resp = tavily_client.search(query=query, max_results=1)
        if "results" in resp and resp["results"]:
            return resp["results"][0]["url"]
    except Exception:
        pass
    return ""

def get_company_career_page(company_name: str, tavily_client: TavilyClient) -> str:
    url = search_with_tavily(f"{company_name} careers", tavily_client)
    return url or search_with_tavily(company_name, tavily_client)

# ------------------------- LANGGRAPH WORKFLOW FUNCTIONS -------------------------
def check_relevance(state: JobState, client: Groq) -> JobState:
    text = (state['Title'] + ' ' + state['Description']).lower()
    # domain variants
    variants = [v.strip() for v in FIELD_KEYWORDS.get(GLOBAL_FIELD, GLOBAL_FIELD).split()] + [GLOBAL_FIELD]
    if any(kw in text for kw in variants):
        state['is_relevant'] = 'Yes'
        return state
    prompt = f"""
Job Title: {state['Title']}
Company: {state['Company']}
Description: {state['Description']}
Is this a genuine {GLOBAL_FIELD} job? Respond {{"is_relevant":"Yes" or "No"}}.
"""
    try:
        res = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.3-70b-versatile"
        )
        out = json.loads(res.choices[0].message.content)
        state['is_relevant'] = out.get('is_relevant','No')
    except Exception:
        state['is_relevant'] = 'No'
    return state

def check_competitor(state: JobState, client: Groq) -> JobState:
    competitor_list = [
        "BYJU'S","Unacademy","Vedantu","Toppr","UpGrad","Simplilearn",
        "WhiteHat Jr.","Classplus","Embibe","EduGorilla","iQuanta",
        "TrainerCentral","Meritnation","Testbook","Edukart","Adda247",
        "CollegeDekho","Leverage Edu","Next Education","Infinity Learn"
    ]
    prompt = f"Job Company: {state['Company']}\nIs it in {competitor_list}? Return {{\"is_competitor\": \"Yes\" or \"No\"}}."
    try:
        res = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.3-70b-versatile"
        )
        out = json.loads(res.choices[0].message.content)
        state['is_competitor'] = out.get('is_competitor','No')
    except Exception:
        state['is_competitor'] = 'No'
    return state

def determine_tier(state: JobState, client: Groq) -> JobState:
    prompt = f"Job Title: {state['Title']}\nExperience: {state['Experience']}\nRespond {{\"job_tier\": \"Fresher\"/\"Mid\"/\"Senior\"}}."
    try:
        res = client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model="llama-3.3-70b-versatile"
        )
        out = json.loads(res.choices[0].message.content)
        state['job_tier'] = out.get('job_tier','N/A')
    except Exception:
        state['job_tier'] = 'N/A'
    return state

def build_graph(client: Groq) -> StateGraph:
    g = StateGraph(JobState)
    g.add_node('relevance', lambda s: check_relevance(s, client))
    g.add_node('competitor', lambda s: check_competitor(s, client))
    g.add_node('tier', lambda s: determine_tier(s, client))
    g.add_edge('relevance','competitor')
    g.add_edge('competitor','tier')
    g.set_entry_point('relevance')
    g.set_finish_point('tier')
    return g

# ------------------------- JOB PROCESSING -------------------------
def process_job(job: dict, field: str, client: Groq, tavily_client: TavilyClient) -> Optional[dict]:
    global GLOBAL_FIELD
    GLOBAL_FIELD = field
    state: JobState = JobState(
        Title=job['Title'], Company=job['Company'], Experience=job['Experience'],
        Description=job['Description'], is_relevant=None, is_competitor=None, job_tier=None
    )
    result = build_graph(client).compile().invoke(state)
    if result['is_relevant'].lower()!='yes' or result['is_competitor'].lower()!='no':
        return None
    job['Job Tier'] = result['job_tier']
    link = get_company_career_page(job['Company'], tavily_client)
    if not link:
        return None
    job['Job Link'] = link
    return job

# ------------------------- RECENCY FILTER -------------------------
def is_job_recent(date_str: str) -> bool:
    low = date_str.lower()
    return any(x in low for x in ['just now','few hours','today','1 day','2 days','3 days'])

# ------------------------- FIRECRAWL SCRAPER -------------------------
def scrape_url(url: str, fc_app: FirecrawlApp) -> pd.DataFrame:
    resp = fc_app.scrape_url(url, formats=['html'], actions=[{'type':'wait','milliseconds':2000}])
    soup = BeautifulSoup(resp.html, 'html.parser')
    rows = []
    for w in soup.select('div.srp-jobtuple-wrapper'):
        rows.append({
            'Title': w.select_one('a.title').get_text(strip=True) or '',
            'Company': w.select_one('a.comp-name, a.subTitle').get_text(strip=True) or '',
            'Experience': w.select_one('span.expwdth, li.experience').get_text(strip=True) or '',
            'Description': w.select_one('span.job-desc, div.job-description').get_text(strip=True) or '',
            'Posted Date': w.select_one('span.fleft.postedDate, span.job-post-day').get_text(strip=True) or '',
            'Location': w.select_one('span.locWdth, li.location').get_text(strip=True) or '',
            'Salary': w.select_one('span.sal-wrap, li.salary').get_text(strip=True) or '',
            'Skills': ', '.join([t.get_text(strip=True) for t in w.select('li.tag, li.tag-li')]),
        })
    return pd.DataFrame(rows)

# ------------------------- DOMAIN SCRAPE & PROCESS -------------------------
def scrape_jobs_for_domain(domain: str, client: Groq, tavily_client: TavilyClient, fc_app: FirecrawlApp) -> pd.DataFrame:
    GLOBAL_FIELD = domain
    keyword = FIELD_KEYWORDS[domain]
    url = f"https://www.naukri.com/jobs-in-india?k={keyword}&l=india&jobAge=3"
    df_raw = scrape_url(url, fc_app)
    processed = []
    for job in df_raw.to_dict('records'):
        p = process_job(job, domain, client, tavily_client)
        if p and is_job_recent(job['Posted Date']):
            processed.append(p)
    return pd.DataFrame(processed)

# ------------------------- EXCEL EXPORT -------------------------
def to_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        df.to_excel(w, index=False)
    return buf.getvalue()

# ------------------------- STREAMLIT APP -------------------------
def main():
    st.set_page_config(page_title='Job Scraper', layout='wide')
    st.title('🌐 Job Scraper - Domain Specific')
    groq_key = st.text_input('Groq API Key', type='password')
    tavily_key = st.text_input('Tavily API Key', type='password')
    fc_key = st.text_input('Firecrawl API Key', type='password')
    if not groq_key or not tavily_key or not fc_key:
        st.warning('Enter all API keys')
        return
    client = Groq(api_key=groq_key)
    tavily = TavilyClient(tavily_key)
    fc_app = FirecrawlApp(api_key=fc_key)
    domain = st.selectbox('Select Domain', list(FIELD_KEYWORDS.keys()))
    if st.button('🔍 Scrape Jobs'):
        with st.spinner(f"Scraping {domain} jobs up to 3 days..."):
            df = scrape_jobs_for_domain(domain, client, tavily, fc_app)
        if df.empty:
            st.info('No relevant jobs found')
        else:
            st.dataframe(df)
            data = to_excel(df)
            st.download_button('📥 Download Excel', data, file_name=f"jobs_{domain}.xlsx")

if __name__=='__main__':
    main()
