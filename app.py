# ----------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import json
import re

# ----------------------------------------------------------------------------
# 2. CONFIGURATION & SETUP
# ----------------------------------------------------------------------------
# --- Gemini API Configuration ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = "AIzaSyBYrmSyTAWM0AVqdSFSYbG__YPdL6eMPtI" # <--- PASTE YOUR API KEY HERE

genai.configure(api_key=GEMINI_API_KEY)

# --- Jobs API Configuration ---
# Get free API keys from: https://developer.adzuna.com/
try:
    ADZUNA_APP_ID = st.secrets.get("ADZUNA_APP_ID", "")
    ADZUNA_APP_KEY = st.secrets.get("ADZUNA_APP_KEY", "")
except:
    ADZUNA_APP_ID = ""  # Add your Adzuna App ID here (optional)
    ADZUNA_APP_KEY = ""  # Add your Adzuna App Key here (optional)

# --- Web Scraper Configuration ---
SCRAPER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ----------------------------------------------------------------------------
# 3. CORE & ADVANCED FUNCTIONS
# ----------------------------------------------------------------------------

# === DATA LAYER (Updated to use Adzuna API) ===
def scrape_live_jobs(job_title, location="Bengaluru"):
    """
    Fetch jobs from Adzuna API (free tier) or fall back to RemoteOK for remote jobs
    """
    job_list = []
    
    # Option 1: Try Adzuna API (requires free API key)
    # You can get a free API key from https://developer.adzuna.com/
    try:
        # Using Adzuna API with configured credentials
        app_id = ADZUNA_APP_ID
        app_key = ADZUNA_APP_KEY
        
        # Convert location to country code (simplified)
        country = "in" if location.lower() in ["bengaluru", "bangalore", "mumbai", "delhi", "india"] else "us"
        
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            'app_id': app_id,
            'app_key': app_key,
            'what': job_title,
            'where': location,
            'results_per_page': 20,
            'sort_by': 'relevance'
        }
        
        # Skip Adzuna if no API keys provided
        if not app_id or not app_key:
            raise Exception("No Adzuna API key provided")
            
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        for job in data.get('results', []):
            job_list.append({
                'title': job.get('title', 'N/A'),
                'company': job.get('company', {}).get('display_name', 'N/A'),
                'description': job.get('description', 'N/A')[:500] + '...' if len(job.get('description', '')) > 500 else job.get('description', 'N/A'),
                'location': job.get('location', {}).get('display_name', location)
            })
            
    except Exception as e:
        st.warning(f"Adzuna API unavailable, trying RemoteOK: {str(e)}")
        
        # Option 2: Fall back to RemoteOK for remote jobs
        try:
            url = "https://remoteok.io/api"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()[1:]  # Skip the first item (metadata)
            
            # Filter jobs by title keywords
            job_keywords = job_title.lower().split()
            for job in data[:20]:  # Limit to 20 jobs
                job_title_text = job.get('position', '').lower()
                if any(keyword in job_title_text for keyword in job_keywords):
                    job_list.append({
                        'title': job.get('position', 'N/A'),
                        'company': job.get('company', 'N/A'),
                        'description': job.get('description', 'N/A')[:500] + '...' if len(job.get('description', '')) > 500 else job.get('description', 'N/A'),
                        'location': 'Remote'
                    })
                    
        except Exception as e2:
            st.warning(f"RemoteOK also failed, trying The Muse: {str(e2)}")
            
            # Option 3: Try The Muse API (free, no API key needed)
            try:
                url = "https://www.themuse.com/api/public/jobs"
                params = {
                    'category': 'Engineering',
                    'level': 'Entry Level,Mid Level,Senior Level',
                    'page': 1
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                job_keywords = job_title.lower().split()
                
                for job in data.get('results', [])[:15]:  # Limit to 15 jobs
                    job_name = job.get('name', '').lower()
                    if any(keyword in job_name for keyword in job_keywords):
                        job_list.append({
                            'title': job.get('name', 'N/A'),
                            'company': job.get('company', {}).get('name', 'N/A'),
                            'description': job.get('contents', 'N/A')[:500] + '...' if len(job.get('contents', '')) > 500 else job.get('contents', 'N/A'),
                            'location': ', '.join([loc.get('name', '') for loc in job.get('locations', [])]) or location
                        })
                        
            except Exception as e3:
                st.warning(f"The Muse API also failed: {str(e3)}")
            st.warning(f"RemoteOK also failed: {str(e2)}")
            
            # Option 3: Final fallback - use sample data
            st.info("Using sample job data as fallback")
            job_list = [
                {
                    'title': f'{job_title} Developer',
                    'company': 'Tech Corp',
                    'description': f'We are looking for a skilled {job_title} developer to join our team. Responsibilities include developing and maintaining applications, collaborating with cross-functional teams, and staying up-to-date with industry trends.',
                    'location': location
                },
                {
                    'title': f'Senior {job_title} Engineer',
                    'company': 'Innovation Labs',
                    'description': f'Senior role for {job_title} development. Must have 5+ years experience in software development, strong problem-solving skills, and ability to mentor junior developers.',
                    'location': location
                },
                {
                    'title': f'{job_title} Specialist',
                    'company': 'StartupXYZ',
                    'description': f'Join our growing team as a {job_title} specialist. Work on cutting-edge projects, flexible work environment, and opportunities for professional growth.',
                    'location': location
                }
            ]
    
    return pd.DataFrame(job_list)

def get_jobs_data(job_title, location, use_cache=True):
    cache_file = f"cache_{job_title.replace(' ', '_')}.csv"
    if use_cache and os.path.exists(cache_file):
        st.info(f"Loading cached job data for '{job_title}'...")
        return pd.read_csv(cache_file)
    else:
        with st.spinner(f"Performing live scrape for '{job_title}' jobs in {location}..."):
            jobs_df = scrape_live_jobs(job_title, location)
            if not jobs_df.empty:
                jobs_df.to_csv(cache_file, index=False)
        return jobs_df

# === AI CORE & NEW ADVANCED FUNCTIONS ===
@st.cache_data
def get_embedding(text):
    if not text or not isinstance(text, str): return None
    try:
        model = 'models/text-embedding-004'
        return genai.embed_content(model=model, content=text)['embedding']
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def extract_skills_from_resume(resume_text):
    prompt = f"""
    Carefully analyze the following resume text and extract ALL technical skills mentioned. 
    Look for:
    - Programming languages (Python, Java, C++, JavaScript, etc.)
    - Frameworks and libraries (React, Django, Spring, etc.)
    - Tools and technologies (Docker, Kubernetes, Git, etc.)
    - Databases (MySQL, MongoDB, PostgreSQL, etc.)
    - Cloud platforms (AWS, Azure, GCP, etc.)
    - Other technical skills and certifications
    
    For each skill, also determine the proficiency level based on:
    - How frequently it's mentioned
    - Depth of projects described
    - Years of experience mentioned
    - Complexity of work done
    
    Return ONLY a JSON object in this exact format:
    {{
        "skills": [
            {{"name": "Python", "proficiency": "Advanced", "evidence": "3+ years, multiple projects"}},
            {{"name": "React", "proficiency": "Intermediate", "evidence": "Used in 2 major projects"}},
            {{"name": "Docker", "proficiency": "Beginner", "evidence": "Basic containerization experience"}}
        ]
    }}
    
    Resume Text:
    ---
    {resume_text}
    ---
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            skills_data = json.loads(json_match.group())
            return skills_data
        else:
            # Fallback: extract skills as simple string
            fallback_prompt = f"Extract technical skills from this resume as a comma-separated list: {resume_text[:1000]}"
            fallback_response = model.generate_content(fallback_prompt)
            return {"skills": [{"name": skill.strip(), "proficiency": "Unknown", "evidence": "Not assessed"} 
                             for skill in fallback_response.text.split(',') if skill.strip()]}
    except Exception as e:
        st.error(f"Error extracting skills: {e}")
        return {"skills": [{"name": "Python", "proficiency": "Intermediate", "evidence": "Default skill"}]}

def judge_proficiency_level(resume_text, skills_data):
    # Create a summary of skills with their proficiency levels
    if isinstance(skills_data, dict) and 'skills' in skills_data:
        skills_summary = "\n".join([
            f"- {skill['name']}: {skill['proficiency']} ({skill['evidence']})" 
            for skill in skills_data['skills']
        ])
    else:
        # Fallback for old format
        skills_summary = str(skills_data)
    
    prompt = f"""
    Act as a senior technical recruiter. Based on the detailed skills analysis and resume content, provide an overall career proficiency assessment.
    
    Skills Analysis:
    {skills_summary}
    
    Resume Content:
    ---
    {resume_text[:1500]}...
    ---
    
    Provide a comprehensive assessment including:
    1. **Overall Proficiency Level:** (Beginner/Intermediate/Advanced/Expert)
    2. **Strengths:** Top 3 strongest skill areas
    3. **Growth Areas:** Top 3 areas for improvement
    4. **Career Stage:** Assessment of current career level
    5. **Recommendations:** 2-3 specific next steps
    
    Format in clean Markdown with clear sections.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def generate_personalized_roadmap(proficiency_level, resume_summary, job_description):
    prompt = f"Act as an expert career coach. A student at the {proficiency_level} level with profile '{resume_summary}' wants a job with description '{job_description}'. Create a hyper-personalized roadmap to bridge the gap from their current level to the job's requirements. Format in clean Markdown."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def generate_project_simulation(skill_gap):
    """Generates a simulated project brief and code review."""
    prompt = f"To help a student learn '{skill_gap}', create a simulated mini-project. Provide the following in Markdown:\n\n1. **Project Brief:** A clear, one-paragraph task description.\n2. **Starter Code:** A small, simple code snippet in a relevant language to get them started.\n3. **AI Code Review:** A simulated code review of a hypothetical 'good solution', explaining why it's well-structured and effective."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def generate_career_graph(job_title, skills_string):
    """Generates a career path in Graphviz DOT language."""
    prompt = f"Create a 3-step career trajectory graph starting from a '{job_title}' role, based on these skills: {skills_string}. For each step, suggest a future role and one key skill to learn. Output ONLY the code for a Graphviz DOT graph, with nodes styled for clarity (e.g., shape=box, style=rounded). Example: `digraph G {{ rankdir=LR; \"Step 1\" -> \"Step 2\"; }}`"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace("```dot", "").replace("```", "")
    return cleaned_response

def analyze_market_trends(jobs_df, skills_list):
    trends = {}
    job_descriptions = " ".join(jobs_df['description'].str.lower())
    for skill in skills_list:
        if skill:
            count = job_descriptions.count(skill.lower())
            trends[skill] = count
    
    total_jobs = len(jobs_df)
    if not total_jobs: return None, None
    
    trends_percent = {skill: (count / total_jobs) * 100 for skill, count in trends.items()}
    top_3_skills = sorted(trends_percent.items(), key=lambda item: item[1], reverse=True)[:3]
    return trends_percent, top_3_skills

# === MATCHING ENGINE ===
def find_best_match(resume_text, jobs_df):
    resume_embedding = get_embedding(resume_text)
    if resume_embedding is None: return None
    
    with st.spinner("Analyzing job descriptions..."):
        jobs_df['embedding'] = jobs_df['description'].apply(get_embedding)
        jobs_df.dropna(subset=['embedding'], inplace=True)

    if jobs_df.empty: return None

    job_embeddings = np.array(jobs_df['embedding'].tolist())
    similarities = cosine_similarity([resume_embedding], job_embeddings)
    best_match_index = similarities.argmax()
    return jobs_df.iloc[best_match_index]

# === *** NEW FEATURE *** RESUME TAILORING ===
def tailor_resume_section(original_resume_text, job_description):
    """Rewrites a section of the resume to align with a job description."""
    prompt = f"""
    You are a highly skilled resume optimization AI. Rewrite the key experience and project bullet points from the following resume to perfectly align with the target job description. Focus on:
    -   Using strong action verbs.
    -   Incorporating relevant keywords from the job description.
    -   Quantifying achievements where possible (invent reasonable numbers if necessary to make it impactful).
    -   Keep it concise and impactful.

    **Original Resume Text:**
    ---
    {original_resume_text}
    ---

    **Target Job Description:**
    ---
    {job_description}
    ---

    Provide the rewritten, optimized resume content.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# === *** NEW FEATURE *** MOCK INTERVIEW SIMULATOR ===
# Initialize chat history if not already present
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "interviewer_model" not in st.session_state:
    st.session_state.interviewer_model = None

def start_mock_interview(job_title, company_name):
    """Initializes the Gemini model for a mock interview."""
    st.session_state.interview_history = [] # Clear previous chat
    interview_prompt = f"""
    You are 'Alex', a professional and friendly hiring manager at {company_name} interviewing a candidate for the {job_title} role.
    Your task is to conduct a mock interview.

    Your process for each turn is:
    1.  Ask ONE interview question relevant to the {job_title} role.
    2.  Wait for the user's answer.
    3.  Provide 1-2 sentences of concise, constructive feedback on their answer.
    4.  Ask the next logical follow-up question.

    Start the conversation now by introducing yourself and asking your first question.
    """
    st.session_state.interviewer_model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Send the initial prompt to get the first question
    response = st.session_state.interviewer_model.generate_content(interview_prompt)
    st.session_state.interview_history.append({"role": "interviewer", "content": response.text})

def continue_mock_interview(user_answer):
    """Sends user's answer to the AI and gets feedback + next question."""
    if st.session_state.interviewer_model:
        st.session_state.interview_history.append({"role": "user", "content": user_answer})
        
        # Build prompt for the AI based on history to maintain context
        conversation_context = "\n".join([
            f"{'Interviewer' if msg['role'] == 'interviewer' else 'Candidate'}: {msg['content']}"
            for msg in st.session_state.interview_history
        ])
        
        follow_up_prompt = f"""
        Continue the mock interview. Here's the conversation so far:
        ---
        {conversation_context}
        ---
        The candidate just responded. Provide feedback on their last answer (1-2 sentences), then ask your next interview question.
        """
        
        response = st.session_state.interviewer_model.generate_content(follow_up_prompt)
        st.session_state.interview_history.append({"role": "interviewer", "content": response.text})


# === *** NEW FEATURE *** COMPREHENSIVE ANALYTICS DASHBOARD ===
def generate_skill_gap_analysis(resume_skills, job_skills):
    """Analyze skill gaps and provide detailed insights"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Analyze the skill gap between candidate skills and job requirements:
    
    Candidate Skills: {resume_skills}
    Job Requirements: {job_skills}
    
    Provide a JSON-structured analysis with:
    1. "matching_skills": List of skills the candidate already has
    2. "missing_critical": Critical skills the candidate lacks
    3. "missing_nice_to_have": Nice-to-have skills that are missing
    4. "skill_strength_score": Overall score from 1-100
    5. "priority_learning_path": Top 3 skills to learn first
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        import json
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not parse analysis"}
    except:
        return {"error": "Analysis failed"}

def predict_salary_range(job_title, location, skills, experience_level):
    """Generate salary predictions based on market data"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Provide a realistic salary prediction for:
    Role: {job_title}
    Location: {location}
    Skills: {skills}
    Experience: {experience_level}
    
    Return a JSON with:
    1. "min_salary": Minimum expected salary
    2. "max_salary": Maximum expected salary
    3. "median_salary": Most likely salary
    4. "currency": Currency (INR/USD based on location)
    5. "factors": List of 3 factors affecting the salary
    6. "growth_potential": 3-year salary growth prediction
    
    Base predictions on current market standards for 2025.
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not generate salary prediction"}
    except:
        return {"error": "Salary prediction failed"}

def generate_cover_letter(resume_text, job_description, company_name):
    """Generate a personalized cover letter"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Write a compelling, personalized cover letter based on:
    
    Resume: {resume_text[:1000]}...
    Job Description: {job_description[:800]}...
    Company: {company_name}
    
    The cover letter should be:
    - Professional but engaging
    - Specific to the role and company
    - Highlight relevant experience from the resume
    - 3-4 paragraphs maximum
    - Include a strong opening and closing
    
    Format in clean, readable text (no special formatting).
    """
    
    response = model.generate_content(prompt)
    return response.text

def create_skills_assessment_quiz(skills_list):
    """Generate an interactive skills assessment quiz"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Create a skills assessment quiz for these skills: {skills_list}
    
    Generate 5 multiple-choice questions that test practical knowledge.
    Return as JSON with this structure:
    {{
        "questions": [
            {{
                "question": "Question text",
                "options": ["A", "B", "C", "D"],
                "correct_answer": 0,
                "explanation": "Why this is correct",
                "skill_tested": "specific skill"
            }}
        ]
    }}
    
    Make questions practical and job-relevant.
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"questions": []}
    except:
        return {"questions": []}

def analyze_industry_trends(job_title, skills_list):
    """Analyze current industry trends and future outlook"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Analyze industry trends for {job_title} role with skills: {skills_list}
    
    Provide analysis as JSON:
    {{
        "trending_skills": ["list of 5 trending skills"],
        "declining_skills": ["list of 3 declining skills"],
        "job_market_outlook": "growth/stable/declining with explanation",
        "emerging_technologies": ["list of 3 emerging techs"],
        "recommended_certifications": ["list of 3 valuable certifications"],
        "market_insights": ["3 key insights about the field"]
    }}
    
    Base on 2025 market trends. Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not analyze trends"}
    except:
        return {"error": "Trend analysis failed"}

def track_user_progress():
    """Initialize and track user progress with achievements"""
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {
            'analyses_completed': 0,
            'skills_assessed': 0,
            'cover_letters_generated': 0,
            'interviews_completed': 0,
            'achievements': [],
            'total_score': 0,
            'level': 'Beginner',
            'badges': []
        }
    
    return st.session_state.user_progress

def award_achievement(achievement_type, title, description):
    """Award achievements to users"""
    progress = track_user_progress()
    achievement = {
        'type': achievement_type,
        'title': title,
        'description': description,
        'date': pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    
    if achievement not in progress['achievements']:
        progress['achievements'].append(achievement)
        st.success(f"🏆 Achievement Unlocked: {title}")
        st.balloons()

def get_learning_resources(skills_list, proficiency_level):
    """Generate curated learning resources based on skills and level"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Recommend learning resources for someone at {proficiency_level} level wanting to improve these skills: {skills_list}
    
    Provide recommendations as JSON:
    {{
        "courses": [
            {{"name": "Course Name", "provider": "Platform", "type": "free/paid", "url": "example.com", "rating": "4.5/5"}}
        ],
        "books": [
            {{"title": "Book Title", "author": "Author Name", "type": "free/paid", "description": "Brief description"}}
        ],
        "projects": [
            {{"title": "Project Name", "difficulty": "beginner/intermediate/advanced", "description": "What you'll build"}}
        ],
        "certifications": [
            {{"name": "Cert Name", "provider": "Organization", "cost": "Free/Paid", "value": "Industry value"}}
        ]
    }}
    
    Focus on 2025-relevant, high-quality resources. Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not generate resources"}
    except:
        return {"error": "Resource generation failed"}

def analyze_professional_network(resume_text, job_title):
    """Analyze networking opportunities and provide recommendations"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Based on this resume and target role {job_title}, provide networking recommendations:
    
    Resume: {resume_text[:800]}...
    
    Return as JSON:
    {{
        "target_professionals": ["List of 3 types of professionals to connect with"],
        "networking_events": ["List of 3 relevant event types or conferences"],
        "online_communities": ["List of 3 professional communities/forums"],
        "industry_leaders": ["3 types of industry leaders to follow"],
        "networking_strategy": ["3 practical networking tips"],
        "linkedin_optimization": ["3 LinkedIn profile improvement suggestions"]
    }}
    
    Focus on actionable, specific recommendations. Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not analyze network"}
    except:
        return {"error": "Network analysis failed"}

# === *** LEGENDARY FEATURES *** AI CAREER MENTOR CHATBOT ===
def initialize_career_mentor():
    """Initialize the AI Career Mentor with context"""
    if 'mentor_history' not in st.session_state:
        st.session_state.mentor_history = []
    if 'mentor_model' not in st.session_state:
        st.session_state.mentor_model = genai.GenerativeModel('gemini-1.5-flash')

def get_mentor_response(user_message, user_context=None):
    """Get response from AI Career Mentor"""
    context = user_context or {}
    
    system_prompt = f"""
    You are Alex, an expert AI Career Mentor with 15+ years of experience in tech recruitment and career development.
    
    User Context:
    - Skills: {context.get('skills', 'Not provided')}
    - Experience Level: {context.get('level', 'Not provided')}
    - Target Role: {context.get('target_role', 'Not provided')}
    - Current Challenge: {user_message}
    
    Provide personalized, actionable advice that is:
    - Specific and practical
    - Encouraging but realistic
    - Industry-focused
    - Includes next steps
    
    Keep responses conversational but professional (2-3 paragraphs max).
    """
    
    try:
        full_prompt = f"{system_prompt}\n\nUser: {user_message}"
        response = st.session_state.mentor_model.generate_content(full_prompt)
        return response.text
    except:
        return "I'm having trouble connecting right now. Please try again in a moment."

# === *** LEGENDARY FEATURES *** ATS RESUME OPTIMIZER ===
def analyze_ats_compatibility(resume_text, job_description):
    """Analyze resume compatibility with ATS systems"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Analyze this resume for ATS (Applicant Tracking System) compatibility against the job description.
    
    Resume:
    {resume_text[:1500]}...
    
    Job Description:
    {job_description[:1000]}...
    
    Provide analysis as JSON:
    {{
        "ats_score": 85,
        "keyword_match_percentage": 75,
        "missing_keywords": ["keyword1", "keyword2"],
        "format_issues": ["issue1", "issue2"],
        "optimization_suggestions": ["suggestion1", "suggestion2"],
        "sections_to_improve": ["section1", "section2"],
        "strengths": ["strength1", "strength2"],
        "overall_assessment": "Good/Fair/Poor with explanation"
    }}
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not analyze ATS compatibility"}
    except:
        return {"error": "ATS analysis failed"}

# === *** LEGENDARY FEATURES *** MARKET DEMAND PREDICTOR ===
def predict_market_demand(skills_list, location, timeframe="6 months"):
    """Predict market demand for skills in the next timeframe"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Predict market demand for these skills in {location} over the next {timeframe}:
    Skills: {skills_list}
    
    Analyze based on:
    - Current industry trends
    - Emerging technologies
    - Economic factors
    - Remote work impact
    
    Return as JSON:
    {{
        "predictions": [
            {{
                "skill": "Python",
                "demand_trend": "increasing/stable/decreasing",
                "demand_score": 85,
                "growth_rate": "+15%",
                "reasons": ["reason1", "reason2"],
                "recommended_action": "action"
            }}
        ],
        "market_outlook": "positive/neutral/negative",
        "top_emerging_skills": ["skill1", "skill2"],
        "skills_to_avoid": ["skill1", "skill2"],
        "investment_recommendation": "high/medium/low priority learning areas"
    }}
    
    Base on 2025 market trends. Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not predict market demand"}
    except:
        return {"error": "Market prediction failed"}

# === *** LEGENDARY FEATURES *** NEGOTIATION COACH ===
def generate_negotiation_strategy(job_offer_details, user_profile):
    """Generate personalized negotiation strategy"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Create a negotiation strategy for this job offer:
    
    Offer Details: {job_offer_details}
    User Profile: {user_profile}
    
    Provide strategy as JSON:
    {{
        "salary_negotiation": {{
            "suggested_counter": "amount",
            "justification_points": ["point1", "point2"],
            "negotiation_scripts": ["script1", "script2"]
        }},
        "benefits_to_negotiate": ["benefit1", "benefit2"],
        "timing_strategy": "when and how to negotiate",
        "fallback_options": ["option1", "option2"],
        "red_flags": ["flag1", "flag2"],
        "success_probability": "high/medium/low",
        "negotiation_timeline": "suggested timeline"
    }}
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not generate negotiation strategy"}
    except:
        return {"error": "Negotiation strategy generation failed"}

# === *** LEGENDARY FEATURES *** LINKEDIN OPTIMIZER ===
def optimize_linkedin_profile(resume_text, target_role):
    """Generate LinkedIn profile optimization recommendations"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Optimize LinkedIn profile based on resume and target role:
    
    Resume: {resume_text[:1200]}...
    Target Role: {target_role}
    
    Provide optimization as JSON:
    {{
        "headline_suggestions": ["headline1", "headline2"],
        "summary_rewrite": "optimized summary text",
        "keywords_to_add": ["keyword1", "keyword2"],
        "skills_section_optimization": ["skill1", "skill2"],
        "experience_improvements": ["improvement1", "improvement2"],
        "connection_strategy": ["strategy1", "strategy2"],
        "content_posting_ideas": ["idea1", "idea2"],
        "profile_completeness_score": 85,
        "networking_targets": ["target1", "target2"]
    }}
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not optimize LinkedIn profile"}
    except:
        return {"error": "LinkedIn optimization failed"}

# === *** LEGENDARY FEATURES *** CAREER SIMULATION GAME ===
def generate_career_scenario(current_level, industry):
    """Generate career decision scenarios for gamification"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Create a career decision scenario for a {current_level} professional in {industry}:
    
    Generate a realistic career challenge with:
    - Clear situation description
    - 3-4 decision options
    - Potential outcomes for each choice
    - Skills/experience gained
    - Career impact (positive/negative)
    
    Return as JSON:
    {{
        "scenario_title": "title",
        "situation": "detailed scenario description",
        "decisions": [
            {{
                "option": "decision text",
                "short_term_outcome": "immediate result",
                "long_term_outcome": "career impact",
                "skills_gained": ["skill1", "skill2"],
                "risk_level": "low/medium/high",
                "success_probability": 75
            }}
        ],
        "context": "industry/role context",
        "learning_objective": "what this teaches"
    }}
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not generate career scenario"}
    except:
        return {"error": "Career scenario generation failed"}

# === *** LEGENDARY FEATURES *** INDUSTRY INSIGHTS ===
def get_industry_insider_insights(industry, role):
    """Get insider insights for specific industry and role"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Provide insider insights for {role} in {industry}:
    
    Include:
    - Day-in-the-life reality
    - Unspoken success factors
    - Career progression patterns
    - Industry politics and culture
    - Networking strategies
    - Common pitfalls to avoid
    
    Return as JSON:
    {{
        "daily_reality": "what the job actually involves",
        "success_secrets": ["secret1", "secret2"],
        "progression_paths": ["path1", "path2"],
        "culture_insights": ["insight1", "insight2"],
        "networking_tips": ["tip1", "tip2"],
        "common_mistakes": ["mistake1", "mistake2"],
        "insider_language": ["term1: definition", "term2: definition"],
        "key_relationships": ["relationship1", "relationship2"]
    }}
    
    Format as valid JSON only.
    """
    
    try:
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not get industry insights"}
    except:
        return {"error": "Industry insights failed"}


# ----------------------------------------------------------------------------
# 4. STREAMLIT "MISSION CONTROL" DASHBOARD
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Career Agent AI", layout="wide")
st.title("🚀 AI Career Agent: Your Dynamic Mission Control")
st.markdown("**From resume to readiness.** Your comprehensive AI-powered career advancement platform.")

# === NAVIGATION TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "🎯 Career Analysis", 
    "💰 Salary Insights", 
    "📝 Cover Letter AI", 
    "🧠 Skills Assessment", 
    "📊 Market Analytics", 
    "🎤 Mock Interview",
    "📈 Progress Tracking",
    "📚 Learning Hub",
    "🤖 AI Career Mentor",
    "🔍 ATS Optimizer",
    "🎮 Career Simulator",
    "🌟 Insider Insights"
])

# Job Source Information
with st.expander("📊 Job Data Sources", expanded=False):
    st.markdown("""
    **Job listings are sourced from:**
    - 🎯 **Adzuna API** (primary) - Add free API keys in sidebar for best results
    - 🌐 **RemoteOK** (fallback #1) - For remote job opportunities  
    - 🏢 **The Muse API** (fallback #2) - Professional job listings, no API key needed
    - 📝 **Sample Data** (last resort) - Realistic job examples for testing
    
    💡 **Tip:** Get free Adzuna API keys at [developer.adzuna.com](https://developer.adzuna.com/) for live job data!
    """)

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("👨‍🚀 Mission Parameters")
    job_title_input = st.text_input("Enter Target Role", "DevOps Engineer")
    location_input = st.text_input("Enter Location", "Bengaluru")
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    # Optional API Configuration
    with st.expander("🔑 API Configuration (Optional)", expanded=False):
        st.markdown("**Adzuna API** (for live job data):")
        adzuna_id = st.text_input("Adzuna App ID", value=ADZUNA_APP_ID, help="Get free at developer.adzuna.com")
        adzuna_key = st.text_input("Adzuna App Key", value=ADZUNA_APP_KEY, type="password", help="Get free at developer.adzuna.com")
        
        # Update global variables if provided
        if adzuna_id:
            globals()['ADZUNA_APP_ID'] = adzuna_id
        if adzuna_key:
            globals()['ADZUNA_APP_KEY'] = adzuna_key
    
    submit_button = st.button("🚀 Launch Full Simulation")

# === TAB 1: CAREER ANALYSIS (Main existing functionality) ===
with tab1:
    st.header("🎯 Comprehensive Career Analysis")
    
    # --- Main Dashboard ---
    if submit_button and uploaded_file is not None:
        # 1. Initial Processing
        with st.spinner("Initializing AI Agent..."):
            pdf_reader = PdfReader(uploaded_file)
            resume_text = "".join(page.extract_text() for page in pdf_reader.pages)
            skills_data = extract_skills_from_resume(resume_text)
            
            # Extract skills list for compatibility with existing code
            skills_list = [skill['name'] for skill in skills_data.get('skills', [])]
            extracted_skills = ', '.join(skills_list)  # For backward compatibility
            
            # Track progress
            progress = track_user_progress()
            progress['analyses_completed'] += 1
            progress['total_score'] += 50
            award_achievement("analysis", "Getting Started", "Completed your first career analysis")

        # 2. Get Job Data & Find Match
        jobs_df = get_jobs_data(job_title_input, location_input)
        if not jobs_df.empty:
            best_match = find_best_match(resume_text, jobs_df)
            
            # 3. Perform Advanced Analysis
            with st.spinner("Analyzing your profile and market trends..."):
                proficiency_judgment = judge_proficiency_level(resume_text, skills_data)
                trends, top_3_skills = analyze_market_trends(jobs_df, skills_list)
                
                # Store these in session state for other tabs
                st.session_state['resume_text'] = resume_text
                st.session_state['extracted_skills'] = extracted_skills
                st.session_state['skills_list'] = skills_list
                st.session_state['skills_data'] = skills_data  # Store detailed skills with proficiency
                st.session_state['best_match'] = best_match
                st.session_state['jobs_df'] = jobs_df
                st.session_state['proficiency_judgment'] = proficiency_judgment
                
                # Extract proficiency level for use in other prompts
                try:
                    proficiency_level = proficiency_judgment.split("Judged Proficiency Level:**")[1].split("\n")[0].strip()
                    st.session_state['proficiency_level'] = proficiency_level
                except IndexError:
                    proficiency_level = "Intermediate" # Default fallback
                    st.session_state['proficiency_level'] = proficiency_level
                
            # 4. Generate AI Content
            with st.spinner("Generating personalized career intelligence..."):
                roadmap = generate_personalized_roadmap(proficiency_level, resume_text, best_match['description'])
                top_skill_gap = top_3_skills[0][0] if top_3_skills else "Kubernetes" # Default for DevOps
                project_simulation = generate_project_simulation(top_skill_gap)
                career_graph_dot = generate_career_graph(job_title_input, extracted_skills)
                tailored_resume_content = tailor_resume_section(resume_text, best_match['description'])
            
            # 5. RENDER THE DASHBOARD
            st.header(f"Mission Briefing: Your Path to a {job_title_input} Role")
            
            # --- Top Row: Key Metrics ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Your Judged Proficiency", value=proficiency_level)
            with col2:
                st.metric(label="Jobs Analyzed", value=f"{len(jobs_df)}")
            with col3:
                st.metric(label="Top Trending Skill", value=f"{top_3_skills[0][0]}", help=f"Found in {top_3_skills[0][1]:.0f}% of local job postings.")
            
        st.divider()

        # --- Main Content Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview & Trends", 
            "🛣️ Roadmap & Projects", 
            "📝 Resume Tailoring", 
            "🗣️ Mock Interview",
            "📈 Career Trajectory"
        ])
        
        with tab1: # Overview & Trends
            st.subheader("🏆 Best Matched Job")
            with st.container(border=True):
                 st.markdown(f"**{best_match['title']}** at **{best_match['company']}**")
                 st.caption(best_match['description'])
            
            # Display extracted skills with proficiency levels
            st.subheader("🎯 Your Technical Skills Profile")
            if 'skills_data' in st.session_state and st.session_state['skills_data'].get('skills'):
                skills_df = pd.DataFrame(st.session_state['skills_data']['skills'])
                
                # Show proficiency distribution
                proficiency_counts = skills_df['proficiency'].value_counts()
                col_prof1, col_prof2, col_prof3 = st.columns(3)
                
                with col_prof1:
                    st.metric("🟢 Advanced Skills", proficiency_counts.get('Advanced', 0))
                with col_prof2:
                    st.metric("🟡 Intermediate Skills", proficiency_counts.get('Intermediate', 0))
                with col_prof3:
                    st.metric("🔴 Beginner Skills", proficiency_counts.get('Beginner', 0))
                
                # Create columns for better display
                st.markdown("**📋 Detailed Skills Breakdown:**")
                col1, col2 = st.columns(2)
                
                for i, skill in enumerate(skills_df.to_dict('records')):
                    with col1 if i % 2 == 0 else col2:
                        proficiency_colors = {
                            'Advanced': '🟢',
                            'Intermediate': '🟡', 
                            'Beginner': '🔴',
                            'Unknown': '⚪'
                        }
                        color = proficiency_colors.get(skill['proficiency'], '⚪')
                        
                        with st.container(border=True):
                            st.markdown(f"**{color} {skill['name']}** - {skill['proficiency']}")
                            st.caption(f"📝 {skill['evidence']}")
            else:
                st.warning("No skills data available. Please run the analysis again.")
                 
            st.subheader("📊 Live Job Market Trends")
            st.write(f"Analysis of the top skills required for **{job_title_input}** roles in **{location_input}** right now.")
            if trends:
                # Show top 5 skills as bars
                sorted_trends = sorted(trends.items(), key=lambda item: item[1], reverse=True)[:5]
                for skill, percent in sorted_trends:
                    st.write(f"**{skill}**")
                    # Normalize percent to 0-1 range for progress bar
                    normalized_percent = min(percent / 100.0, 1.0)
                    st.progress(normalized_percent)
            st.subheader("AI-Powered Proficiency Judgment")
            st.markdown(proficiency_judgment)


        with tab2: # Roadmap & Projects
            st.subheader(f"🛠️ Your Personalized Roadmap from a {proficiency_level} Level")
            st.markdown(roadmap)
            
            with st.expander(f"Expand for a Hands-On Project Simulation for '{top_skill_gap}'"):
                st.subheader(f"Mini-Project: Getting Started with {top_skill_gap}")
                st.markdown(project_simulation)

        with tab3: # Resume Tailoring
            st.subheader("📝 AI-Powered Resume Tailoring")
            st.info("Your AI Agent will now optimize your resume content for the best-matched job.")
            tailor_col1, tailor_col2 = st.columns(2)
            with tailor_col1:
                st.subheader("Your Original Resume (Snippet)")
                st.code(resume_text[:1000] + "...", language="text") # Show first 1000 chars
            with tailor_col2:
                st.subheader("AI-Optimized Content for This Job")
                st.markdown(tailored_resume_content)

        with tab4: # Mock Interview
            st.subheader("🗣️ AI Mock Interview Simulator")
            st.info(f"Start a mock interview for the **{best_match['title']}** role at **{best_match['company']}**.")
            
            if st.button("Start New Mock Interview"):
                start_mock_interview(best_match['title'], best_match['company'])

            # Display chat history
            for message in st.session_state.interview_history:
                if message["role"] == "interviewer":
                    st.markdown(f"**Interviewer:** {message['content']}")
                else:
                    st.text_area("Your Answer:", value=message["content"], height=50, disabled=True, key=f"user_msg_{hash(message['content'])}")
            
            # User input for new messages
            if st.session_state.interviewer_model:
                user_input = st.text_input("Your response:", key="user_interview_response")
                if user_input:
                    continue_mock_interview(user_input)
                    st.rerun() # Rerun to display the new messages


        with tab5: # Career Trajectory
            st.subheader("📈 Your Future: 5-Year Career Trajectory Map")
            st.info("Visualize potential growth paths based on your current skills and target role.")
            if "digraph" in career_graph_dot:
                st.graphviz_chart(career_graph_dot)
            else:
                st.warning("Could not generate a career graph. The AI response was invalid.")
            
    else:
        st.error("Could not retrieve job data or find a suitable match. Please try a different job title.")

    if submit_button:
        st.warning("Please upload your resume to launch the simulation.")
    else:
        st.info("👆 Upload your resume and click 'Launch Full Simulation' to get started!")

# === TAB 2: SALARY INSIGHTS ===
with tab2:
    st.header("💰 AI-Powered Salary Intelligence")
    
    if 'proficiency_level' in st.session_state:
        with st.spinner("Analyzing market salary data..."):
            salary_data = predict_salary_range(
                job_title_input, 
                location_input, 
                st.session_state.get('extracted_skills', ''), 
                st.session_state.get('proficiency_level', 'Intermediate')
            )
        
        if 'error' not in salary_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💸 Minimum Salary", f"{salary_data.get('min_salary', 'N/A')} {salary_data.get('currency', '')}")
            with col2:
                st.metric("🎯 Expected Salary", f"{salary_data.get('median_salary', 'N/A')} {salary_data.get('currency', '')}")
            with col3:
                st.metric("🚀 Maximum Salary", f"{salary_data.get('max_salary', 'N/A')} {salary_data.get('currency', '')}")
            
            st.subheader("📊 Salary Factors Analysis")
            for factor in salary_data.get('factors', []):
                st.write(f"• {factor}")
            
            st.subheader("📈 3-Year Growth Projection")
            st.info(salary_data.get('growth_potential', 'Growth data unavailable'))
        else:
            st.error("Unable to generate salary insights. Please run the analysis first.")
    else:
        st.info("Complete the Career Analysis first to see salary insights!")

# === TAB 3: AI COVER LETTER GENERATOR ===
with tab3:
    st.header("📝 Intelligent Cover Letter Generator")
    
    if 'resume_text' in st.session_state and 'best_match' in st.session_state:
        company_name = st.text_input("Enter Company Name", value=st.session_state['best_match'].get('company', 'Target Company'))
        
        if st.button("🤖 Generate Cover Letter"):
            with st.spinner("Crafting your personalized cover letter..."):
                cover_letter = generate_cover_letter(
                    st.session_state['resume_text'],
                    st.session_state['best_match']['description'],
                    company_name
                )
                
                # Track progress
                progress = track_user_progress()
                progress['cover_letters_generated'] += 1
                progress['total_score'] += 30
                if progress['cover_letters_generated'] >= 3:
                    award_achievement("communication", "Cover Letter Pro", "Generated 3 cover letters")
            
            st.subheader("📄 Your AI-Generated Cover Letter")
            st.text_area("Cover Letter", value=cover_letter, height=400)
            
            # Download option
            st.download_button(
                label="📥 Download Cover Letter",
                data=cover_letter,
                file_name=f"cover_letter_{company_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    else:
        st.info("Complete the Career Analysis first to generate cover letters!")

# === TAB 4: SKILLS ASSESSMENT QUIZ ===
with tab4:
    st.header("🧠 Interactive Skills Assessment")
    
    if 'skills_list' in st.session_state:
        if st.button("🎯 Generate Skills Quiz"):
            with st.spinner("Creating personalized assessment..."):
                quiz_data = create_skills_assessment_quiz(st.session_state['skills_list'][:5])
            
            if quiz_data.get('questions'):
                st.session_state['quiz_data'] = quiz_data
                st.session_state['quiz_answers'] = {}
                st.session_state['quiz_submitted'] = False
        
        if 'quiz_data' in st.session_state and not st.session_state.get('quiz_submitted', False):
            st.subheader("📝 Skills Assessment Quiz")
            
            for i, question in enumerate(st.session_state['quiz_data']['questions']):
                st.markdown(f"**Question {i+1}:** {question['question']}")
                answer = st.radio(
                    f"Select your answer for Q{i+1}:",
                    options=question['options'],
                    key=f"q_{i}"
                )
                st.session_state['quiz_answers'][i] = question['options'].index(answer)
                st.markdown("---")
            
            if st.button("📊 Submit Assessment"):
                st.session_state['quiz_submitted'] = True
                
                # Track progress
                progress = track_user_progress()
                progress['skills_assessed'] += 1
                progress['total_score'] += 40
                if progress['skills_assessed'] >= 2:
                    award_achievement("knowledge", "Skills Explorer", "Completed multiple skill assessments")
                
                st.rerun()
        
        if st.session_state.get('quiz_submitted', False):
            st.subheader("🎉 Assessment Results")
            correct_answers = 0
            total_questions = len(st.session_state['quiz_data']['questions'])
            
            for i, question in enumerate(st.session_state['quiz_data']['questions']):
                user_answer = st.session_state['quiz_answers'][i]
                correct_answer = question['correct_answer']
                
                if user_answer == correct_answer:
                    st.success(f"✅ Q{i+1}: Correct!")
                    correct_answers += 1
                else:
                    st.error(f"❌ Q{i+1}: Incorrect")
                    st.info(f"💡 Explanation: {question['explanation']}")
            
            score = (correct_answers / total_questions) * 100
            st.metric("📊 Your Score", f"{score:.1f}%")
            
            if score >= 80:
                st.balloons()
                st.success("🏆 Excellent! You have strong knowledge in these skills!")
            elif score >= 60:
                st.info("👍 Good job! Consider reviewing the areas you missed.")
            else:
                st.warning("📚 Keep learning! Focus on the skills that need improvement.")
    else:
        st.info("Complete the Career Analysis first to take the assessment!")

# === TAB 5: MARKET ANALYTICS ===
with tab5:
    st.header("📊 Industry Trends & Market Analytics")
    
    if 'skills_list' in st.session_state:
        if st.button("🔍 Analyze Industry Trends"):
            with st.spinner("Analyzing market trends and future outlook..."):
                trends_data = analyze_industry_trends(job_title_input, st.session_state['skills_list'])
            
            if 'error' not in trends_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Trending Skills")
                    for skill in trends_data.get('trending_skills', []):
                        st.write(f"🔥 {skill}")
                    
                    st.subheader("📉 Declining Skills")
                    for skill in trends_data.get('declining_skills', []):
                        st.write(f"⬇️ {skill}")
                
                with col2:
                    st.subheader("🚀 Emerging Technologies")
                    for tech in trends_data.get('emerging_technologies', []):
                        st.write(f"⚡ {tech}")
                    
                    st.subheader("🏆 Recommended Certifications")
                    for cert in trends_data.get('recommended_certifications', []):
                        st.write(f"📜 {cert}")
                
                st.subheader("🔮 Market Outlook")
                st.info(trends_data.get('job_market_outlook', 'Market analysis unavailable'))
                
                st.subheader("💡 Key Market Insights")
                for insight in trends_data.get('market_insights', []):
                    st.write(f"• {insight}")
            else:
                st.error("Unable to analyze market trends. Please try again.")
    else:
        st.info("Complete the Career Analysis first to see market analytics!")

# === TAB 6: ENHANCED MOCK INTERVIEW ===
with tab6:
    st.header("🎤 Advanced Mock Interview Simulator")
    
    if 'best_match' in st.session_state:
        st.info(f"🎯 **Interview for:** {st.session_state['best_match']['title']} at {st.session_state['best_match']['company']}")
        
        if st.button("🚀 Start Enhanced Interview"):
            start_mock_interview(st.session_state['best_match']['title'], st.session_state['best_match']['company'])

        # Display chat history
        for message in st.session_state.interview_history:
            if message["role"] == "interviewer":
                st.markdown(f"🤖 **Interviewer:** {message['content']}")
            else:
                st.markdown(f"👤 **You:** {message['content']}")
        
        # User input for new messages
        if st.session_state.interviewer_model:
            user_input = st.text_area("Your response:", height=100, key="enhanced_interview_response")
            if st.button("📤 Send Response") and user_input:
                continue_mock_interview(user_input)
                st.rerun()
        
        # Interview tips
        with st.expander("💡 Interview Tips & Best Practices"):
            st.markdown("""
            **🎯 Quick Tips for Success:**
            - Use the STAR method (Situation, Task, Action, Result) for behavioral questions
            - Research the company and role thoroughly
            - Prepare specific examples from your experience
            - Ask thoughtful questions about the role and team
            - Practice your technical skills relevant to the position
            """)
    else:
        st.info("Complete the Career Analysis first to start the interview simulation!")

# === TAB 7: PROGRESS TRACKING ===
with tab7:
    st.header("📈 Your Career Development Progress")
    
    # Initialize progress tracking
    progress = track_user_progress()
    
    # Progress Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Analyses", progress['analyses_completed'])
    with col2:
        st.metric("🧠 Skills Tests", progress['skills_assessed'])
    with col3:
        st.metric("📝 Cover Letters", progress['cover_letters_generated'])
    with col4:
        st.metric("🎤 Interviews", progress['interviews_completed'])
    
    # User Level and Score
    st.subheader("🏆 Your Career Level")
    level_progress = min(progress['total_score'] / 1000 * 100, 100)  # Max 1000 points
    st.progress(level_progress / 100)
    st.write(f"**Current Level:** {progress['level']} | **Score:** {progress['total_score']}/1000")
    
    # Achievements Section
    st.subheader("🏆 Achievements Unlocked")
    if progress['achievements']:
        for achievement in progress['achievements']:
            st.success(f"🏆 **{achievement['title']}** - {achievement['description']} (Earned: {achievement['date']})")
    else:
        st.info("Complete activities to unlock achievements!")
    
    # Quick Actions to Boost Progress
    st.subheader("🚀 Boost Your Progress")
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        if st.button("📊 Complete Full Analysis"):
            progress['analyses_completed'] += 1
            progress['total_score'] += 100
            award_achievement("analysis", "Career Explorer", "Completed comprehensive career analysis")
    
    with action_col2:
        if st.button("🎯 Set Learning Goal"):
            st.text_input("Enter your learning goal:", key="learning_goal")
            if st.session_state.get('learning_goal'):
                st.success("Goal set! Keep tracking your progress.")
    
    # Progress Tips
    with st.expander("💡 Tips to Accelerate Your Progress"):
        st.markdown("""
        **🎯 Quick Wins:**
        - Complete skills assessments to earn points
        - Generate cover letters for practice
        - Take mock interviews regularly
        - Set and achieve weekly learning goals
        - Engage with all platform features
        
        **🏆 Achievement Categories:**
        - 📚 Knowledge Builder: Complete assessments
        - 🎯 Career Strategist: Use all analysis features  
        - 💼 Interview Master: Excel in mock interviews
        - 📝 Communication Pro: Create quality cover letters
        """)

# === TAB 8: LEARNING RESOURCE HUB ===
with tab8:
    st.header("📚 Personalized Learning Hub")
    
    if 'skills_list' in st.session_state and 'proficiency_level' in st.session_state:
        if st.button("🔍 Generate Learning Resources"):
            with st.spinner("Curating personalized learning resources..."):
                resources = get_learning_resources(
                    st.session_state['skills_list'], 
                    st.session_state['proficiency_level']
                )
            
            if 'error' not in resources:
                # Courses Section
                st.subheader("🎓 Recommended Courses")
                for course in resources.get('courses', []):
                    with st.expander(f"📖 {course['name']} ({course['type']})"):
                        st.write(f"**Provider:** {course['provider']}")
                        st.write(f"**Rating:** {course['rating']}")
                        if course.get('url'):
                            st.link_button("🔗 Visit Course", course['url'])
                
                # Books Section  
                st.subheader("📚 Essential Reading")
                for book in resources.get('books', []):
                    st.write(f"📖 **{book['title']}** by {book['author']} ({book['type']})")
                    st.write(f"   _{book['description']}_")
                
                # Projects Section
                st.subheader("🛠️ Hands-on Projects")
                for project in resources.get('projects', []):
                    difficulty_color = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                    st.write(f"{difficulty_color.get(project['difficulty'], '⚪')} **{project['title']}** ({project['difficulty']})")
                    st.write(f"   {project['description']}")
                
                # Certifications Section
                st.subheader("🏆 Valuable Certifications")
                for cert in resources.get('certifications', []):
                    st.write(f"🏆 **{cert['name']}** - {cert['provider']} ({cert['cost']})")
                    st.write(f"   _{cert['value']}_")
            else:
                st.error("Unable to generate learning resources. Please try again.")
        
        # Professional Networking Section
        st.subheader("🤝 Professional Networking Guide")
        
        if 'resume_text' in st.session_state:
            if st.button("🔍 Analyze Networking Opportunities"):
                with st.spinner("Analyzing networking opportunities..."):
                    network_analysis = analyze_professional_network(
                        st.session_state['resume_text'], 
                        job_title_input
                    )
                
                if 'error' not in network_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👥 Connect With")
                        for professional in network_analysis.get('target_professionals', []):
                            st.write(f"• {professional}")
                        
                        st.subheader("🎪 Events & Conferences")
                        for event in network_analysis.get('networking_events', []):
                            st.write(f"• {event}")
                    
                    with col2:
                        st.subheader("💬 Online Communities")
                        for community in network_analysis.get('online_communities', []):
                            st.write(f"• {community}")
                        
                        st.subheader("🌟 Industry Leaders")
                        for leader in network_analysis.get('industry_leaders', []):
                            st.write(f"• {leader}")
                    
                    st.subheader("🎯 Networking Strategy")
                    for tip in network_analysis.get('networking_strategy', []):
                        st.info(f"💡 {tip}")
                    
                    st.subheader("🔗 LinkedIn Optimization")
                    for suggestion in network_analysis.get('linkedin_optimization', []):
                        st.write(f"✅ {suggestion}")
        
        # Learning Path Tracker
        st.subheader("📊 Learning Path Tracker")
        
        if 'learning_progress' not in st.session_state:
            st.session_state.learning_progress = {}
        
        learning_goal = st.text_input("Set a learning goal:", key="new_learning_goal")
        if st.button("➕ Add Learning Goal") and learning_goal:
            st.session_state.learning_progress[learning_goal] = 0
            st.success("Learning goal added!")
        
        # Display and update learning progress
        for goal, progress_val in st.session_state.learning_progress.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📚 {goal}")
                st.progress(progress_val / 100)
            with col2:
                if st.button("➕", key=f"inc_{goal}"):
                    st.session_state.learning_progress[goal] = min(progress_val + 10, 100)
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_{goal}"):
                    del st.session_state.learning_progress[goal]
                    st.rerun()
    
    else:
        st.info("Complete the Career Analysis first to access personalized learning resources!")

# === TAB 9: AI CAREER MENTOR CHATBOT ===
with tab9:
    st.header("🤖 AI Career Mentor - Your 24/7 Career Coach")
    
    # Initialize mentor
    initialize_career_mentor()
    
    # Get user context for personalized advice
    user_context = {
        'skills': ', '.join(st.session_state.get('skills_list', [])),
        'level': st.session_state.get('proficiency_level', 'Not assessed'),
        'target_role': job_title_input
    }
    
    # Display chat history
    st.subheader("💬 Chat with Alex, Your AI Career Mentor")
    
    # Chat container
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.mentor_history:
            if message['role'] == 'user':
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Alex:** {message['content']}")
    
    # Quick action buttons
    st.subheader("🚀 Quick Career Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💼 Career Change Advice"):
            quick_message = f"I want to transition to a {job_title_input} role. What should I focus on?"
            response = get_mentor_response(quick_message, user_context)
            st.session_state.mentor_history.append({'role': 'user', 'content': quick_message})
            st.session_state.mentor_history.append({'role': 'mentor', 'content': response})
            st.rerun()
    
    with col2:
        if st.button("📈 Skill Development"):
            quick_message = "What skills should I prioritize for career growth?"
            response = get_mentor_response(quick_message, user_context)
            st.session_state.mentor_history.append({'role': 'user', 'content': quick_message})
            st.session_state.mentor_history.append({'role': 'mentor', 'content': response})
            st.rerun()
    
    with col3:
        if st.button("🎯 Interview Prep"):
            quick_message = "How should I prepare for interviews in my target role?"
            response = get_mentor_response(quick_message, user_context)
            st.session_state.mentor_history.append({'role': 'user', 'content': quick_message})
            st.session_state.mentor_history.append({'role': 'mentor', 'content': response})
            st.rerun()
    
    # Custom question input
    user_question = st.text_input("💭 Ask Alex anything about your career:", key="mentor_question")
    if st.button("📤 Send Question") and user_question:
        with st.spinner("Alex is thinking..."):
            response = get_mentor_response(user_question, user_context)
            st.session_state.mentor_history.append({'role': 'user', 'content': user_question})
            st.session_state.mentor_history.append({'role': 'mentor', 'content': response})
        st.rerun()

# === TAB 10: ATS RESUME OPTIMIZER ===
with tab10:
    st.header("🔍 ATS Resume Optimizer & Scoring System")
    
    if 'resume_text' in st.session_state and 'best_match' in st.session_state:
        st.info("🎯 Optimize your resume to pass Applicant Tracking Systems (ATS)")
        
        if st.button("🚀 Analyze ATS Compatibility"):
            with st.spinner("Scanning resume for ATS compatibility..."):
                ats_analysis = analyze_ats_compatibility(
                    st.session_state['resume_text'],
                    st.session_state['best_match']['description']
                )
            
            if 'error' not in ats_analysis:
                # ATS Score Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    score = ats_analysis.get('ats_score', 0)
                    st.metric("🎯 ATS Score", f"{score}/100")
                    if score >= 80:
                        st.success("Excellent ATS compatibility!")
                    elif score >= 60:
                        st.warning("Good, but room for improvement")
                    else:
                        st.error("Needs significant optimization")
                
                with col2:
                    keyword_match = ats_analysis.get('keyword_match_percentage', 0)
                    st.metric("🔑 Keyword Match", f"{keyword_match}%")
                
                with col3:
                    assessment = ats_analysis.get('overall_assessment', 'Not assessed')
                    st.metric("📊 Overall Rating", assessment.split()[0])
                
                # Detailed Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("❌ Issues to Fix")
                    for issue in ats_analysis.get('format_issues', []):
                        st.write(f"• {issue}")
                    
                    st.subheader("🔑 Missing Keywords")
                    for keyword in ats_analysis.get('missing_keywords', []):
                        st.write(f"• {keyword}")
                
                with col2:
                    st.subheader("✅ Current Strengths")
                    for strength in ats_analysis.get('strengths', []):
                        st.write(f"• {strength}")
                    
                    st.subheader("📈 Sections to Improve")
                    for section in ats_analysis.get('sections_to_improve', []):
                        st.write(f"• {section}")
                
                # Optimization Suggestions
                st.subheader("🛠️ Optimization Recommendations")
                for suggestion in ats_analysis.get('optimization_suggestions', []):
                    st.info(f"💡 {suggestion}")
                
                # LinkedIn Profile Optimization
                st.subheader("💼 LinkedIn Profile Optimization")
                if st.button("🔗 Optimize LinkedIn Profile"):
                    with st.spinner("Generating LinkedIn optimization..."):
                        linkedin_opt = optimize_linkedin_profile(
                            st.session_state['resume_text'],
                            job_title_input
                        )
                    
                    if 'error' not in linkedin_opt:
                        st.markdown("**🎯 Suggested Headlines:**")
                        for headline in linkedin_opt.get('headline_suggestions', []):
                            st.code(headline)
                        
                        st.markdown("**📝 Optimized Summary:**")
                        st.text_area("LinkedIn Summary", value=linkedin_opt.get('summary_rewrite', ''), height=200)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔑 Keywords to Add:**")
                            for keyword in linkedin_opt.get('keywords_to_add', []):
                                st.write(f"• {keyword}")
                        
                        with col2:
                            st.markdown("**🎯 Networking Strategy:**")
                            for strategy in linkedin_opt.get('connection_strategy', []):
                                st.write(f"• {strategy}")
            else:
                st.error("Unable to analyze ATS compatibility. Please try again.")
    else:
        st.info("Complete the Career Analysis first to optimize your resume!")

# === TAB 11: CAREER SIMULATION GAME ===
with tab11:
    st.header("🎮 Career Strategy Simulator")
    
    if 'proficiency_level' in st.session_state:
        st.info("🎯 Make strategic career decisions and see their outcomes!")
        
        # Initialize game state
        if 'game_scenarios' not in st.session_state:
            st.session_state.game_scenarios = []
        if 'game_score' not in st.session_state:
            st.session_state.game_score = 0
        if 'scenarios_completed' not in st.session_state:
            st.session_state.scenarios_completed = 0
        
        # Game stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Career Score", st.session_state.game_score)
        with col2:
            st.metric("🏆 Scenarios Completed", st.session_state.scenarios_completed)
        with col3:
            level = "Novice" if st.session_state.game_score < 100 else "Strategist" if st.session_state.game_score < 300 else "Career Master"
            st.metric("🌟 Player Level", level)
        
        # Generate new scenario
        if st.button("🎲 Start New Career Challenge"):
            with st.spinner("Generating career scenario..."):
                scenario = generate_career_scenario(
                    st.session_state.get('proficiency_level', 'Intermediate'),
                    "Technology"  # Can be made dynamic
                )
            
            if 'error' not in scenario:
                st.session_state.current_scenario = scenario
                st.session_state.scenario_completed = False
        
        # Display current scenario
        if 'current_scenario' in st.session_state and not st.session_state.get('scenario_completed', False):
            scenario = st.session_state.current_scenario
            
            st.subheader(f"🎭 {scenario.get('scenario_title', 'Career Challenge')}")
            st.markdown(scenario.get('situation', 'No scenario loaded'))
            
            st.subheader("🤔 What do you do?")
            decision_options = scenario.get('decisions', [])
            
            selected_decision = st.radio(
                "Choose your strategy:",
                options=range(len(decision_options)),
                format_func=lambda x: decision_options[x]['option'] if x < len(decision_options) else "No option"
            )
            
            if st.button("✅ Make Decision"):
                chosen_decision = decision_options[selected_decision]
                
                # Calculate score based on success probability
                points_earned = int(chosen_decision.get('success_probability', 50) / 10)
                st.session_state.game_score += points_earned
                st.session_state.scenarios_completed += 1
                st.session_state.scenario_completed = True
                
                # Show outcome
                st.success(f"🎉 You earned {points_earned} points!")
                st.markdown(f"**📊 Outcome:** {chosen_decision.get('short_term_outcome', 'Unknown')}")
                st.markdown(f"**🔮 Long-term Impact:** {chosen_decision.get('long_term_outcome', 'Unknown')}")
                st.markdown(f"**🎯 Skills Gained:** {', '.join(chosen_decision.get('skills_gained', []))}")
                
                # Update progress tracking
                progress = track_user_progress()
                progress['total_score'] += points_earned
                if st.session_state.scenarios_completed >= 5:
                    award_achievement("strategy", "Career Strategist", "Completed 5 career scenarios")
        
        # Scenario history
        if st.session_state.scenarios_completed > 0:
            with st.expander("📚 Your Career Decision History"):
                st.write(f"You've successfully navigated {st.session_state.scenarios_completed} career challenges!")
                st.write(f"Current career strategy score: {st.session_state.game_score}")
    else:
        st.info("Complete the Career Analysis first to start the career simulator!")

# === TAB 12: INDUSTRY INSIDER INSIGHTS ===
with tab12:
    st.header("🌟 Industry Insider Insights")
    
    st.info("🔍 Get exclusive insider knowledge about your target industry and role")
    
    # Industry selection
    industry_options = ["Technology", "Finance", "Healthcare", "Marketing", "Consulting", "Startups", "E-commerce"]
    selected_industry = st.selectbox("Select Industry:", industry_options)
    
    if st.button("🎯 Get Insider Insights"):
        with st.spinner("Gathering insider intelligence..."):
            insights = get_industry_insider_insights(selected_industry, job_title_input)
        
        if 'error' not in insights:
            # Daily Reality
            st.subheader("📅 Day-in-the-Life Reality")
            st.markdown(insights.get('daily_reality', 'No insights available'))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔐 Success Secrets")
                for secret in insights.get('success_secrets', []):
                    st.success(f"💡 {secret}")
                
                st.subheader("🚀 Career Progression")
                for path in insights.get('progression_paths', []):
                    st.write(f"📈 {path}")
                
                st.subheader("🤝 Key Relationships")
                for relationship in insights.get('key_relationships', []):
                    st.write(f"👥 {relationship}")
            
            with col2:
                st.subheader("🎭 Culture Insights")
                for insight in insights.get('culture_insights', []):
                    st.info(f"🏢 {insight}")
                
                st.subheader("⚠️ Common Mistakes")
                for mistake in insights.get('common_mistakes', []):
                    st.warning(f"❌ {mistake}")
                
                st.subheader("🗣️ Insider Language")
                for term in insights.get('insider_language', []):
                    st.write(f"📖 {term}")
            
            # Networking strategies
            st.subheader("🌐 Networking Strategies")
            for tip in insights.get('networking_tips', []):
                st.write(f"🤝 {tip}")
            
            # Market demand prediction
            st.subheader("📊 Market Demand Forecast")
            if 'skills_list' in st.session_state:
                if st.button("🔮 Predict Skill Demand"):
                    with st.spinner("Analyzing market trends..."):
                        demand_prediction = predict_market_demand(
                            st.session_state['skills_list'][:5],  # Top 5 skills
                            location_input
                        )
                    
                    if 'error' not in demand_prediction:
                        st.markdown(f"**🎯 Market Outlook:** {demand_prediction.get('market_outlook', 'Not available')}")
                        
                        # Skill demand predictions
                        for prediction in demand_prediction.get('predictions', []):
                            trend_emoji = "📈" if prediction['demand_trend'] == "increasing" else "📊" if prediction['demand_trend'] == "stable" else "📉"
                            st.write(f"{trend_emoji} **{prediction['skill']}**: {prediction['demand_trend']} ({prediction.get('growth_rate', 'N/A')})")
                            st.caption(f"Reasons: {', '.join(prediction.get('reasons', []))}")
                        
                        # Investment recommendations
                        st.subheader("💡 Learning Investment Strategy")
                        st.info(demand_prediction.get('investment_recommendation', 'No recommendations available'))
        else:
            st.error("Unable to get industry insights. Please try again.")
    
    # Negotiation coach section
    st.subheader("💰 Salary Negotiation Coach")
    with st.expander("🤝 Get Negotiation Strategy"):
        offer_details = st.text_area("Describe your job offer (salary, benefits, etc.):", height=100)
        
        if st.button("💡 Generate Negotiation Strategy") and offer_details:
            with st.spinner("Crafting negotiation strategy..."):
                user_profile = {
                    'skills': st.session_state.get('extracted_skills', ''),
                    'experience': st.session_state.get('proficiency_level', 'Intermediate'),
                    'target_role': job_title_input
                }
                strategy = generate_negotiation_strategy(offer_details, str(user_profile))
            
            if 'error' not in strategy:
                st.subheader("🎯 Your Negotiation Strategy")
                
                # Salary negotiation
                salary_section = strategy.get('salary_negotiation', {})
                st.markdown(f"**💰 Suggested Counter:** {salary_section.get('suggested_counter', 'Not available')}")
                
                st.markdown("**🎯 Justification Points:**")
                for point in salary_section.get('justification_points', []):
                    st.write(f"• {point}")
                
                st.markdown("**💬 Negotiation Scripts:**")
                for script in salary_section.get('negotiation_scripts', []):
                    st.code(script)
                
                # Other negotiation aspects
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎁 Benefits to Negotiate:**")
                    for benefit in strategy.get('benefits_to_negotiate', []):
                        st.write(f"• {benefit}")
                
                with col2:
                    st.markdown("**🚩 Red Flags to Watch:**")
                    for flag in strategy.get('red_flags', []):
                        st.write(f"⚠️ {flag}")
                
                st.info(f"**⏰ Timing Strategy:** {strategy.get('timing_strategy', 'Not available')}")
                st.info(f"**📊 Success Probability:** {strategy.get('success_probability', 'Not assessed')}")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <h4>🚀 AI Career Agent - Your Comprehensive Career Development Platform</h4>
    <p>Powered by Google Gemini AI | Built with Streamlit | © 2025</p>
    <p><em>Transforming careers through intelligent automation and personalized guidance</em></p>
</div>
""", unsafe_allow_html=True)
