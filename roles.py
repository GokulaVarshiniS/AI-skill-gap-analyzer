"""
Role definitions with skills, weights, and categories.
Each skill has:
- weight: importance (1-3)
- type: Core/Optional
- description: skill description for context
"""

ROLES = {
    "Data Analyst": {
        "description": "Analyzes data to help organizations make informed decisions",
        "skills": {
            "Python": {"weight": 3, "type": "Core", "description": "Programming for data manipulation"},
            "SQL": {"weight": 3, "type": "Core", "description": "Database querying and management"},
            "EDA": {"weight": 3, "type": "Core", "description": "Exploratory Data Analysis techniques"},
            "Data Visualization": {"weight": 2, "type": "Core", "description": "Creating charts and dashboards"},
            "Statistics": {"weight": 2, "type": "Core", "description": "Statistical analysis and inference"},
            "Excel": {"weight": 2, "type": "Core", "description": "Spreadsheet analysis"},
            "Machine Learning": {"weight": 1, "type": "Optional", "description": "Basic ML concepts"}
        }
    },
    
    "Data Scientist": {
        "description": "Builds predictive models and extracts insights from complex data",
        "skills": {
            "Python": {"weight": 3, "type": "Core", "description": "Programming for data science"},
            "EDA": {"weight": 3, "type": "Core", "description": "Exploratory Data Analysis"},
            "Statistics": {"weight": 3, "type": "Core", "description": "Statistical modeling"},
            "Machine Learning": {"weight": 3, "type": "Core", "description": "ML algorithms and techniques"},
            "Feature Engineering": {"weight": 2, "type": "Core", "description": "Creating meaningful features"},
            "SQL": {"weight": 2, "type": "Core", "description": "Data extraction"},
            "Deep Learning": {"weight": 1, "type": "Optional", "description": "Neural networks"}
        }
    },
    
    "Machine Learning Engineer": {
        "description": "Designs and deploys ML systems at scale",
        "skills": {
            "Python": {"weight": 3, "type": "Core", "description": "Production Python code"},
            "Machine Learning": {"weight": 3, "type": "Core", "description": "ML algorithms"},
            "Model Evaluation": {"weight": 2, "type": "Core", "description": "Metrics and validation"},
            "Feature Engineering": {"weight": 2, "type": "Core", "description": "Feature pipelines"},
            "Deployment": {"weight": 3, "type": "Core", "description": "MLOps and deployment"},
            "Deep Learning": {"weight": 2, "type": "Core", "description": "Neural networks"},
            "Cloud Services": {"weight": 1, "type": "Optional", "description": "AWS/GCP/Azure"}
        }
    },
    
    "AI Engineer": {
        "description": "Builds AI-powered applications and systems",
        "skills": {
            "Python": {"weight": 3, "type": "Core", "description": "AI development"},
            "Machine Learning": {"weight": 3, "type": "Core", "description": "ML fundamentals"},
            "Deep Learning": {"weight": 3, "type": "Core", "description": "Deep neural networks"},
            "NLP / CV": {"weight": 2, "type": "Core", "description": "NLP or Computer Vision"},
            "Model Optimization": {"weight": 2, "type": "Core", "description": "Performance tuning"},
            "Deployment": {"weight": 2, "type": "Core", "description": "Production deployment"},
            "LLM Integration": {"weight": 2, "type": "Core", "description": "Working with LLMs"}
        }
    },
    
    "Backend Developer": {
        "description": "Builds server-side applications and APIs",
        "skills": {
            "Python": {"weight": 3, "type": "Core", "description": "Backend development"},
            "SQL": {"weight": 3, "type": "Core", "description": "Database design"},
            "API Development": {"weight": 3, "type": "Core", "description": "REST/GraphQL APIs"},
            "System Design": {"weight": 2, "type": "Core", "description": "Architecture patterns"},
            "Cloud Services": {"weight": 2, "type": "Core", "description": "Cloud deployment"},
            "Docker": {"weight": 2, "type": "Core", "description": "Containerization"},
            "Machine Learning": {"weight": 1, "type": "Optional", "description": "ML integration"}
        }
    },
    
    "Full Stack Developer": {
        "description": "Builds complete web applications",
        "skills": {
            "Python": {"weight": 2, "type": "Core", "description": "Backend Python"},
            "JavaScript": {"weight": 3, "type": "Core", "description": "Frontend development"},
            "SQL": {"weight": 2, "type": "Core", "description": "Database management"},
            "API Development": {"weight": 2, "type": "Core", "description": "API design"},
            "HTML/CSS": {"weight": 2, "type": "Core", "description": "Web fundamentals"},
            "React/Vue": {"weight": 2, "type": "Core", "description": "Frontend frameworks"},
            "DevOps": {"weight": 1, "type": "Optional", "description": "CI/CD pipelines"}
        }
    }
}

# Skill categories for grouping
SKILL_CATEGORIES = {
    "Programming": ["Python", "JavaScript", "SQL"],
    "Data Analysis": ["EDA", "Statistics", "Data Visualization", "Excel"],
    "Machine Learning": ["Machine Learning", "Deep Learning", "Feature Engineering", "Model Evaluation", "Model Optimization"],
    "AI Specialization": ["NLP / CV", "LLM Integration"],
    "Engineering": ["Deployment", "API Development", "System Design", "Docker", "Cloud Services", "DevOps"],
    "Web Development": ["HTML/CSS", "React/Vue"]
}

def get_role_skills(role_name: str) -> dict:
    """Get skills for a specific role"""
    if role_name in ROLES:
        return ROLES[role_name]["skills"]
    return {}

def get_all_roles() -> list:
    """Get list of all available roles"""
    return list(ROLES.keys())

def get_core_skills(role_name: str) -> list:
    """Get core skills for a role"""
    skills = get_role_skills(role_name)
    return [skill for skill, meta in skills.items() if meta["type"] == "Core"]

def get_optional_skills(role_name: str) -> list:
    """Get optional skills for a role"""
    skills = get_role_skills(role_name)
    return [skill for skill, meta in skills.items() if meta["type"] == "Optional"]