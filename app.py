"""
AI-Based Skill Gap Analyzer - Main Streamlit Application
"""

import streamlit as st
import random
from datetime import datetime

from roles import ROLES, get_role_skills, get_all_roles
from questions import QUESTIONS, get_questions_for_skill
from evaluator import SkillEvaluator, calculate_role_readiness
from llm_service import get_llm_service
from utils import (
    format_skill_report, export_to_json, get_level_emoji,
    get_skill_icon, generate_motivational_message
)

# Page config
st.set_page_config(
    page_title="AI Skill Gap Analyzer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .skill-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'page': 'home',
        'selected_role': None,
        'selected_skills': [],
        'current_skill_idx': 0,
        'current_q_idx': 0,
        'evaluator': None,
        'answers': {},
        'eval_results': {},
        'questions_per_skill': {},
        'llm_service': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_assessment():
    """Reset for new assessment"""
    st.session_state.current_skill_idx = 0
    st.session_state.current_q_idx = 0
    st.session_state.answers = {}
    st.session_state.eval_results = {}
    st.session_state.questions_per_skill = {}
    st.session_state.evaluator = SkillEvaluator(st.session_state.llm_service)


def render_home():
    """Home page"""
    st.markdown('<h1 class="main-header">🧠 AI Skill Gap Analyzer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Discover your strengths, identify skill gaps, and get personalized learning recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Adaptive Assessment\nQuestions adjust to your level")
    with col2:
        st.markdown("### 📊 AI Analysis\nIntelligent answer evaluation")
    with col3:
        st.markdown("### 🗺️ Learning Path\nPersonalized recommendations")
    
    st.divider()
    
    # API Config (Optional)
    with st.expander("⚙️ Configure AI (Optional - Skip this!)"):
        st.info("💡 The app works WITHOUT an API key! This is optional for enhanced AI evaluation.")
        api_key = st.text_input("Groq API Key (optional)", type="password", 
                                help="Leave empty to use rule-based evaluation")
        if api_key:
            st.session_state.llm_service = get_llm_service(api_key)
            st.success("✅ AI configured!")
        else:
            st.session_state.llm_service = get_llm_service()
    
    st.divider()
    
    # Role Selection
    st.subheader("🎯 Select Your Target Role")
    
    roles = get_all_roles()
    selected_role = st.selectbox("Choose role:", roles, index=None, placeholder="Select a role...")
    
    if selected_role:
        st.session_state.selected_role = selected_role
        role_info = ROLES[selected_role]
        
        st.markdown(f"**Description:** {role_info['description']}")
        
        st.subheader("📋 Required Skills")
        skills = role_info['skills']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Core Skills:**")
            for s, m in skills.items():
                if m['type'] == 'Core':
                    st.markdown(f"- {get_skill_icon(s)} **{s}** ({'⭐' * m['weight']})")
        with col2:
            st.markdown("**Optional Skills:**")
            for s, m in skills.items():
                if m['type'] == 'Optional':
                    st.markdown(f"- {get_skill_icon(s)} {s} ({'⭐' * m['weight']})")
        
        st.divider()
        
        # Skill Selection
        st.subheader("🔧 Customize Assessment")
        all_skills = list(skills.keys())
        selected_skills = st.multiselect(
            "Select skills to assess:",
            all_skills,
            default=all_skills
        )
        st.session_state.selected_skills = selected_skills
        
        q_count = st.slider("Questions per skill:", 1, 3, 2)
        
        if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
            if selected_skills:
                reset_assessment()
                for skill in selected_skills:
                    skill_qs = get_questions_for_skill(skill)
                    if skill_qs:
                        st.session_state.questions_per_skill[skill] = random.sample(
                            skill_qs, min(q_count, len(skill_qs))
                        )
                st.session_state.page = 'assessment'
                st.rerun()
            else:
                st.error("Please select at least one skill!")


def render_assessment():
    """Assessment page"""
    role = st.session_state.selected_role
    skills = st.session_state.selected_skills
    questions = st.session_state.questions_per_skill
    
    # Progress
    total_qs = sum(len(qs) for qs in questions.values())
    answered = len(st.session_state.answers)
    
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f"### 🎯 {role}")
    with col2:
        progress = answered / total_qs if total_qs > 0 else 0
        st.progress(progress, text=f"Progress: {answered}/{total_qs}")
    with col3:
        if st.button("❌ Exit"):
            st.session_state.page = 'home'
            st.rerun()
    
    st.divider()
    
    # Current position
    skill_idx = st.session_state.current_skill_idx
    q_idx = st.session_state.current_q_idx
    
    if skill_idx >= len(skills):
        st.session_state.page = 'results'
        st.rerun()
        return
    
    current_skill = skills[skill_idx]
    skill_qs = questions.get(current_skill, [])
    
    if q_idx >= len(skill_qs):
        st.session_state.current_skill_idx += 1
        st.session_state.current_q_idx = 0
        st.rerun()
        return
    
    current_q = skill_qs[q_idx]
    skill_meta = ROLES[role]['skills'].get(current_skill, {})
    
    # Display skill info
    st.markdown(f"## {get_skill_icon(current_skill)} {current_skill}")
    st.caption(f"Type: {skill_meta.get('type', 'Core')} | Weight: {'⭐' * skill_meta.get('weight', 1)}")
    st.markdown(f"**Question {q_idx + 1} of {len(skill_qs)}**")
    
    # Question
    q_text = current_q.get('question', current_q) if isinstance(current_q, dict) else current_q
    expected = current_q.get('expected_concepts', []) if isinstance(current_q, dict) else []
    
    st.markdown(f"### 💭 {q_text}")
    
    if expected:
        with st.expander("💡 Hint: Key concepts to consider"):
            st.write(", ".join(expected))
    
    # Answer input
    answer_key = f"{current_skill}_{q_idx}"
    existing = st.session_state.answers.get(answer_key, {}).get('answer', '')
    
    answer = st.text_area(
        "Your Answer:", 
        value=existing, 
        height=200,
        placeholder="Type your answer here. Be specific and include examples where relevant.",
        key=f"ans_{answer_key}"
    )
    
    # Show previous evaluation if exists
    if answer_key in st.session_state.eval_results:
        result = st.session_state.eval_results[answer_key]
        with st.expander("📊 Previous Evaluation", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{result.get('score', 0)}/10")
            c2.metric("Level", result.get('level', 'N/A'))
            c3.metric("Concepts", len(result.get('concepts_covered', [])))
            st.markdown(f"**Feedback:** {result.get('feedback', '')}")
            if result.get('suggestions'):
                st.markdown("**Suggestions:**")
                for s in result.get('suggestions', []):
                    st.markdown(f"- {s}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if q_idx > 0 or skill_idx > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                if q_idx > 0:
                    st.session_state.current_q_idx -= 1
                else:
                    st.session_state.current_skill_idx -= 1
                    prev_skill = skills[st.session_state.current_skill_idx]
                    st.session_state.current_q_idx = len(questions.get(prev_skill, [])) - 1
                st.rerun()
    
    with col2:
        if st.button("💾 Evaluate & Save", type="primary", use_container_width=True):
            if answer and answer.strip():
                if st.session_state.evaluator is None:
                    st.session_state.evaluator = SkillEvaluator(st.session_state.llm_service)
                
                with st.spinner("🤔 Evaluating your answer..."):
                    evaluation = st.session_state.evaluator.add_answer_evaluation(
                        current_skill, q_text, answer, expected
                    )
                
                st.session_state.answers[answer_key] = {
                    'skill': current_skill,
                    'question': q_text,
                    'answer': answer
                }
                st.session_state.eval_results[answer_key] = evaluation
                st.success(f"✅ Score: {evaluation.get('score', 0)}/10 ({evaluation.get('level', 'N/A')})")
                st.rerun()
            else:
                st.warning("Please provide an answer before continuing!")
    
    with col3:
        is_last = (skill_idx == len(skills) - 1 and q_idx == len(skill_qs) - 1)
        btn_text = "📊 View Results" if is_last else "Next ➡️"
        
        if st.button(btn_text, use_container_width=True):
            # Auto-save if answer exists but not saved
            if answer_key not in st.session_state.answers and answer and answer.strip():
                if st.session_state.evaluator is None:
                    st.session_state.evaluator = SkillEvaluator(st.session_state.llm_service)
                evaluation = st.session_state.evaluator.add_answer_evaluation(
                    current_skill, q_text, answer, expected
                )
                st.session_state.answers[answer_key] = {
                    'skill': current_skill,
                    'question': q_text,
                    'answer': answer
                }
                st.session_state.eval_results[answer_key] = evaluation
            
            if is_last:
                st.session_state.page = 'results'
            else:
                st.session_state.current_q_idx += 1
            st.rerun()
    
    # Sidebar progress
    with st.sidebar:
        st.markdown("### 📋 Assessment Progress")
        for idx, skill in enumerate(skills):
            skill_qs_list = questions.get(skill, [])
            answered_count = sum(1 for i in range(len(skill_qs_list)) 
                                if f"{skill}_{i}" in st.session_state.answers)
            
            if idx < skill_idx:
                status = "✅"
            elif idx == skill_idx:
                status = "🔄"
            else:
                status = "⏳"
            
            st.markdown(f"{status} **{skill}** ({answered_count}/{len(skill_qs_list)})")


def render_results():
    """Results page"""
    st.markdown('<h1 class="main-header">📊 Skill Gap Analysis Report</h1>', unsafe_allow_html=True)
    
    role = st.session_state.selected_role
    role_skills = ROLES[role]['skills']
    assessed = {s: role_skills[s] for s in st.session_state.selected_skills if s in role_skills}
    
    if st.session_state.evaluator is None:
        st.session_state.evaluator = SkillEvaluator(st.session_state.llm_service)
    
    analysis = st.session_state.evaluator.generate_gap_analysis(role, assessed)
    readiness = calculate_role_readiness(analysis)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><h2>{analysis.overall_score}%</h2><p>Overall Score</p></div>',
                   unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h2>{readiness["status"]}</h2><p>Readiness</p></div>',
                   unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h2>{len(analysis.strengths)}</h2><p>Strengths</p></div>',
                   unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><h2>{len(analysis.gaps)}</h2><p>Skill Gaps</p></div>',
                   unsafe_allow_html=True)
    
    st.divider()
    st.markdown(f"**{readiness['message']}**")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "✅ Strengths & Gaps", "🗺️ Learning Path", "📄 Full Report"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Skill Distribution")
            for skill, assessment in analysis.skill_assessments.items():
                level_color = {
                    "Beginner": "🔴",
                    "Intermediate": "🟡", 
                    "Advanced": "🟢",
                    "Expert": "🌟"
                }
                st.markdown(
                    f"{get_skill_icon(skill)} **{skill}**: "
                    f"{level_color.get(assessment.level.value, '⚪')} "
                    f"{assessment.level.value} ({assessment.raw_score}/10)"
                )
        
        with col2:
            st.subheader("Score Breakdown")
            for skill, assessment in analysis.skill_assessments.items():
                st.progress(assessment.raw_score / 10, text=f"{skill}: {assessment.raw_score}/10")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Strength Areas")
            if analysis.strengths:
                for s in analysis.strengths:
                    st.success(f"**{s.skill}** - {s.level.value} ({s.raw_score}/10)")
            else:
                st.info("Complete more assessments to identify your strengths!")
        
        with col2:
            st.markdown("### ❌ Skill Gaps")
            if analysis.gaps:
                for g in analysis.gaps:
                    priority = "🔥 Critical" if g.skill_type == "Core" else "📌 Recommended"
                    st.error(f"**{g.skill}** - {g.level.value} ({g.raw_score}/10) - {priority}")
            else:
                st.success("🎉 No significant skill gaps identified!")
        
        st.divider()
        
        st.markdown("### 🔥 Critical Focus Areas")
        if analysis.critical_gaps:
            for crit in analysis.critical_gaps:
                st.error(f"**{crit.skill}** - Core skill requiring immediate attention")
                if crit.suggestions:
                    for sugg in crit.suggestions[:2]:
                        st.markdown(f"  - {sugg}")
        else:
            st.success("All core skills are at acceptable levels!")
    
    with tab3:
        st.markdown("### 🗺️ Personalized Learning Path")
        st.markdown("Follow this recommended order to maximize your learning efficiency:")
        
        if analysis.learning_path:
            for item in analysis.learning_path:
                with st.expander(f"**Priority {item['priority']}: {item['skill']}**", 
                               expanded=(item['priority'] <= 2)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Current Level:** {item['current_level']}")
                        st.markdown(f"**Target Level:** {item['target_level']}")
                        st.markdown(f"**Importance:** {item['importance']}")
                        st.markdown(f"**Estimated Effort:** {item['estimated_effort']}")
                    
                    with col2:
                        st.markdown("**Milestones:**")
                        # FIXED: Added enumerate with index to make keys unique
                        for idx, milestone in enumerate(item['milestones']):
                            st.checkbox(
                                milestone, 
                                key=f"milestone_{item['priority']}_{item['skill']}_{idx}",
                                value=False
                            )
                    
                    st.markdown("**Focus Areas:**")
                    for focus in item['focus_areas']:
                        st.markdown(f"- {focus}")
        else:
            st.success("🎉 Excellent! You're already well-prepared for this role!")
    
    with tab4:
        st.markdown("### 📄 Complete Report")
        
        report = format_skill_report(analysis)
        st.markdown(report)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            json_report = export_to_json(analysis)
            st.download_button(
                "📥 Download JSON Report",
                data=json_report,
                file_name=f"skill_report_{role.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                "📥 Download Text Report",
                data=report,
                file_name=f"skill_report_{role.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    st.divider()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Assessment", use_container_width=True):
            reset_assessment()
            st.session_state.page = 'assessment'
            st.rerun()
    with col2:
        if st.button("🎯 Try Different Role", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.selected_role = None
            st.session_state.selected_skills = []
            reset_assessment()
            st.rerun()
    
    st.markdown(generate_motivational_message(analysis.overall_score))


def main():
    """Main application entry point"""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Skill Gap Analyzer")
        st.divider()
        
        pages = {
            "home": "🏠 Home",
            "assessment": "📝 Assessment", 
            "results": "📊 Results"
        }
        
        for key, name in pages.items():
            if st.session_state.page == key:
                st.markdown(f"**→ {name}**")
            else:
                if st.button(name, key=f"nav_{key}"):
                    st.session_state.page = key
                    st.rerun()
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This tool helps you:
        - Identify skill strengths
        - Discover knowledge gaps
        - Get personalized learning paths
        - Track career readiness
        """)
        st.caption("Built with ❤️ using Streamlit")
    
    # Render current page
    if st.session_state.page == 'home':
        render_home()
    elif st.session_state.page == 'assessment':
        render_assessment()
    elif st.session_state.page == 'results':
        render_results()


if __name__ == "__main__":
    main()