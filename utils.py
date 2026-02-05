"""
Utility functions for the Skill Gap Analyzer.
"""

import json
from typing import Dict, List
from datetime import datetime


def format_skill_report(gap_analysis) -> str:
    """Format gap analysis as a readable report"""
    
    report = []
    report.append(f"# Skill Gap Analysis Report")
    report.append(f"**Target Role:** {gap_analysis.role}")
    report.append(f"**Overall Score:** {gap_analysis.overall_score}%")
    report.append(f"**Level:** {gap_analysis.overall_level}")
    report.append("")
    
    # Strengths
    report.append("## ✅ Strengths")
    if gap_analysis.strengths:
        for s in gap_analysis.strengths:
            report.append(f"- **{s.skill}** ({s.level.value}) - Score: {s.raw_score}/10")
    else:
        report.append("- No strong skills identified yet")
    report.append("")
    
    # Gaps
    report.append("## ❌ Skill Gaps")
    if gap_analysis.gaps:
        for g in gap_analysis.gaps:
            report.append(f"- **{g.skill}** ({g.level.value}) - Score: {g.raw_score}/10")
    else:
        report.append("- No significant gaps")
    report.append("")
    
    # Critical Focus
    report.append("## 🔥 Critical Focus Areas")
    if gap_analysis.critical_gaps:
        for c in gap_analysis.critical_gaps:
            report.append(f"- **{c.skill}** - {c.skill_type} skill with weight {c.weight}")
    else:
        report.append("- Core skills are well covered")
    report.append("")
    
    # Priority Order
    report.append("## 📋 Priority Learning Order")
    for i, skill in enumerate(gap_analysis.priority_order[:5], 1):
        assessment = gap_analysis.skill_assessments.get(skill)
        if assessment:
            report.append(f"{i}. **{skill}** - Current: {assessment.level.value}")
    report.append("")
    
    # Learning Path
    report.append("## 🗺️ Learning Path")
    for item in gap_analysis.learning_path:
        report.append(f"### {item['priority']}. {item['skill']}")
        report.append(f"- Current Level: {item['current_level']}")
        report.append(f"- Target Level: {item['target_level']}")
        report.append(f"- Estimated Effort: {item['estimated_effort']}")
        report.append("- Focus Areas:")
        for focus in item['focus_areas']:
            report.append(f"  - {focus}")
        report.append("")
    
    return "\n".join(report)


def export_to_json(gap_analysis, filepath: str = None) -> str:
    """Export gap analysis to JSON"""
    
    data = {
        "role": gap_analysis.role,
        "overall_score": gap_analysis.overall_score,
        "overall_level": gap_analysis.overall_level,
        "timestamp": datetime.now().isoformat(),
        "strengths": [
            {
                "skill": s.skill,
                "score": s.raw_score,
                "level": s.level.value
            } for s in gap_analysis.strengths
        ],
        "gaps": [
            {
                "skill": g.skill,
                "score": g.raw_score,
                "level": g.level.value,
                "type": g.skill_type
            } for g in gap_analysis.gaps
        ],
        "priority_order": gap_analysis.priority_order,
        "learning_path": gap_analysis.learning_path
    }
    
    json_str = json.dumps(data, indent=2)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(json_str)
    
    return json_str


def get_progress_bar_color(score: float) -> str:
    """Get color for progress bar based on score"""
    if score < 30:
        return "red"
    elif score < 50:
        return "orange"
    elif score < 70:
        return "yellow"
    else:
        return "green"


def get_level_emoji(level: str) -> str:
    """Get emoji for skill level"""
    level_emojis = {
        "Beginner": "🌱",
        "Intermediate": "🌿",
        "Advanced": "🌳",
        "Expert": "🏆"
    }
    return level_emojis.get(level, "📊")


def get_skill_icon(skill: str) -> str:
    """Get icon for skill category"""
    skill_icons = {
        "Python": "🐍",
        "SQL": "🗄️",
        "Machine Learning": "🤖",
        "Deep Learning": "🧠",
        "Statistics": "📊",
        "EDA": "🔍",
        "Feature Engineering": "🔧",
        "Deployment": "🚀",
        "Data Visualization": "📈",
        "NLP / CV": "💬",
        "Model Optimization": "⚡",
        "Model Evaluation": "✅",
        "API Development": "🔌",
        "Cloud Services": "☁️"
    }
    return skill_icons.get(skill, "📚")


def calculate_completion_percentage(answered: int, total: int) -> float:
    """Calculate completion percentage"""
    if total == 0:
        return 0
    return round((answered / total) * 100, 1)


def generate_motivational_message(score: float) -> str:
    """Generate motivational message based on score"""
    if score >= 80:
        return "🌟 Excellent! You're demonstrating expert-level knowledge!"
    elif score >= 60:
        return "💪 Great progress! You have a solid foundation."
    elif score >= 40:
        return "📈 Good start! Keep practicing to improve."
    else:
        return "🌱 Every expert was once a beginner. Keep learning!"