"""
Skill evaluation and gap analysis logic.
Combines LLM-based answer evaluation with weighted scoring and progress tracking.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SkillLevel(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


@dataclass
class SkillAssessment:
    """Assessment result for a single skill"""
    skill: str
    skill_type: str  # Core or Optional
    weight: int
    raw_score: float  # 0-10
    weighted_score: float
    level: SkillLevel
    concepts_covered: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    questions_answered: int = 0
    feedback: str = ""


@dataclass
class GapAnalysis:
    """Complete gap analysis for a role"""
    role: str
    overall_score: float
    overall_level: str
    strengths: List[SkillAssessment]
    gaps: List[SkillAssessment]
    critical_gaps: List[SkillAssessment]
    optional_gaps: List[SkillAssessment]
    skill_assessments: Dict[str, SkillAssessment]
    priority_order: List[str]
    learning_path: List[Dict]


class SkillEvaluator:
    """Main evaluator class for skill gap analysis"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.skill_scores: Dict[str, List[Dict]] = {}
    
    def add_answer_evaluation(
        self,
        skill: str,
        question: str,
        answer: str,
        expected_concepts: List[str] = None
    ) -> Dict:
        """Add an answer evaluation for a skill"""
        
        if self.llm_service:
            evaluation = self.llm_service.evaluate_answer(
                question=question,
                answer=answer,
                skill=skill,
                expected_concepts=expected_concepts
            )
        else:
            evaluation = self._basic_evaluate(answer)
        
        if skill not in self.skill_scores:
            self.skill_scores[skill] = []
        
        self.skill_scores[skill].append(evaluation)
        return evaluation
    
    def _basic_evaluate(self, answer: str) -> Dict:
        """Basic evaluation without LLM"""
        length = len(answer.strip())
        word_count = len(answer.split())
        
        if length < 20:
            score, level = 2, "Beginner"
        elif length < 60:
            score, level = 4, "Beginner"
        elif word_count < 30:
            score, level = 5, "Intermediate"
        elif word_count < 60:
            score, level = 7, "Advanced"
        else:
            score, level = 8, "Advanced"
        
        return {
            "score": score,
            "level": level,
            "feedback": "Answer evaluated based on depth and detail.",
            "concepts_covered": [],
            "suggestions": ["Provide more specific examples and explanations."]
        }
    
    def calculate_skill_assessment(
        self,
        skill: str,
        skill_meta: Dict
    ) -> SkillAssessment:
        """Calculate overall assessment for a skill"""
        
        evaluations = self.skill_scores.get(skill, [])
        
        if not evaluations:
            return SkillAssessment(
                skill=skill,
                skill_type=skill_meta.get("type", "Core"),
                weight=skill_meta.get("weight", 1),
                raw_score=0,
                weighted_score=0,
                level=SkillLevel.BEGINNER,
                questions_answered=0,
                feedback="No questions answered for this skill."
            )
        
        # Calculate average score
        avg_score = sum(e["score"] for e in evaluations) / len(evaluations)
        weighted_score = avg_score * skill_meta.get("weight", 1)
        
        # Determine level
        if avg_score < 3:
            level = SkillLevel.BEGINNER
        elif avg_score < 6:
            level = SkillLevel.INTERMEDIATE
        elif avg_score < 8:
            level = SkillLevel.ADVANCED
        else:
            level = SkillLevel.EXPERT
        
        # Aggregate concepts and suggestions
        all_concepts = []
        all_suggestions = []
        for e in evaluations:
            all_concepts.extend(e.get("concepts_covered", []))
            all_suggestions.extend(e.get("suggestions", []))
        
        # Deduplicate
        unique_concepts = list(set(all_concepts))
        unique_suggestions = list(set(all_suggestions))[:5]
        
        # Generate feedback
        latest_feedback = evaluations[-1].get("feedback", "") if evaluations else ""
        
        return SkillAssessment(
            skill=skill,
            skill_type=skill_meta.get("type", "Core"),
            weight=skill_meta.get("weight", 1),
            raw_score=round(avg_score, 1),
            weighted_score=round(weighted_score, 1),
            level=level,
            concepts_covered=unique_concepts,
            suggestions=unique_suggestions,
            questions_answered=len(evaluations),
            feedback=latest_feedback
        )
    
    def generate_gap_analysis(
        self,
        role: str,
        role_skills: Dict
    ) -> GapAnalysis:
        """Generate complete gap analysis for a role"""
        
        assessments: Dict[str, SkillAssessment] = {}
        strengths: List[SkillAssessment] = []
        gaps: List[SkillAssessment] = []
        critical_gaps: List[SkillAssessment] = []
        optional_gaps: List[SkillAssessment] = []
        
        total_weighted_score = 0
        max_weighted_score = 0
        
        # Calculate assessments for each skill
        for skill, meta in role_skills.items():
            assessment = self.calculate_skill_assessment(skill, meta)
            assessments[skill] = assessment
            
            total_weighted_score += assessment.weighted_score
            max_weighted_score += 10 * meta.get("weight", 1)
            
            # Categorize as strength or gap
            if assessment.level in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
                strengths.append(assessment)
            elif assessment.level == SkillLevel.BEGINNER:
                gaps.append(assessment)
                if assessment.skill_type == "Core":
                    critical_gaps.append(assessment)
                else:
                    optional_gaps.append(assessment)
            else:  # Intermediate
                if assessment.skill_type == "Core":
                    gaps.append(assessment)
        
        # Calculate overall score
        overall_score = (total_weighted_score / max_weighted_score * 100) if max_weighted_score > 0 else 0
        
        # Determine overall level
        if overall_score < 30:
            overall_level = "Beginner"
        elif overall_score < 50:
            overall_level = "Intermediate"
        elif overall_score < 75:
            overall_level = "Advanced"
        else:
            overall_level = "Expert"
        
        # Generate priority order
        priority_order = self._calculate_priority_order(assessments, role_skills)
        
        # Generate learning path
        learning_path = self._generate_learning_path(critical_gaps + gaps, role_skills)
        
        return GapAnalysis(
            role=role,
            overall_score=round(overall_score, 1),
            overall_level=overall_level,
            strengths=strengths,
            gaps=gaps,
            critical_gaps=critical_gaps,
            optional_gaps=optional_gaps,
            skill_assessments=assessments,
            priority_order=priority_order,
            learning_path=learning_path
        )
    
    def _calculate_priority_order(
        self,
        assessments: Dict[str, SkillAssessment],
        role_skills: Dict
    ) -> List[str]:
        """Calculate priority order for skill improvement"""
        
        priority_scores = []
        
        for skill, assessment in assessments.items():
            meta = role_skills.get(skill, {})
            
            # Priority factors:
            # 1. Core skills over optional
            type_factor = 3 if assessment.skill_type == "Core" else 1
            
            # 2. Higher weight skills
            weight_factor = meta.get("weight", 1)
            
            # 3. Lower current score = higher priority
            gap_factor = (10 - assessment.raw_score) / 10
            
            priority = type_factor * weight_factor * gap_factor
            priority_scores.append((skill, priority, assessment.raw_score))
        
        # Sort by priority (highest first), then by current score (lowest first)
        priority_scores.sort(key=lambda x: (-x[1], x[2]))
        
        return [skill for skill, _, _ in priority_scores]
    
    def _generate_learning_path(
        self,
        gaps: List[SkillAssessment],
        role_skills: Dict
    ) -> List[Dict]:
        """Generate personalized learning path"""
        
        learning_path = []
        
        # Sort gaps by priority (Core skills first, then by weight)
        sorted_gaps = sorted(
            gaps,
            key=lambda x: (
                0 if x.skill_type == "Core" else 1,
                -x.weight,
                x.raw_score
            )
        )
        
        for i, gap in enumerate(sorted_gaps[:5], 1):  # Top 5 priorities
            path_item = {
                "priority": i,
                "skill": gap.skill,
                "current_level": gap.level.value,
                "target_level": "Advanced",
                "importance": "Critical" if gap.skill_type == "Core" else "Recommended",
                "focus_areas": gap.suggestions[:3] if gap.suggestions else [
                    f"Build foundational knowledge in {gap.skill}",
                    f"Practice with real-world {gap.skill} problems",
                    f"Complete hands-on projects using {gap.skill}"
                ],
                "estimated_effort": self._estimate_effort(gap),
                "milestones": [
                    f"Complete basic {gap.skill} tutorial",
                    f"Build a mini-project using {gap.skill}",
                    f"Solve 10 practice problems",
                    f"Contribute to a {gap.skill} project"
                ]
            }
            learning_path.append(path_item)
        
        return learning_path
    
    def _estimate_effort(self, gap: SkillAssessment) -> str:
        """Estimate learning effort based on current level"""
        if gap.level == SkillLevel.BEGINNER:
            return "4-6 weeks (8-10 hours/week)"
        elif gap.level == SkillLevel.INTERMEDIATE:
            return "2-4 weeks (6-8 hours/week)"
        else:
            return "1-2 weeks (4-6 hours/week)"
    
    def get_adaptive_difficulty(self, skill: str) -> str:
        """Get recommended difficulty for next question based on performance"""
        evaluations = self.skill_scores.get(skill, [])
        
        if not evaluations:
            return "basic"
        
        avg_score = sum(e["score"] for e in evaluations) / len(evaluations)
        
        if avg_score < 4:
            return "basic"
        elif avg_score < 7:
            return "intermediate"
        else:
            return "advanced"
    
    def reset(self):
        """Reset all evaluations"""
        self.skill_scores = {}


def calculate_role_readiness(gap_analysis: GapAnalysis) -> Dict:
    """Calculate readiness score for the target role"""
    
    # Count skills at each level
    level_counts = {
        "Expert": 0,
        "Advanced": 0,
        "Intermediate": 0,
        "Beginner": 0
    }
    
    core_skills_ready = 0
    total_core_skills = 0
    
    for skill, assessment in gap_analysis.skill_assessments.items():
        level_counts[assessment.level.value] += 1
        
        if assessment.skill_type == "Core":
            total_core_skills += 1
            if assessment.level in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
                core_skills_ready += 1
    
    core_readiness = (core_skills_ready / total_core_skills * 100) if total_core_skills > 0 else 0
    
    # Determine readiness status
    if core_readiness >= 80 and gap_analysis.overall_score >= 70:
        status = "Ready"
        message = "You're well-prepared for this role. Focus on optional skills to stand out."
    elif core_readiness >= 50 and gap_analysis.overall_score >= 50:
        status = "Almost Ready"
        message = "You have a solid foundation. Focus on critical gaps to become fully ready."
    elif core_readiness >= 30:
        status = "In Progress"
        message = "You're making progress. Prioritize core skills and build practical projects."
    else:
        status = "Getting Started"
        message = "Focus on building foundational skills. Follow the learning path step by step."
    
    return {
        "status": status,
        "message": message,
        "core_readiness": round(core_readiness, 1),
        "overall_readiness": gap_analysis.overall_score,
        "level_distribution": level_counts,
        "skills_to_improve": len(gap_analysis.gaps),
        "critical_skills_remaining": len(gap_analysis.critical_gaps)
    }