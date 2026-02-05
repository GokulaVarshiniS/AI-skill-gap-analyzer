"""
LLM Service for intelligent answer evaluation.
Supports multiple providers: Groq (free), OpenAI, and fallback rule-based evaluation.
"""

import os
import re
import json
from typing import Tuple, Dict, List

# Try to import LLM libraries
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMService:
    """Service for LLM-based answer evaluation"""
    
    def __init__(self, provider: str = "groq", api_key: str = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == "groq" and GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
        elif self.provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.client = openai
        else:
            self.client = None
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        skill: str,
        expected_concepts: List[str] = None
    ) -> Dict:
        """
        Evaluate an answer using LLM or fallback to rule-based evaluation.
        
        Returns:
            Dict with keys: score (0-10), level, feedback, concepts_covered, suggestions
        """
        if not answer or not answer.strip():
            return {
                "score": 0,
                "level": "No Response",
                "feedback": "No answer provided.",
                "concepts_covered": [],
                "suggestions": ["Please provide an answer to the question."]
            }
        
        # Try LLM evaluation first
        if self.client:
            try:
                return self._llm_evaluate(question, answer, skill, expected_concepts)
            except Exception as e:
                print(f"LLM evaluation failed: {e}")
        
        # Fallback to rule-based evaluation
        return self._rule_based_evaluate(question, answer, skill, expected_concepts)
    
    def _llm_evaluate(
        self,
        question: str,
        answer: str,
        skill: str,
        expected_concepts: List[str]
    ) -> Dict:
        """Evaluate answer using LLM"""
        
        concepts_str = ", ".join(expected_concepts) if expected_concepts else "relevant technical concepts"
        
        prompt = f"""You are an expert technical interviewer evaluating a candidate's response.

SKILL BEING ASSESSED: {skill}

QUESTION: {question}

EXPECTED CONCEPTS: {concepts_str}

CANDIDATE'S ANSWER: {answer}

Evaluate the answer and provide a JSON response with the following structure:
{{
    "score": <number from 0-10>,
    "level": "<Beginner|Intermediate|Advanced|Expert>",
    "feedback": "<2-3 sentences of constructive feedback>",
    "concepts_covered": [<list of concepts the candidate demonstrated understanding of>],
    "concepts_missing": [<list of important concepts not mentioned>],
    "suggestions": [<2-3 specific suggestions for improvement>]
}}

Scoring guidelines:
- 0-2: No understanding or irrelevant answer
- 3-4: Basic understanding with significant gaps
- 5-6: Moderate understanding, covers basics
- 7-8: Good understanding with practical knowledge
- 9-10: Expert-level understanding with deep insights

Respond ONLY with the JSON object, no additional text."""

        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            else:  # OpenAI
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            
            # Parse JSON response
            result = self._parse_llm_response(response_text)
            return result
            
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM JSON response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Ensure all required fields are present
                return {
                    "score": result.get("score", 5),
                    "level": result.get("level", "Intermediate"),
                    "feedback": result.get("feedback", "Answer evaluated."),
                    "concepts_covered": result.get("concepts_covered", []),
                    "suggestions": result.get("suggestions", result.get("concepts_missing", []))
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback if parsing fails
        return {
            "score": 5,
            "level": "Intermediate",
            "feedback": "Answer evaluated. Consider providing more detailed explanations.",
            "concepts_covered": [],
            "suggestions": ["Expand on your answer with examples."]
        }
    
    def _rule_based_evaluate(
        self,
        question: str,
        answer: str,
        skill: str,
        expected_concepts: List[str]
    ) -> Dict:
        """Rule-based evaluation fallback"""
        
        answer_lower = answer.lower().strip()
        answer_length = len(answer_lower)
        word_count = len(answer_lower.split())
        
        # Check for concept coverage
        concepts_covered = []
        concepts_missing = []
        
        if expected_concepts:
            for concept in expected_concepts:
                if concept.lower() in answer_lower:
                    concepts_covered.append(concept)
                else:
                    concepts_missing.append(concept)
        
        # Calculate base score
        length_score = min(3, word_count // 15)  # Up to 3 points for length
        concept_score = (len(concepts_covered) / max(len(expected_concepts), 1)) * 4 if expected_concepts else 2
        
        # Check for examples and specificity
        has_example = any(word in answer_lower for word in ["example", "for instance", "such as", "e.g.", "like when"])
        example_score = 1.5 if has_example else 0
        
        # Check for technical depth
        technical_indicators = ["because", "therefore", "this means", "in practice", "the reason", "works by"]
        depth_score = 1.5 if any(ind in answer_lower for ind in technical_indicators) else 0
        
        total_score = min(10, length_score + concept_score + example_score + depth_score)
        
        # Determine level
        if total_score < 3:
            level = "Beginner"
        elif total_score < 5:
            level = "Intermediate"
        elif total_score < 8:
            level = "Advanced"
        else:
            level = "Expert"
        
        # Generate feedback
        feedback_parts = []
        if len(concepts_covered) > 0:
            feedback_parts.append(f"Good coverage of: {', '.join(concepts_covered[:3])}.")
        if not has_example:
            feedback_parts.append("Consider adding specific examples.")
        if word_count < 30:
            feedback_parts.append("Try to provide more detailed explanations.")
        
        feedback = " ".join(feedback_parts) if feedback_parts else "Answer provides a basic explanation."
        
        # Generate suggestions
        suggestions = []
        if concepts_missing:
            suggestions.append(f"Explore: {', '.join(concepts_missing[:3])}")
        if not has_example:
            suggestions.append("Include real-world examples to demonstrate practical understanding")
        if word_count < 50:
            suggestions.append("Expand your answer with more details and context")
        
        return {
            "score": round(total_score, 1),
            "level": level,
            "feedback": feedback,
            "concepts_covered": concepts_covered,
            "suggestions": suggestions[:3] if suggestions else ["Continue practicing this topic"]
        }
    
    def generate_learning_path(self, skill_gaps: List[Dict]) -> List[Dict]:
        """Generate personalized learning recommendations"""
        
        recommendations = []
        
        resources = {
            "Python": {
                "beginner": ["Python Crash Course (book)", "Codecademy Python", "freeCodeCamp Python"],
                "intermediate": ["Fluent Python", "Real Python tutorials", "LeetCode Python problems"],
                "advanced": ["Python internals course", "Contributing to open source", "System design with Python"]
            },
            "SQL": {
                "beginner": ["SQLBolt", "Mode Analytics SQL Tutorial", "W3Schools SQL"],
                "intermediate": ["LeetCode SQL", "Stratascratch", "DataLemur SQL"],
                "advanced": ["Database internals", "Query optimization guides", "PostgreSQL documentation"]
            },
            "Machine Learning": {
                "beginner": ["Andrew Ng's ML Course", "Scikit-learn tutorials", "Kaggle Learn ML"],
                "intermediate": ["Hands-On ML with Scikit-Learn (book)", "Fast.ai", "Kaggle competitions"],
                "advanced": ["Papers With Code", "ML system design", "Research paper reading groups"]
            },
            "Deep Learning": {
                "beginner": ["Deep Learning Specialization (Coursera)", "PyTorch tutorials"],
                "intermediate": ["Fast.ai courses", "d2l.ai book", "Implement papers from scratch"],
                "advanced": ["Research papers", "Custom architectures", "Production DL systems"]
            }
        }
        
        for gap in skill_gaps:
            skill = gap.get("skill", "")
            level = gap.get("level", "beginner").lower()
            priority = gap.get("priority", "medium")
            
            skill_resources = resources.get(skill, {})
            level_resources = skill_resources.get(level, ["Practice problems", "Online tutorials", "Project-based learning"])
            
            recommendations.append({
                "skill": skill,
                "priority": priority,
                "current_level": level,
                "target_level": "advanced" if level != "advanced" else "expert",
                "resources": level_resources,
                "estimated_time": "2-4 weeks" if level == "beginner" else "4-8 weeks",
                "action_items": [
                    f"Complete foundational exercises in {skill}",
                    f"Build a project demonstrating {skill}",
                    f"Practice with real-world datasets/problems"
                ]
            })
        
        return recommendations


def get_llm_service(api_key: str = None) -> LLMService:
    """Factory function to get LLM service with available provider"""
    
    # Try Groq first (free tier available)
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    if groq_key and GROQ_AVAILABLE:
        return LLMService(provider="groq", api_key=groq_key)
    
    # Try OpenAI
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if openai_key and OPENAI_AVAILABLE:
        return LLMService(provider="openai", api_key=openai_key)
    
    # Return service without API (will use rule-based evaluation)
    return LLMService(provider="none", api_key=None)