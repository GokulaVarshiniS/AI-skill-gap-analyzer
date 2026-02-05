"""
Question bank with multiple difficulty levels and follow-up questions.
Each skill has questions at different levels:
- basic: Fundamental concepts
- intermediate: Applied knowledge
- advanced: Deep understanding and real-world scenarios
"""

QUESTIONS = {
    "Python": {
        "basic": [
            {
                "question": "What are the main data types in Python? Give examples of when you'd use each.",
                "expected_concepts": ["int", "float", "string", "list", "dict", "tuple", "set"],
                "follow_up": "How would you choose between a list and a tuple?"
            },
            {
                "question": "Explain the difference between a list and a dictionary in Python.",
                "expected_concepts": ["ordered", "key-value", "indexing", "hashing"],
                "follow_up": "When would you prefer a dictionary over a list?"
            }
        ],
        "intermediate": [
            {
                "question": "Explain the difference between a list and a generator. When would you use each?",
                "expected_concepts": ["memory efficiency", "lazy evaluation", "iteration", "yield"],
                "follow_up": "How would you convert a list comprehension to a generator?"
            },
            {
                "question": "What are decorators in Python and how do they work?",
                "expected_concepts": ["wrapper", "function", "modification", "@syntax"],
                "follow_up": "Can you give an example of a practical use case?"
            }
        ],
        "advanced": [
            {
                "question": "Explain Python's Global Interpreter Lock (GIL) and its implications for multi-threading.",
                "expected_concepts": ["thread safety", "CPU-bound", "multiprocessing", "concurrency"],
                "follow_up": "How would you handle CPU-intensive tasks in Python?"
            },
            {
                "question": "How does Python's garbage collection work? What are reference cycles?",
                "expected_concepts": ["reference counting", "garbage collector", "memory", "cycles"],
                "follow_up": "How can you identify memory leaks in Python?"
            }
        ]
    },
    
    "SQL": {
        "basic": [
            {
                "question": "What is the difference between WHERE and HAVING clauses?",
                "expected_concepts": ["filtering", "aggregate", "group by", "conditions"],
                "follow_up": "Can you use both in the same query?"
            },
            {
                "question": "Explain the different types of JOINs in SQL.",
                "expected_concepts": ["INNER", "LEFT", "RIGHT", "FULL", "CROSS"],
                "follow_up": "When would you use a LEFT JOIN over an INNER JOIN?"
            }
        ],
        "intermediate": [
            {
                "question": "How do indexes improve query performance? What are the trade-offs?",
                "expected_concepts": ["B-tree", "lookup speed", "write overhead", "index types"],
                "follow_up": "How would you decide which columns to index?"
            },
            {
                "question": "Explain window functions with a practical example.",
                "expected_concepts": ["ROW_NUMBER", "RANK", "PARTITION BY", "OVER"],
                "follow_up": "How is a window function different from GROUP BY?"
            }
        ],
        "advanced": [
            {
                "question": "How would you optimize a slow-running query? Walk through your approach.",
                "expected_concepts": ["EXPLAIN", "execution plan", "indexing", "query restructuring"],
                "follow_up": "What tools do you use for query analysis?"
            },
            {
                "question": "Explain database normalization and when you might denormalize.",
                "expected_concepts": ["normal forms", "redundancy", "performance", "read vs write"],
                "follow_up": "How do you balance normalization with query performance?"
            }
        ]
    },
    
    "EDA": {
        "basic": [
            {
                "question": "What are the first steps you take when exploring a new dataset?",
                "expected_concepts": ["shape", "types", "missing values", "statistics", "distribution"],
                "follow_up": "How do you document your findings?"
            },
            {
                "question": "How do you identify and handle missing values?",
                "expected_concepts": ["detection", "imputation", "deletion", "patterns"],
                "follow_up": "When would you drop rows vs impute values?"
            }
        ],
        "intermediate": [
            {
                "question": "How do you detect and handle outliers in your data?",
                "expected_concepts": ["IQR", "z-score", "visualization", "domain knowledge"],
                "follow_up": "Are outliers always bad? When might you keep them?"
            },
            {
                "question": "What insights do you try to extract during EDA?",
                "expected_concepts": ["patterns", "relationships", "distributions", "anomalies"],
                "follow_up": "How do you communicate these insights to stakeholders?"
            }
        ],
        "advanced": [
            {
                "question": "How do you handle highly skewed data? What transformations would you consider?",
                "expected_concepts": ["log transform", "Box-Cox", "normalization", "impact on models"],
                "follow_up": "How does skewness affect different ML algorithms?"
            },
            {
                "question": "Describe your approach to analyzing multivariate relationships.",
                "expected_concepts": ["correlation", "heatmap", "pair plots", "dimensionality reduction"],
                "follow_up": "How do you identify multicollinearity?"
            }
        ]
    },
    
    "Statistics": {
        "basic": [
            {
                "question": "Explain the difference between mean, median, and mode. When would you use each?",
                "expected_concepts": ["central tendency", "skewness", "outliers", "distribution"],
                "follow_up": "Which is more robust to outliers?"
            },
            {
                "question": "What is standard deviation and why is it important?",
                "expected_concepts": ["spread", "variance", "normal distribution", "variability"],
                "follow_up": "How does it relate to variance?"
            }
        ],
        "intermediate": [
            {
                "question": "Explain the difference between correlation and causation with an example.",
                "expected_concepts": ["relationship", "cause-effect", "confounding", "experiments"],
                "follow_up": "How would you establish causation?"
            },
            {
                "question": "What is a p-value and how do you interpret it?",
                "expected_concepts": ["significance", "null hypothesis", "threshold", "Type I error"],
                "follow_up": "What are the common misconceptions about p-values?"
            }
        ],
        "advanced": [
            {
                "question": "Explain Bayesian vs Frequentist approaches to statistics.",
                "expected_concepts": ["prior", "posterior", "probability interpretation", "updating beliefs"],
                "follow_up": "When would you prefer a Bayesian approach?"
            },
            {
                "question": "How do you handle multiple hypothesis testing?",
                "expected_concepts": ["Bonferroni", "FDR", "family-wise error", "correction methods"],
                "follow_up": "What is the difference between FWER and FDR?"
            }
        ]
    },
    
    "Machine Learning": {
        "basic": [
            {
                "question": "What is the difference between supervised and unsupervised learning?",
                "expected_concepts": ["labeled data", "classification", "regression", "clustering"],
                "follow_up": "Can you give examples of each?"
            },
            {
                "question": "Explain overfitting and how to prevent it.",
                "expected_concepts": ["training vs test", "generalization", "regularization", "validation"],
                "follow_up": "How do you detect overfitting?"
            }
        ],
        "intermediate": [
            {
                "question": "Walk me through the complete ML pipeline from data to deployment.",
                "expected_concepts": ["data prep", "feature engineering", "training", "validation", "deployment"],
                "follow_up": "What's the most challenging part of this pipeline?"
            },
            {
                "question": "How do you select the right model for a problem?",
                "expected_concepts": ["problem type", "data size", "interpretability", "performance"],
                "follow_up": "What trade-offs do you consider?"
            }
        ],
        "advanced": [
            {
                "question": "Explain the bias-variance tradeoff and its implications.",
                "expected_concepts": ["underfitting", "overfitting", "model complexity", "error decomposition"],
                "follow_up": "How do ensemble methods address this?"
            },
            {
                "question": "How would you handle a highly imbalanced classification problem?",
                "expected_concepts": ["resampling", "SMOTE", "class weights", "metrics", "threshold"],
                "follow_up": "Why is accuracy misleading for imbalanced data?"
            }
        ]
    },
    
    "Feature Engineering": {
        "basic": [
            {
                "question": "What is feature engineering and why is it important?",
                "expected_concepts": ["transformation", "creation", "model performance", "domain knowledge"],
                "follow_up": "Can you give a simple example?"
            },
            {
                "question": "How do you handle categorical variables in ML?",
                "expected_concepts": ["one-hot encoding", "label encoding", "ordinal", "cardinality"],
                "follow_up": "What about high-cardinality categories?"
            }
        ],
        "intermediate": [
            {
                "question": "Describe different feature transformation techniques you've used.",
                "expected_concepts": ["scaling", "normalization", "log transform", "binning", "polynomial"],
                "follow_up": "When would you use each transformation?"
            },
            {
                "question": "How do you approach feature selection?",
                "expected_concepts": ["correlation", "importance", "recursive elimination", "dimensionality"],
                "follow_up": "What's the difference between filter and wrapper methods?"
            }
        ],
        "advanced": [
            {
                "question": "How do you engineer features for time series data?",
                "expected_concepts": ["lag features", "rolling statistics", "seasonality", "trends"],
                "follow_up": "How do you avoid data leakage with time series?"
            },
            {
                "question": "Explain feature engineering for text data.",
                "expected_concepts": ["tokenization", "TF-IDF", "embeddings", "n-grams"],
                "follow_up": "When would you use TF-IDF vs embeddings?"
            }
        ]
    },
    
    "Model Evaluation": {
        "basic": [
            {
                "question": "What metrics would you use for a classification problem?",
                "expected_concepts": ["accuracy", "precision", "recall", "F1", "confusion matrix"],
                "follow_up": "When is accuracy not a good metric?"
            },
            {
                "question": "Explain cross-validation and why it's useful.",
                "expected_concepts": ["train-test split", "k-fold", "validation", "generalization"],
                "follow_up": "What is stratified cross-validation?"
            }
        ],
        "intermediate": [
            {
                "question": "Explain the precision-recall tradeoff with a real-world example.",
                "expected_concepts": ["threshold", "business impact", "false positives", "false negatives"],
                "follow_up": "How do you choose the right threshold?"
            },
            {
                "question": "What is AUC-ROC and how do you interpret it?",
                "expected_concepts": ["curve", "threshold-independent", "TPR", "FPR"],
                "follow_up": "When is AUC-PR preferred over AUC-ROC?"
            }
        ],
        "advanced": [
            {
                "question": "How do you evaluate regression models beyond MSE/MAE?",
                "expected_concepts": ["R-squared", "residual analysis", "heteroscedasticity", "MAPE"],
                "follow_up": "What do residual plots tell you?"
            },
            {
                "question": "How would you evaluate a recommendation system?",
                "expected_concepts": ["precision@k", "recall@k", "NDCG", "A/B testing", "offline vs online"],
                "follow_up": "Why is online evaluation important?"
            }
        ]
    },
    
    "Deep Learning": {
        "basic": [
            {
                "question": "What is a neural network and how does it learn?",
                "expected_concepts": ["layers", "weights", "backpropagation", "gradient descent"],
                "follow_up": "What is an activation function?"
            },
            {
                "question": "Explain the difference between CNN and RNN.",
                "expected_concepts": ["convolution", "sequence", "spatial", "temporal", "use cases"],
                "follow_up": "What types of problems suit each architecture?"
            }
        ],
        "intermediate": [
            {
                "question": "What is the vanishing gradient problem and how can you address it?",
                "expected_concepts": ["deep networks", "ReLU", "residual connections", "LSTM"],
                "follow_up": "Why does ReLU help?"
            },
            {
                "question": "Explain dropout and batch normalization.",
                "expected_concepts": ["regularization", "overfitting", "internal covariate shift", "training"],
                "follow_up": "When would you use each?"
            }
        ],
        "advanced": [
            {
                "question": "Explain the attention mechanism and transformers.",
                "expected_concepts": ["self-attention", "query-key-value", "parallel processing", "BERT"],
                "follow_up": "Why are transformers better than RNNs for long sequences?"
            },
            {
                "question": "How do you approach transfer learning in deep learning?",
                "expected_concepts": ["pretrained", "fine-tuning", "feature extraction", "domain adaptation"],
                "follow_up": "When should you freeze layers vs fine-tune?"
            }
        ]
    },
    
    "Deployment": {
        "basic": [
            {
                "question": "What are the key considerations when deploying an ML model?",
                "expected_concepts": ["latency", "scalability", "monitoring", "versioning"],
                "follow_up": "How do you ensure reproducibility?"
            },
            {
                "question": "Explain the difference between batch and real-time inference.",
                "expected_concepts": ["throughput", "latency", "use cases", "architecture"],
                "follow_up": "When would you choose each approach?"
            }
        ],
        "intermediate": [
            {
                "question": "What challenges occur in model deployment that don't exist in development?",
                "expected_concepts": ["data drift", "scaling", "monitoring", "latency", "dependencies"],
                "follow_up": "How do you detect model degradation in production?"
            },
            {
                "question": "Explain containerization for ML models.",
                "expected_concepts": ["Docker", "reproducibility", "dependencies", "orchestration"],
                "follow_up": "How does Kubernetes help with ML deployment?"
            }
        ],
        "advanced": [
            {
                "question": "Describe an end-to-end MLOps pipeline.",
                "expected_concepts": ["CI/CD", "feature stores", "model registry", "monitoring", "retraining"],
                "follow_up": "What tools would you use?"
            },
            {
                "question": "How do you implement A/B testing for ML models?",
                "expected_concepts": ["traffic splitting", "metrics", "statistical significance", "rollout"],
                "follow_up": "How do you handle the exploration-exploitation tradeoff?"
            }
        ]
    },
    
    "NLP / CV": {
        "basic": [
            {
                "question": "What is tokenization and why is it important in NLP?",
                "expected_concepts": ["words", "subwords", "vocabulary", "preprocessing"],
                "follow_up": "What is the difference between word and subword tokenization?"
            },
            {
                "question": "Explain the basic image preprocessing steps for computer vision.",
                "expected_concepts": ["resizing", "normalization", "augmentation", "channels"],
                "follow_up": "Why is data augmentation important?"
            }
        ],
        "intermediate": [
            {
                "question": "How do word embeddings work and why are they useful?",
                "expected_concepts": ["vector representation", "semantic similarity", "Word2Vec", "context"],
                "follow_up": "What's the difference between static and contextual embeddings?"
            },
            {
                "question": "Explain object detection vs image classification.",
                "expected_concepts": ["localization", "bounding boxes", "multiple objects", "architectures"],
                "follow_up": "What are some common object detection architectures?"
            }
        ],
        "advanced": [
            {
                "question": "How do large language models (LLMs) work?",
                "expected_concepts": ["transformer", "attention", "pretraining", "fine-tuning", "prompting"],
                "follow_up": "What is prompt engineering?"
            },
            {
                "question": "Explain a real-world AI application you've built or would build.",
                "expected_concepts": ["problem definition", "architecture", "challenges", "impact"],
                "follow_up": "What were the biggest technical challenges?"
            }
        ]
    },
    
    "Model Optimization": {
        "basic": [
            {
                "question": "What is hyperparameter tuning?",
                "expected_concepts": ["parameters vs hyperparameters", "grid search", "random search"],
                "follow_up": "What hyperparameters are most important to tune?"
            },
            {
                "question": "How do you speed up model training?",
                "expected_concepts": ["batch size", "learning rate", "early stopping", "hardware"],
                "follow_up": "What trade-offs exist with larger batch sizes?"
            }
        ],
        "intermediate": [
            {
                "question": "Explain different hyperparameter optimization strategies.",
                "expected_concepts": ["grid search", "random search", "Bayesian optimization", "Hyperband"],
                "follow_up": "When would you use each approach?"
            },
            {
                "question": "How do you reduce model latency for inference?",
                "expected_concepts": ["quantization", "pruning", "batching", "model distillation"],
                "follow_up": "What's the accuracy-speed tradeoff?"
            }
        ],
        "advanced": [
            {
                "question": "Explain model distillation and when you'd use it.",
                "expected_concepts": ["teacher-student", "knowledge transfer", "compression", "deployment"],
                "follow_up": "How does it compare to quantization?"
            },
            {
                "question": "How do you optimize neural networks for edge deployment?",
                "expected_concepts": ["quantization", "architecture search", "hardware constraints", "latency"],
                "follow_up": "What frameworks help with edge optimization?"
            }
        ]
    },
    
    "Data Visualization": {
        "basic": [
            {
                "question": "What makes a good data visualization?",
                "expected_concepts": ["clarity", "appropriate chart", "labels", "audience"],
                "follow_up": "What are common visualization mistakes?"
            },
            {
                "question": "When would you use a bar chart vs a line chart?",
                "expected_concepts": ["categorical", "time series", "comparison", "trends"],
                "follow_up": "What about scatter plots?"
            }
        ],
        "intermediate": [
            {
                "question": "How do you visualize high-dimensional data?",
                "expected_concepts": ["PCA", "t-SNE", "UMAP", "parallel coordinates"],
                "follow_up": "What are the limitations of t-SNE?"
            },
            {
                "question": "What tools do you use for creating dashboards?",
                "expected_concepts": ["Tableau", "Power BI", "Plotly", "interactivity", "filters"],
                "follow_up": "How do you design for different audiences?"
            }
        ],
        "advanced": [
            {
                "question": "How do you design visualizations for real-time data?",
                "expected_concepts": ["streaming", "updates", "performance", "aggregation"],
                "follow_up": "What challenges arise with real-time dashboards?"
            }
        ]
    },
    
    "API Development": {
        "basic": [
            {
                "question": "What is a REST API and its key principles?",
                "expected_concepts": ["HTTP methods", "stateless", "resources", "endpoints"],
                "follow_up": "What's the difference between PUT and PATCH?"
            },
            {
                "question": "How do you handle API authentication?",
                "expected_concepts": ["API keys", "OAuth", "JWT", "security"],
                "follow_up": "When would you use JWT vs API keys?"
            }
        ],
        "intermediate": [
            {
                "question": "How do you design an API for an ML model?",
                "expected_concepts": ["input validation", "response format", "error handling", "versioning"],
                "follow_up": "How do you handle long-running predictions?"
            },
            {
                "question": "Explain API rate limiting and why it's important.",
                "expected_concepts": ["throttling", "protection", "fairness", "implementation"],
                "follow_up": "How would you implement it?"
            }
        ],
        "advanced": [
            {
                "question": "How do you design APIs for high availability?",
                "expected_concepts": ["load balancing", "caching", "redundancy", "monitoring"],
                "follow_up": "What's your approach to API versioning?"
            }
        ]
    },
    
    "Cloud Services": {
        "basic": [
            {
                "question": "What are the main cloud service models (IaaS, PaaS, SaaS)?",
                "expected_concepts": ["infrastructure", "platform", "software", "use cases"],
                "follow_up": "Which would you use for ML deployment?"
            }
        ],
        "intermediate": [
            {
                "question": "How do you deploy ML models on cloud platforms?",
                "expected_concepts": ["containers", "serverless", "managed services", "scaling"],
                "follow_up": "What are the cost considerations?"
            }
        ],
        "advanced": [
            {
                "question": "Design a scalable ML inference system on the cloud.",
                "expected_concepts": ["auto-scaling", "load balancing", "caching", "cost optimization"],
                "follow_up": "How do you handle traffic spikes?"
            }
        ]
    }
}

def get_questions_for_skill(skill: str, level: str = None) -> list:
    """Get questions for a specific skill and optional level"""
    if skill not in QUESTIONS:
        return []
    
    if level and level in QUESTIONS[skill]:
        return QUESTIONS[skill][level]
    
    # Return all questions for the skill
    all_questions = []
    for lvl in ["basic", "intermediate", "advanced"]:
        if lvl in QUESTIONS[skill]:
            all_questions.extend(QUESTIONS[skill][lvl])
    return all_questions

def get_adaptive_question(skill: str, current_level: str) -> dict:
    """Get next question based on current performance level"""
    questions = QUESTIONS.get(skill, {})
    
    level_map = {
        "beginner": "basic",
        "intermediate": "intermediate",
        "advanced": "advanced"
    }
    
    target_level = level_map.get(current_level, "basic")
    
    if target_level in questions and questions[target_level]:
        import random
        return random.choice(questions[target_level])
    
    return None