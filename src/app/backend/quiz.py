from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
from typing import List, Optional
import openai  # or your preferred AI service
from fastapi.middleware.cors import CORSMiddleware

router = APIRouter()

# CORS middleware to allow requests from the Next.js frontend
router.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    content: str
    output_type: str

class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class Quiz(BaseModel):
    title: str
    description: str
    questions: List[QuizQuestion]

class QuizResponse(BaseModel):
    quiz: Quiz

@router.post("/api/generate_quiz")
async def generate_quiz(request: QuizRequest):
    try:
        # Use OpenAI or your preferred AI service to generate quiz questions
        prompt = f"""
        Generate a quiz based on the following content:
        {request.content}

        Create 5 multiple-choice questions. For each question:
        1. Write a clear question
        2. Provide 4 possible answers
        3. Indicate the correct answer
        4. Add a brief explanation of why it's correct

        Format the response as a JSON object with:
        - title: A title for the quiz
        - description: A brief description
        - questions: Array of questions with:
          - id: unique identifier
          - question: the question text
          - options: array of 4 possible answers
          - correctAnswer: the correct answer
          - explanation: why this is correct
        """

        # Call your AI service here
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are a quiz generator that creates educational assessments."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the AI response and format it
        quiz_data = response.choices[0].message.content
        # Process the quiz_data to match your Quiz model structure
        
        # For demonstration, here's a sample quiz structure
        quiz = Quiz(
            title="Understanding the Content",
            description="Test your knowledge of the key concepts",
            questions=[
                QuizQuestion(
                    id=str(uuid.uuid4()),
                    question="What is the main topic discussed in the content?",
                    options=[
                        "Option A",
                        "Option B",
                        "Option C",
                        "Option D"
                    ],
                    correct_answer="Option A",
                    explanation="This is the correct answer because..."
                ),
                # Add more questions...
            ]
        )

        return QuizResponse(quiz=quiz)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/process")
async def process_content(request: QuizRequest):
    # Your existing processing logic for other output types
    pass