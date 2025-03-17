from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
from requests.auth import HTTPBasicAuth
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Jira credentials from environment variables
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Set up authentication for Jira
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

app = FastAPI(title="Jira Ticket Rewriter API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class Project(BaseModel):
    id: str
    key: str
    name: str
    projectTypeKey: str

class Ticket(BaseModel):
    key: str
    summary: str
    description: Optional[str] = None

class RewrittenTicket(BaseModel):
    key: str
    original_title: str
    rewritten_title: str
    rewritten_description: str
    acceptance_criteria: List[str]
    technical_context: str

class UpdateTicketRequest(BaseModel):
    tickets: List[RewrittenTicket]

# Improved prompt creation - more focused and less likely to drift from original topic
def _create_prompt(ticket_data: dict) -> str:
    """Create a more focused prompt that maintains the original ticket's intent"""
    summary = ticket_data.get('summary', 'No Title')
    description = ticket_data.get('description', 'No Description')
    
    return f"""
    Transform the following Jira ticket into a well-structured user story with acceptance criteria,
    while preserving the original intent and core subject matter. Do NOT add any new features
    or completely change the ticket's purpose.
    
    Original Ticket Title: {summary}
    Original Description: {description}
    
    IMPORTANT: Stay focused on the EXACT topic described. Do not transform this into a generic
    performance or loading issue unless explicitly mentioned in the original ticket.
    
    Respond in this EXACT format with no deviations:
    
    USER STORY:
    As a [user role from context or 'user' if unclear], I want [capability directly related to the original ticket] so that [benefit directly related to the original ticket].
    
    ACCEPTANCE CRITERIA:
    1. [Specific measurable criterion directly related to the ticket]
    2. [Another specific criterion directly related to the ticket]
    3. [Additional criterion if needed]
    
    TECHNICAL CONTEXT:
    [Brief technical explanation about the issue, focusing only on what was mentioned or strongly implied in the original ticket]
    """

def _parse_ai_response(response_text: str, original_ticket: dict) -> dict:
    """Parse the AI response with fallback to original ticket information"""
    user_story = ""
    acceptance_criteria = []
    technical_context = ""
    
    # Split response by sections
    sections = response_text.split('\n\n')
    
    # Process each section
    current_section = None
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if "USER STORY:" in section:
            current_section = "user_story"
            user_story = section.replace("USER STORY:", "").strip()
        elif "ACCEPTANCE CRITERIA:" in section:
            current_section = "acceptance_criteria"
            criteria_text = section.replace("ACCEPTANCE CRITERIA:", "").strip()
            # Process criteria immediately
            for line in criteria_text.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    acceptance_criteria.append(line)
        elif "TECHNICAL CONTEXT:" in section:
            current_section = "technical_context"
            technical_context = section.replace("TECHNICAL CONTEXT:", "").strip()
        # Handle multi-line sections
        elif current_section == "user_story":
            user_story += " " + section
        elif current_section == "acceptance_criteria":
            for line in section.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    acceptance_criteria.append(line)
        elif current_section == "technical_context":
            technical_context += " " + section
    
    # Fallback to original title if no user story was parsed
    if not user_story:
        original_title = original_ticket.get("summary", "")
        user_story = f"As a user, I want {original_title.lower()} so that I can work efficiently."
    
    # Ensure we have at least some acceptance criteria
    if not acceptance_criteria:
        acceptance_criteria = ["1. The functionality should work as expected."]
    
    # Format the acceptance criteria properly
    formatted_criteria = []
    for criterion in acceptance_criteria:
        # Remove the number prefix if present
        formatted = re.sub(r'^\d+\.\s*', '', criterion)
        if formatted:
            formatted_criteria.append(formatted)
    
    # Fallback technical context if none provided
    if not technical_context:
        technical_context = f"This ticket addresses {original_ticket.get('summary', 'an issue')} which impacts user experience."
    
    return {
        "user_story": user_story,
        "acceptance_criteria": formatted_criteria,
        "technical_context": technical_context
    }

def _generate_contextual_fallback_response(ticket_data: dict) -> dict:
    """Generate a fallback response that's specific to the original ticket"""
    summary = ticket_data.get('summary', 'No Title')
    description = ticket_data.get('description', '')
    
    # Start with a basic user story focused on the original title
    user_story = f"As a user, I want {summary.lower()} so that I can work more efficiently."
    
    # Generate contextual acceptance criteria based on the ticket title
    acceptance_criteria = []
    
    # Check for UI-related keywords
    ui_keywords = ['ui', 'interface', 'screen', 'design', 'layout', 'page', 'view', 'form']
    is_ui_related = any(keyword in summary.lower() or keyword in description.lower() for keyword in ui_keywords)
    
    # Check for performance-related keywords
    performance_keywords = ['slow', 'lag', 'performance', 'speed', 'loading', 'timeout', 'quick', 'fast']
    is_performance_issue = any(keyword in summary.lower() or keyword in description.lower() for keyword in performance_keywords)
    
    # Check for "dynamic" keywords which seem important from your example
    dynamic_keywords = ['dynamic', 'static', 'interactive', 'responsive']
    is_dynamic_related = any(keyword in summary.lower() or keyword in description.lower() for keyword in dynamic_keywords)
    
    if is_ui_related and is_dynamic_related:
        acceptance_criteria = [
            "UI elements should update in response to user interactions without page reloads",
            "Data should be loaded asynchronously to maintain responsive user experience",
            "UI state should persist according to user interactions and selections",
            "All interactive elements should provide appropriate visual feedback when used"
        ]
        technical_context = f"The current UI implementation is static and requires improvements to make it dynamic and interactive. This will enhance user experience by providing immediate feedback and smoother interactions."
    elif is_performance_issue:
        acceptance_criteria = [
            "The functionality should respond within 1 second of user interaction",
            "Performance should be consistent across all supported browsers and devices",
            "Operations should not block the UI thread",
            "Appropriate loading indicators should be displayed for operations taking longer than 500ms"
        ]
        technical_context = f"The {summary} is experiencing performance issues that need to be addressed to improve user experience."
    else:
        # Generic criteria based on the ticket summary
        acceptance_criteria = [
            f"The {summary} functionality should work as expected",
            "Implementation should follow existing code style and patterns",
            "Changes should maintain or improve current performance metrics",
            "Solution should be tested across all supported browsers and devices"
        ]
        technical_context = f"This ticket addresses {summary} which will improve user experience and workflow efficiency."
    
    return {
        "user_story": user_story,
        "acceptance_criteria": acceptance_criteria,
        "technical_context": technical_context
    }

# API Routes
@app.get("/")
async def root():
    return {"message": "Jira Ticket Rewriter API"}

@app.get("/projects", response_model=List[Project])
async def get_projects():
    """Fetch all projects from Jira"""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/project"
    try:
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch projects: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch projects")

@app.get("/projects/{project_key}/issues", response_model=List[Ticket])
async def get_issues(project_key: str):
    """Fetch issues for a specific project"""
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search"
    payload = json.dumps({
        "jql": f"project = {project_key} ORDER BY created DESC",
        "fields": ["summary", "description"]
    })
    try:
        response = requests.post(url, data=payload, headers=headers, auth=auth)
        response.raise_for_status()
        issues = response.json().get("issues", [])
        
        # Transform the issues to match our model
        formatted_issues = []
        for issue in issues:
            fields = issue.get("fields", {})
            summary = fields.get('summary', 'No summary')
            description = fields.get('description', {})
            description_text = ""
            
            if description and isinstance(description, dict):
                if "content" in description and len(description["content"]) > 0:
                    first_paragraph = description["content"][0]
                    if "content" in first_paragraph and len(first_paragraph["content"]) > 0:
                        first_text_element = first_paragraph["content"][0]
                        if "text" in first_text_element:
                            description_text = first_text_element["text"]
            
            formatted_issues.append({
                "key": issue['key'],
                "summary": summary,
                "description": description_text
            })
        
        return formatted_issues
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch issues: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch issues")

@app.post("/rewrite-tickets", response_model=List[RewrittenTicket])
async def rewrite_tickets(tickets: List[Ticket]):
    """Rewrite selected tickets using Gemini AI with acceptance criteria"""
    rewritten_tickets = []
    
    for ticket in tickets:
        # Original ticket data for reference
        original_ticket = {
            "summary": ticket.summary,
            "description": ticket.description or ""
        }
        
        # Create more focused prompt
        prompt = _create_prompt(original_ticket)
        
        try:
            # Generate content with Gemini API
            response = model.generate_content(prompt)
            
            # Make sure we have text to process
            if not hasattr(response, 'text') or not response.text:
                logger.warning(f"Empty or invalid response from AI for ticket {ticket.key}")
                
                # Use contextual fallback response
                parsed_response = _generate_contextual_fallback_response(original_ticket)
            else:
                generated_text = response.text
                logger.info(f"AI response for ticket {ticket.key}: {generated_text[:100]}...")
                
                # Parse the AI response using improved parser that references original ticket
                parsed_response = _parse_ai_response(generated_text, original_ticket)
            
            # Extract components
            user_story = parsed_response["user_story"]
            acceptance_criteria = parsed_response["acceptance_criteria"]
            technical_context = parsed_response["technical_context"]
            
            # Make sure we have valid lists for acceptance criteria
            if not isinstance(acceptance_criteria, list):
                acceptance_criteria = [acceptance_criteria] if acceptance_criteria else []
            
            # Ensure numbered format for acceptance criteria
            numbered_criteria = []
            for i, criterion in enumerate(acceptance_criteria, 1):
                # Remove any existing numbers
                clean_criterion = re.sub(r'^\d+\.\s*', '', criterion)
                numbered_criteria.append(f"{i}. {clean_criterion}")
            
            # Format description from user story and technical context
            description = f"{user_story}\n\n{technical_context}"
            
            rewritten_tickets.append({
                "key": ticket.key,
                "original_title": ticket.summary,
                "rewritten_title": user_story,
                "rewritten_description": description,
                "acceptance_criteria": numbered_criteria,
                "technical_context": technical_context
            })
            
        except Exception as e:
            logger.error(f"Failed to generate user story for {ticket.key}: {str(e)}")
            # Use improved contextual fallback values
            parsed_response = _generate_contextual_fallback_response(original_ticket)
            
            user_story = parsed_response["user_story"]
            acceptance_criteria = parsed_response["acceptance_criteria"]
            technical_context = parsed_response["technical_context"]
            
            # Make sure we have valid lists for acceptance criteria
            if not isinstance(acceptance_criteria, list):
                acceptance_criteria = [acceptance_criteria] if acceptance_criteria else []
            
            # Ensure numbered format for acceptance criteria
            numbered_criteria = []
            for i, criterion in enumerate(acceptance_criteria, 1):
                # Remove any existing numbers
                clean_criterion = re.sub(r'^\d+\.\s*', '', criterion)
                numbered_criteria.append(f"{i}. {clean_criterion}")
            
            rewritten_tickets.append({
                "key": ticket.key,
                "original_title": ticket.summary,
                "rewritten_title": user_story,
                "rewritten_description": f"{user_story}\n\n{technical_context}",
                "acceptance_criteria": numbered_criteria,
                "technical_context": technical_context
            })
    
    if not rewritten_tickets:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate any user stories"
        )
    
    return rewritten_tickets

@app.put("/update-tickets")
async def update_tickets(request: UpdateTicketRequest):
    """Update tickets in Jira with rewritten content including acceptance criteria"""
    updated_tickets = []
    failed_tickets = []
    
    for ticket in request.tickets:
        try:
            # Format description to include acceptance criteria
            formatted_description = ticket.rewritten_description + "\n\n"
            formatted_description += "Acceptance Criteria:\n"
            for i, criterion in enumerate(ticket.acceptance_criteria, 1):
                formatted_description += f"{i}. {criterion}\n"
            
            url = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{ticket.key}"
            
            # Format the description for Jira's Atlassian Document Format
            adf_content = []
            
            # Add main description paragraph
            description_paras = ticket.rewritten_description.split('\n\n')
            for para in description_paras:
                if para.strip():
                    adf_content.append({
                        "type": "paragraph",
                        "content": [{
                            "type": "text",
                            "text": para.strip()
                        }]
                    })
            
            # Add acceptance criteria heading
            adf_content.append({
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{
                    "type": "text",
                    "text": "Acceptance Criteria"
                }]
            })
            
            # Add acceptance criteria as bullet list
            list_items = []
            for criterion in ticket.acceptance_criteria:
                list_items.append({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": [{
                            "type": "text",
                            "text": criterion
                        }]
                    }]
                })
            
            adf_content.append({
                "type": "bulletList",
                "content": list_items
            })
            
            payload = json.dumps({
                "fields": {
                    "summary": ticket.rewritten_title,
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": adf_content
                    }
                }
            })
            
            response = requests.put(url, data=payload, headers=headers, auth=auth)
            response.raise_for_status()
            updated_tickets.append(ticket.key)
        except Exception as e:
            logger.error(f"Failed to update ticket {ticket.key}: {str(e)}")
            failed_tickets.append({"key": ticket.key, "error": str(e)})
    
    return {
        "success": len(updated_tickets) > 0,
        "updated_tickets": updated_tickets,
        "failed_tickets": failed_tickets
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)