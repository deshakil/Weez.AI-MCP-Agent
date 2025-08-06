import os
import json
from openai import AzureOpenAI
from datetime import datetime, timezone

TODAY_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version="2024-12-01-preview",
    azure_endpoint="https://weez-openai-resource.openai.azure.com/"
)

def parse_user_intent(raw_query: str) -> dict:
    """
    Parses complex user queries into structured intent including:
    action, platforms, file types, query text, limit, summary type, time range.
    Supports multiple intents per query.
    Enhanced for brain.py compatibility with needs_clarification field.
    """
    prompt = f"""
You are a smart assistant that parses user queries for a file AI agent.
Return JSON only. DO NOT include explanations.

Query: "{raw_query}"

IMPORTANT: Be proactive and avoid asking for clarification unless absolutely necessary.
Most queries with topics like "my project", "our meeting", "design docs" should be searchable.

Extract the following:

1. action: search / summarize / ask / upload / delete / download / unknown
   - Use "search" for finding documents
   - Use "summarize" for creating summaries  
   - Use "ask" for questions about content (RAG)

2. platforms: ["google_drive", "slack", "onedrive", "dropbox", "notion" ...]

3. file_types: ["pdf", "docx", "doc", "txt", "ppt", "pptx", "xls", "xlsx", ...]

4. query_text: the actual content-related intent of the user.
    - This is what the user is trying to find, understand, or describe **about the file**.
    - It must be independent of the action. Do not include "summarize", "search", "rag" etc.
    - Keep it purely about the file's subject matter, topic, section, or meaning.
    - For example: 
      - ❌ "Summarize the security part" → ✅ "security part" or "security-related information"
      - ❌ "Search for Kubernetes issues" → ✅ "Kubernetes issues" 
      - ❌ "Help me summarize my AI project" → ✅ "AI project"
    - Use the user's descriptive language if it reflects the content they care about (not the command).
    - Focus query_text only on the content topic or section — not on the user's command or task.

5. limit: number of files or chunks to consider (default: 5)

6. summary_type: short / general / detailed (default: general)

7. time_range: natural language expression like "last 3 days", "past week", "yesterday", "June 2024"

8. date_range: an object with ISO 8601 format dates like:
   {{
     "from": "2025-07-01", 
     "to": "2025-07-11"
   }}
   - If time_range is present, convert it into exact start & end date
   - Use current date as reference: Today is {TODAY_DATE}
   - If time_range is not present, return an empty string ("") and empty date_range

9. needs_clarification: boolean - ONLY true if the query is extremely vague like "help", "hello", or completely unclear

10. clarification_reason: string - explain why clarification is needed (only if needs_clarification is true)

11. intents: if multiple goals are present, break into separate intents with same structure

GUIDELINES FOR needs_clarification:
- Set to FALSE for queries mentioning topics like "my project", "AI project", "meeting notes", "design docs" etc.
- Set to FALSE if there's ANY searchable content mentioned
- Set to TRUE ONLY for extremely vague queries like: "help", "hello", "what can you do", queries under 3 characters
- Be VERY conservative - prefer searching over asking for clarification

Return strictly in this JSON format:
{{
  "action": "...",
  "platforms": ["..."],
  "file_types": ["..."], 
  "query_text": "...",
  "limit": 5,
  "summary_type": "general",
  "time_range": "last 3 days",
  "date_range": {{
    "from": "2025-07-08",
    "to": "2025-07-11"
  }},
  "needs_clarification": false,
  "clarification_reason": "",
  "intents": [
    {{
      "action": "...",
      "platforms": ["..."],
      "file_types": ["..."],
      "query_text": "...",
      "limit": 5,
      "summary_type": "short",
      "time_range": "past week",
      "date_range": {{
        "from": "2025-07-04", 
        "to": "2025-07-11"
      }},
      "needs_clarification": false,
      "clarification_reason": ""
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure needs_clarification exists and is conservative
        if "needs_clarification" not in result:
            result["needs_clarification"] = False
            
        if "clarification_reason" not in result:
            result["clarification_reason"] = ""
            
        # Double-check: if query_text has meaningful content, don't ask for clarification
        if result.get("query_text") and len(result["query_text"].strip()) > 2:
            result["needs_clarification"] = False
            result["clarification_reason"] = ""
            
        return result
        
    except Exception as e:
        print("Intent parsing error:", e)
        # Return conservative fallback that tries to search
        return {
            "action": "search",
            "platforms": [],
            "file_types": [],
            "query_text": raw_query,
            "limit": 5,
            "summary_type": "general", 
            "time_range": "",
            "date_range": {},
            "needs_clarification": False,  # Conservative: try searching first
            "clarification_reason": "",
            "intents": []
        }
