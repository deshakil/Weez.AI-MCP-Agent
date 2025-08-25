"""
FastAPI routes for conversation management using CosmosMemoryManager
These routes connect your React frontend to your Cosmos DB backend
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel
from urllib.parse import unquote

# Import your memory manager
from ai_layer.memory import CosmosMemoryManager, CosmosMemoryError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weez AI Conversation API", version="1.0.0")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080/chat", "http://localhost:5173"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory manager (you might want to use dependency injection in production)
memory_manager = CosmosMemoryManager()


# Pydantic models for request/response
class StoreConversationRequest(BaseModel):
    user_query: str
    agent_response: str
    timestamp: Optional[str] = None


class ConversationMessage(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    user_query: str
    agent_response: str
    timestamp: str


class ConversationSummary(BaseModel):
    conversation_id: str
    first_message_time: str
    last_message_time: str
    message_count: int


class SearchRequest(BaseModel):
    search_term: str
    conversation_id: Optional[str] = None
    limit: Optional[int] = 10


# Dependency to get memory manager
def get_memory_manager() -> CosmosMemoryManager:
    return memory_manager


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health_status = memory_manager.health_check()
        return {"status": "healthy", "cosmos_db": health_status}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/api/conversations/{user_id}", response_model=List[ConversationSummary])
async def get_user_conversations(
        user_id: str,
        limit: Optional[int] = None,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get all conversation summaries for a user.
    This corresponds to memory.get_user_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversations = memory.get_user_conversations(user_id, limit)

        # Convert to response format
        result = []
        for conv in conversations:
            result.append(ConversationSummary(
                conversation_id=conv["conversation_id"],
                first_message_time=conv["first_message_time"],
                last_message_time=conv["last_message_time"],
                message_count=conv["message_count"]
            ))

        logger.info(f"Retrieved {len(result)} conversations for user {user_id}")
        return result

    except CosmosMemoryError as e:
        logger.error(f"Failed to get conversations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations/{user_id}/{conversation_id}", response_model=List[ConversationMessage])
async def get_conversation_history(
        user_id: str,
        conversation_id: str,
        limit: Optional[int] = None,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get full conversation history for a specific conversation.
    This corresponds to memory.get_conversation_history()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        messages = memory.get_conversation_history(user_id, conversation_id, limit)

        # Convert to response format
        result = []
        for msg in messages:
            result.append(ConversationMessage(
                id=msg["id"],
                user_id=msg["user_id"],
                conversation_id=msg["conversation_id"],
                user_query=msg["user_query"],
                agent_response=msg["agent_response"],
                timestamp=msg["timestamp"]
            ))

        logger.info(f"Retrieved {len(result)} messages for conversation {conversation_id}")
        return result

    except CosmosMemoryError as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/{conversation_id}")
async def store_conversation(
        user_id: str,
        conversation_id: str,
        request: StoreConversationRequest,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Store a new conversation message.
    This corresponds to memory.store_conversation()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        memory.store_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_query=request.user_query,
            agent_response=request.agent_response,
            timestamp=request.timestamp
        )

        logger.info(f"Stored conversation for user {user_id} in conversation {conversation_id}")
        return {"status": "success", "message": "Conversation stored successfully"}

    except CosmosMemoryError as e:
        logger.error(f"Failed to store conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/conversations/{user_id}/{conversation_id}")
async def delete_conversation(
        user_id: str,
        conversation_id: str,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Delete an entire conversation thread.
    This corresponds to memory.delete_conversation()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        deleted_count = memory.delete_conversation(user_id, conversation_id)

        logger.info(f"Deleted {deleted_count} messages from conversation {conversation_id}")
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} messages",
            "deleted_count": deleted_count
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/search")
async def search_conversations(
        user_id: str,
        request: SearchRequest,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Search through conversation history.
    This corresponds to memory.search_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id

        results = memory.search_conversations(
            user_id=user_id,
            search_term=request.search_term,
            conversation_id=request.conversation_id,
            limit=request.limit
        )

        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(ConversationMessage(
                id=result["id"],
                user_id=result["user_id"],
                conversation_id=result["conversation_id"],
                user_query=result["user_query"],
                agent_response=result["agent_response"],
                timestamp=result["timestamp"]
            ))

        logger.info(f"Found {len(search_results)} search results for '{request.search_term}'")
        return {
            "results": search_results,
            "total_count": len(search_results),
            "search_term": request.search_term
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to search conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations/{user_id}/{conversation_id}/context")
async def get_conversation_context(
        user_id: str,
        conversation_id: str,
        context_limit: int = 5,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Get recent conversation context for AI prompts.
    This corresponds to memory.get_recent_context()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id
        conversation_id = unquote(conversation_id)  # Decode URL-encoded conversation_id

        context = memory.get_recent_context(user_id, conversation_id, context_limit)

        return {
            "conversation_id": conversation_id,
            "context": context,
            "context_limit": context_limit
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to get conversation context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/conversations/{user_id}/cleanup")
async def cleanup_old_conversations(
        user_id: str,
        days_old: int = 90,
        memory: CosmosMemoryManager = Depends(get_memory_manager)
):
    """
    Clean up old conversations.
    This corresponds to memory.cleanup_old_conversations()
    """
    try:
        user_id = unquote(user_id)  # Decode URL-encoded user_id

        deleted_count = memory.cleanup_old_conversations(user_id, days_old)

        logger.info(f"Cleaned up {deleted_count} old messages for user {user_id}")
        return {
            "status": "success",
            "message": f"Cleaned up {deleted_count} old messages",
            "deleted_count": deleted_count,
            "days_old": days_old
        }

    except CosmosMemoryError as e:
        logger.error(f"Failed to cleanup conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(CosmosMemoryError)
async def cosmos_memory_error_handler(request, exc: CosmosMemoryError):
    """Handle Cosmos DB memory errors."""
    logger.error(f"Cosmos Memory Error: {str(exc)}")
    return HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
