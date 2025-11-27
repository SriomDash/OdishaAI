from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import ItineraryForm
import logging


try:
    from chakadola_graph import chakadola_chain
    logger = logging.getLogger(__name__)
    logger.info("✅ Chakadola chain imported successfully")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import chakadola_chain: {e}")
    chakadola_chain = None

app = FastAPI(
    title="Chakadola Yatra API",
    description="AI-powered Odisha travel itinerary planner",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Chakadola Yatra API",
        "status": "running",
        "graph_available": chakadola_chain is not None
    }

@app.post("/itinerary")
async def create_itinerary(data: ItineraryForm):
    """
    Generate a complete travel itinerary with:
    - AI-selected or user-specified places
    - RAG-enriched place information
    - Weather predictions
    - Cost breakdowns
    - Optimized route mapping
    - Day-wise detailed plans
    """
    
    if chakadola_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Itinerary generation service is unavailable. Please check server logs."
        )
    
    try:
        logger.info(f"📥 Received itinerary request for {data.duration} days")
        
        # Convert Pydantic model to dict for the graph
        request_data = data.dict()
        
        # Invoke the LangGraph workflow
        result = chakadola_chain.invoke({"request": request_data})
        
        # Check for errors in the workflow
        if result.get("error"):
            logger.error(f"❌ Workflow error: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Workflow error: {result['error']}"
            )
        
        # Extract the final itinerary
        final_itinerary = result.get("final_itinerary", {})
        
        if not final_itinerary:
            logger.error("❌ No itinerary generated")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate itinerary"
            )
        
        logger.info(f"✅ Successfully generated itinerary with {len(final_itinerary.get('days', []))} days")
        
        # Return structured response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Itinerary generated successfully",
                "data": {
                    "itinerary": final_itinerary,
                    "map_data": result.get("map_info", {}),
                    "selected_places": result.get("selected_places", []),
                    "rag_results": result.get("rag_results", []),
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test if the graph can be invoked
        test_result = chakadola_chain is not None
        
        return {
            "status": "healthy" if test_result else "degraded",
            "components": {
                "api": "running",
                "graph": "available" if test_result else "unavailable",
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Invalid input data",
            "error": str(exc)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred",
            "error": str(exc)
        }
    )