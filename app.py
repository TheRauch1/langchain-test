from dotenv import load_dotenv

from backend.summarize import SummarizationService

load_dotenv()

service = SummarizationService()
from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
router = APIRouter()


@router.get("/summarize")
async def summarize():
    return StreamingResponse(
        service.generate_response("data/1.pdf"),
        media_type="text/event-stream",
    )


app.include_router(router)
