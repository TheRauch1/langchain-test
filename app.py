import asyncio

from dotenv import load_dotenv

from backend.model import Model
from backend.summarize import SummarizationService, app

load_dotenv()

service = SummarizationService()


async def run():
    async for step in app.astream(
        {
            "contents": [
                doc.page_content
                for doc in service.extract_pdf_text("data/somatosensory.pdf")
            ]
        },
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))


print(asyncio.run(run()))
