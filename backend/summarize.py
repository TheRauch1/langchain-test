import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableEach
from langchain_text_splitters import TokenTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from backend.model import Model

token_max = 1000


class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


class PromptLibrary:
    SUMMARIZE_MAP = """You are a helpful AI assistant that can summarize text. Please summarize the following text: {content}"""
    SUMMARIZE_REDUCE = """You are a helpful AI assistant that creates a final summary from the summaries beforehand with the following summaries: {content}"""


class SummarizationService:
    def __init__(self) -> None:
        m = Model()
        self.llm = m.model
        map_template = PromptLibrary.SUMMARIZE_MAP
        reduce_template = PromptLibrary.SUMMARIZE_REDUCE

        self.map_chain = (
            PromptTemplate(template=map_template) | self.llm | StrOutputParser()
        )
        self.reduce_chain = (
            PromptTemplate(template=reduce_template) | self.llm | StrOutputParser()
        )

    def extract_pdf_text(self, pdf_path: str) -> str:
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100
        )
        documents = PyPDFLoader(pdf_path).load_and_split(text_splitter)
        return documents

    def length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

    # Here we generate a summary, given a document
    async def generate_summary(self, state: SummaryState):
        print("generating summary")
        response = await self.map_chain.ainvoke(state["content"])
        print(response)
        return {"summaries": [response]}

    # Here we define the logic to map out over the documents
    # We will use this an edge in the graph
    def map_summaries(self, state: OverallState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node
        return [
            Send("generate_summary", {"content": content})
            for content in state["contents"]
        ]

    def collect_summaries(self, state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }

    async def collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], self.length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))

        return {"collapsed_summaries": results}

    def should_collapse(
        self,
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self.length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    # Here we will generate the final summary
    async def generate_final_summary(self, state: OverallState):
        response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        print(response)
        return {"final_summary": response}


sum_service = SummarizationService()

# Construct the graph
# Nodes:
graph = StateGraph(OverallState)
graph.add_node("generate_summary", sum_service.generate_summary)  # same as before
graph.add_node("collect_summaries", sum_service.collect_summaries)
graph.add_node("collapse_summaries", sum_service.collapse_summaries)
graph.add_node("generate_final_summary", sum_service.generate_final_summary)

# Edges:
graph.add_conditional_edges(START, sum_service.map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", sum_service.should_collapse)
graph.add_conditional_edges("collapse_summaries", sum_service.should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()
