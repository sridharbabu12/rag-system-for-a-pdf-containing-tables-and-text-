from rich.console import Console
console = Console(width=110)
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.load import dumps, loads
import langchain
langchain.debug = True
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from pinecone_text.sparse import BM25Encoder
from langchain.schema.runnable import RunnablePassthrough
import os
from pinecone import Pinecone
import warnings
warnings.filterwarnings(action='ignore')

from langchain.embeddings import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


prompt = ChatPromptTemplate(input_variables=['original_query'],
                            messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
                                                                                        template="""You are given a contract or legal document. Based on the specified query, generate a comprehensive list of questions designed to extract both direct and inferred structured information from the document. Each question should:

Target specific key terms, clauses, and sections explicitly mentioned in the document.
Consider synonymous terms or related phrases that might provide additional context for the query.
Include questions that explore surrounding or logically connected details to uncover indirect or inferred information.
Be detailed enough to ensure the extracted answers are precise and structured."""
                                                            )),
                            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['original_query'], template='Generate multiple search queries related to: {question} \n OUTPUT (5 queries):'))])
     
llm=ChatOpenAI(openai_api_key=os.getenv('API_KEY'),model_name="gpt-3.5-turbo")

generate_queries = (
    prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

bm25_encoder= BM25Encoder().default() 
index_name="pdfdataextractionmultivector01"
pc = Pinecone(api_key="pcsk_6Zqppg_Jhh8LKSj4yZ5596ivyNrFYz45eXSJDQyPiuqRmFzJsfw4pEqHwFz36cszw8P7uS")
index=pc.Index(index_name)

retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index,top_k=5,alpha=0.1)

ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

template = """Answer the question based only on the following context:

Given below context with no conversation history:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

full_rag_fusion_chain = (
    {
        "context": ragfusion_chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

import asyncio

# ... existing code ...

async def main():
    result = await full_rag_fusion_chain.ainvoke({"question": "What Assets are associated with Options in the contract?"}, return_only_outputs=True)
    print(result)

# Run the main function
asyncio.run(main())