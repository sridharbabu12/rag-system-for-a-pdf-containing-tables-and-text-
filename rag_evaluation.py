from rich.console import Console
console = Console(width=110)
import asyncio
import langchain
langchain.debug = True
from datasets import Dataset
from ragas import evaluate
import os
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings(action='ignore')
from multiquery_and_rankfusion import ragfusion_chain,full_rag_fusion_chain 

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Replace with actual key management

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

questions = [
    "What rights are granted under this agreement?",
    "What territories are covered by the contract?",
    "What types of channels are included in the agreement?",
    "What languages are licensed for the content?",
    "Are there any holdbacks mentioned in the agreement?",
    "What types of options are outlined in this agreement?",
]

ground_truths= ["Television, Near Video on Demand, Pay Per View, FVOD, SVOD, Subscription VOD, DTR, Pay per View, PPV, Theatrical, EST",
"exclusive Italy, Republic of San Marino and Vatican City; non-exclusive Italian speaking Malta, non-exclusive Italian speaking Principality of Monaco, non-exclusive Italian speaking Capodistria and non-exclusive Italian speaking Switzerland",
"paytv",
"Italian, English",
"The contract includes the following hold back periods: - During the Pay TV License Period, the Audiovisual works shall not have any Television Distribution by means of Pay TV in the Territory (other than the Non-Exclusive Territory), either integral or partial, in any language, by any person or entity other than RSG Media. - For Audiovisual works n. 4, 8 and 9, the Audiovisual works shall not be distributed in the Territory, in any language, embodied in any physical, pre-recorded video medium whatsoever, in combination with any newspaper or magazine, during certain time periods. - No person or entity other than RSG Media shall in any way promote and/or advertise in the Territory (other than the Non-Exclusive Territory) the future Pay TV Television Distribution of the Audiovisual works, during certain time periods.",
"The contract does not explicitly mention any additional options or rights that can be exercised by the Licensee. The contract covers the specific Audiovisual works and their respective license periods"             
]  

async def evaluate_rag_pipeline():
    contexts = []
    for query in questions:
        result = await ragfusion_chain.ainvoke({"question": query}, return_only_outputs=True)
        context = []
        for doc in result:
            context.append(doc[0].page_content)
        contexts.append(context)

    answer = []
    for query in questions:
        result = await full_rag_fusion_chain.ainvoke({"question": query}, return_only_outputs=True)
        answer.append(result)
    
    data = {
        "question": questions,
        "answer": answer,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    return dataset

# Run the async function
import asyncio
dataset = asyncio.run(evaluate_rag_pipeline())
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy]
)
print(results)

"""contexts=[]
for query in questions:
        result=await ragfusion_chain.ainvoke({"question":query},return_only_outputs=True)
        context=[]
        for doc in result:
                context.append(doc[0].page_content)
        contexts.append(context)

answer=[]
for query in questions:
    result=await full_rag_fusion_chain.ainvoke({"question": query},return_only_outputs=True)
    answer.append(result)
    
data = {
    "question": questions,
    "answer": answer,
    "contexts": contexts,
    "ground_truths": ground_truths
}

dataset=Dataset.from_dict(data)

results=evaluate(
    dataset=dataset,
    metrics=[faithfulness,
    answer_relevancy,
    ]
)
print(results)"""