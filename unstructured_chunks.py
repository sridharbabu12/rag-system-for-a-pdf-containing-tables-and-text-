from unstructured.partition.api import partition_via_api
from unstructured.chunking.title import chunk_by_title
from pinecone import Pinecone,ServerlessSpec
import re
from pinecone_text.sparse import BM25Encoder

pc = Pinecone(api_key="pcsk_6Zqppg_Jhh8LKSj4yZ5596ivyNrFYz45eXSJDQyPiuqRmFzJsfw4pEqHwFz36cszw8P7uS")


filename = "D:\AI and Analytics project\data\input\RSG_Sample_iCon_Contract_2.pdf"
api_key = "tIqzjUPPZNQzHfbOGyCqlUHeJdJbde"

elements = partition_via_api(
  filename=filename, 
  api_key=api_key, 
  strategy="hi_res",
  ocr_language=['eng'],
  extract_image_block_types=["Table"]
  
  
)

chunk_elements = chunk_by_title(elements)

table_chunks=[chunk for chunk in chunk_elements if chunk.to_dict()['type']=='Table']

index_name="pdfdataextractionmultivector01"

pc.create_index(
    name=index_name,
    dimension=1024, # Replace with your model dimensions
    metric="dotproduct", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )  
)

index=pc.Index(index_name)

my_documents=[chunk.text for chunk in chunk_elements ]

def clean_text(text):
    # Step 1: Remove excess newlines and extra spaces
    cleaned_text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with one

    # Step 2: Remove duplicate company information (e.g., repeating addresses, VAT numbers)
    cleaned_text = re.sub(r'(Registered offices:.*?VAT Registration number: IT04619241005)(.*?)(Registered offices:.*?VAT Registration number: IT04619241005)', r'\1', cleaned_text)

    # Step 3: Normalize quotation marks to standard double quotes (optional)
    cleaned_text = cleaned_text.replace('“', '"').replace('”', '"')

    # Step 4: Standardize punctuation (remove spaces before punctuation marks)
    cleaned_text = re.sub(r'(\s)([.,!?])', r'\2', cleaned_text)  # Remove space before punctuation marks
    
    cleaned_text=cleaned_text.replace(']',"")
    # Step 5: Ensure no trailing or leading spaces are present
    return cleaned_text.strip()

documents_for_embedding1=[]
for text in my_documents:
    if text.strip():
        documents_for_embedding1.append(clean_text(text))
     
bm25 = BM25Encoder().default()   
document=[]
data_for_insertion=[]
for i in range(len(documents_for_embedding1)):
    embedding=pc.inference.embed(
    model="multilingual-e5-large",
    inputs=documents_for_embedding1[i],
    parameters={"input_type": "passage", "truncate": "END"})
    sparse_vector=bm25.encode_documents(documents_for_embedding1[i])
    document.append(documents_for_embedding1[i])
    
    data_for_insertion.append({
        "id": chunk_elements[i].id,  # Unique ID from element_id
        "values": embedding[0]['values'],
        "sparse_values":sparse_vector,# Convert embedding to list format
        "metadata": {
            "context": documents_for_embedding1[i]
        }
    })
    
upsert_response = index.upsert(vectors=data_for_insertion)