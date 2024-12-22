import pymupdf
from pprint import pprint
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
from unstructured_chunks import index,document
pc = Pinecone(api_key="pcsk_6Zqppg_Jhh8LKSj4yZ5596ivyNrFYz45eXSJDQyPiuqRmFzJsfw4pEqHwFz36cszw8P7uS")





# Open the document
doc = pymupdf.open(r"D:\AI and Analytics project\data\input\RSG_Sample_iCon_Contract_2.pdf")

# Initialize a list to store all extracted tables
all_tables = []

# Loop through all pages in the document
for page_num in range(len(doc)):  # Iterate over all pages
    page = doc[page_num]  # Get each page
    tabs = page.find_tables()  # Locate and extract any tables on the page
    
    print(f"{len(tabs.tables)} table(s) found on page {page_num + 1}")  # Display number of tables found on the current page

    if tabs.tables:  # If there are tables on the page
        for table in tabs.tables:
            extracted_table = table.extract()  # Extract the content of the table
            all_tables.append(extracted_table)  # Add the extracted table to the list

# After the loop, all_tables contains all the extracted tables
print(f"\nTotal tables extracted: {len(all_tables)}")

import re

def has_no_column_names_based_on_integers(table):
    """
    Determines if a table has no column names based on the first row
    containing strings starting with integers.
    """
    first_row = table[0]
    for cell in first_row:
        if isinstance(cell, str) and re.match(r"^\d", cell):
            return True
    return False

def assign_column_names(tables):
    """
    Assign column names to tables with no column names based on the
    column names of the previous table.
    """
    updated_tables = []
    previous_table_columns = None  # To store column names of the previous table

    for table in tables:
        if has_no_column_names_based_on_integers(table):
            if previous_table_columns:
                # Add column names from the previous table
                updated_table = [previous_table_columns] + table
                updated_tables.append(updated_table)
            else:
                # If no previous table column names, leave it as-is or handle differently
                updated_tables.append(table)
        else:
            # Treat the first row as column names
            previous_table_columns = table[0]
            updated_tables.append(table)
    
    return updated_tables

# Process tables
updated_tables = assign_column_names(all_tables)

# Display the updated tables
for i, table in enumerate(updated_tables):
    print(f"Table {i + 1}:")
    for row in table:
        print(row)
    print()

def extract_table_as_dicts(all_tables):
    all_extracted_data = []
    
    for table in all_tables:
        columns = table[0]  # First row as columns
        table_data = []
        
        # Iterate over the remaining rows
        for row in table[1:]:
            row_dict = {columns[i]: row[i] for i in range(len(columns))}
            table_data.append(row_dict)
        
        all_extracted_data.append(table_data)
    
    return all_extracted_data

table_data_for_embedding=extract_table_as_dicts(updated_tables)

def prepare_data_for_embedding(dict_tables):
    """
    Converts each row of the table (dict) into a concatenated string for embeddings.
    """
    table_strings = []
    for table in dict_tables:
        table_as_strings = [
            ", ".join([f"{key}: {value}" for key, value in row.items()])
            for row in table
        ]
        table_strings.append(table_as_strings)
    return table_strings


table_strings = prepare_data_for_embedding(table_data_for_embedding)

flattened_list = [item for sublist in table_strings for item in sublist]
bm25 = BM25Encoder().default()  

data_for_insertion1=[]
for i in range(len(flattened_list)):
    text=flattened_list[i]
    text=text.replace("\n"," ")
    embedding=pc.inference.embed(
    model="multilingual-e5-large",
    inputs=text,
    parameters={"input_type": "passage", "truncate": "END"})
    sparse_vector=bm25.encode_documents(text)
    document.append(text)
    
    data_for_insertion1.append({
        "id":str(i),  # Unique ID from element_id
        "values": embedding[0]['values'],
        "sparse_values":sparse_vector,# Convert embedding to list format
        "metadata": {
            "context": text
        }
    }
    )
    
bm25.fit(document)
bm25.dump("bm25_values.json")
bm25_encoder=BM25Encoder().load("bm25_values.json")
upsert_response = index.upsert(vectors=data_for_insertion1)