from pymilvus import MilvusClient,model

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")


if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)
embedding_fn = model.DefaultEmbeddingFunction()
def chunk_text_file(file_path, chunk_size):
   
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_chunk = []
        for line in file:
            if line.strip():  
                current_chunk.append(line.strip())
                if len(current_chunk) == chunk_size:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
    return chunks


file_path = "input.txt"  
chunk_size = 1  
chunks = chunk_text_file("dataset.txt", chunk_size)



#print(chunks)
docs=chunks


vectors = embedding_fn.encode_documents(docs)
#print(vectors[0])
#print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i]}
    for i in range(len(vectors))
]

res = client.insert(collection_name="demo_collection", data=data)

#print(res)

query_vectors = embedding_fn.encode_queries(["what is alzheimer?"])
threshold = 0.6

# Perform search in Milvus
result = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=5,  # fetch more results to apply threshold filtering
    output_fields=["text"],  # specifies fields to be returned
)
print(result)

# Extract entities and apply threshold filtering
names = []
for inner_list in result:
    for item in inner_list:
        similarity_score = item.get("score", 0)  # Get the similarity score
        if similarity_score < threshold:  # Apply threshold
            names.append(item.get("entity"))

# Retrieve top 5 chunks based on the filtered results
sentences = []
for item in names[:5]:  # Take the top 5 filtered chunks
    if 'text' in item:
        sentences.append(item['text'])
print (sentences)
from transformers import AutoModelForCausalLM, AutoTokenizer

def query_local_gpt(prompt, model_name="gpt2"):
    """
    Sends the input prompt to a local GPT model (e.g., GPT-2) and retrieves a response.

    :param prompt: The text prompt to send
    :param model_name: The Hugging Face model name to load
    :return: Response text from the model
    """
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize the input prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate a response
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # Decode and return the response
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"

# Example Usage
augmented_input = " ".join(sentences) + " [QUERY] " + "what is alzheimer?"
response = query_local_gpt(augmented_input, model_name="gpt2")  # You can replace "gpt2" with other models like "EleutherAI/gpt-neo-125M"

print("Local GPT Response:")
print(response)