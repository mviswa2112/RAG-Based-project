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

"""docs = [
    "Diabetes Mellitus affects the body's ability to regulate blood sugar levels due to insulin issues.",
    "Type 1 Diabetes is an autoimmune condition where the immune system attacks the pancreas, reducing insulin production.",
    "Type 2 Diabetes typically develops when the body becomes resistant to insulin, requiring lifestyle changes to manage it.",
    "Symptoms of Diabetes include frequent urination, excessive thirst, blurry vision, and unexplained weight loss.",
    "Gestational Diabetes occurs during pregnancy and can increase the risk of developing Type 2 Diabetes later in life.",
    "Hypertension causes persistently high blood pressure, increasing the risk of heart disease.",
    "Asthma leads to airway inflammation and breathing difficulties, often triggered by allergens.",
    "Alzheimer's Disease progressively impairs memory and cognitive functions in older adults.",
    "Influenza spreads as a contagious virus that affects the respiratory system.",
    "HIV/AIDS weakens the immune system, leaving the body vulnerable to severe infections.",
    "Tuberculosis infects the lungs, causing chronic coughing, fever, and weight loss.",
    "Rheumatoid Arthritis is an autoimmune disorder causing joint pain and inflammation.",
    "Malaria, transmitted through mosquito bites, causes fever, chills, and severe illness.",
    "Cancer develops when abnormal cells grow uncontrollably and damage tissues and organs.",
    "Parkinson's Disease is a neurodegenerative disorder affecting movement and coordination.",
    "Epilepsy causes recurrent seizures due to abnormal brain activity.",
    "Chronic Kidney Disease leads to a gradual loss of kidney function over time.",
    "Psoriasis is an autoimmune skin condition causing itchy, scaly patches.",
    "Crohn's Disease causes inflammation of the digestive tract, leading to pain and malnutrition.",
    "Hepatitis B is a viral infection that inflames the liver and may cause long-term damage.",
    "Hepatitis C is a viral liver infection that may lead to cirrhosis or liver cancer.",
    "Eczema (Atopic Dermatitis) causes itchy, inflamed skin, often triggered by irritants.",
    "Sickle Cell Anemia is a genetic blood disorder causing misshapen red blood cells.",
    "Cystic Fibrosis is a genetic condition affecting the lungs and digestive system.",
    "Lupus (SLE) is an autoimmune disease that can damage multiple organs.",
    "Multiple Sclerosis (MS) occurs when the immune system attacks the nervous system.",
    "Dengue Fever is a mosquito-borne illness causing fever, rash, and severe pain.",
    "Zika Virus is a mosquito-transmitted virus causing birth defects in pregnant women.",
    "Polio is a viral infection that can cause paralysis and muscle weakness.",
    "Measles is a highly contagious viral disease causing rash and fever.",
    "Chickenpox is a viral infection causing itchy blisters and rash.",
    "Hemophilia is a genetic disorder where blood doesn’t clot properly.",
    "Thalassemia is a blood disorder causing reduced hemoglobin production.",
    "Osteoporosis weakens bones, increasing the risk of fractures.",
    "Obesity involves excess body fat, increasing the risk of chronic diseases.",
    "Coronary Artery Disease affects blood flow to the heart due to plaque buildup.",
    "Stroke occurs when blood supply to the brain is interrupted.",
    "Anemia is a condition caused by low red blood cell count or hemoglobin levels.",
    "Gout is a type of arthritis causing sudden joint pain due to uric acid buildup.",
    "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus.",
    "Pneumonia causes lung inflammation due to infection with bacteria, viruses, or fungi.",
    "Bronchitis involves inflammation of the bronchial tubes, causing cough and mucus buildup.",
    "Depression is a mental health condition causing persistent sadness and loss of interest.",
    "Anxiety Disorder involves excessive worry and fear.",
    "Schizophrenia is a mental disorder affecting thinking, emotions, and behavior.",
    "Bipolar Disorder causes extreme mood swings between mania and depression.",
    "Autism Spectrum Disorder (ASD) affects communication and social interaction.",
    "Attention Deficit Hyperactivity Disorder (ADHD) impairs focus and self-control.",
    "Migraine is a severe headache often accompanied by nausea and sensitivity to light.",
    "Huntington's Disease is a genetic disorder causing progressive brain degeneration.",
    "Lyme Disease is a tick-borne illness causing rash, fever, and joint pain.",
    "Cholera is an infection causing severe diarrhea and dehydration.",
    "Ebola Virus Disease causes severe fever, bleeding, and organ failure.",
    "Diphtheria is a bacterial infection affecting the throat and airways."
]"""


vectors = embedding_fn.encode_documents(docs)
#print(vectors[0])
#print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i]}
    for i in range(len(vectors))
]

res = client.insert(collection_name="demo_collection", data=data)

#print(res)

query_vectors = embedding_fn.encode_queries(["What is cancer in a single sentence?"])
#print(query_vectors[0])
result = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=5,  # number of returned entities
    output_fields=["text"],  # specifies fields to be returned
)

#print(result)
names = []
for inner_list in result:
    for item in inner_list:
        names.append(item.get("entity"))

sentences = []
for item in names:
    if 'text' in item:
        sentences.append(item['text'])

from transformers import AutoModelForCausalLM, AutoTokenizer

def query_gpt_neo(prompt, model_name="EleutherAI/gpt-neo-1.3B"):
    """
    Sends a prompt to GPT-Neo (free alternative to GPT-3) and retrieves a response.

    :param prompt: The text prompt to send
    :param model_name: The name of the Hugging Face model to load
    :return: Response text from the model
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize input prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate response
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # Decode response
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"

# Example usage
prompt = " ".join(sentences) + " [QUERY] " + "What is cancer in a single sentence?"
response = query_gpt_neo(prompt)

print("GPT-Neo Response:")
print(response)
