import os
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin



api_key = os.getenv("GEMINI_API_KEY")

index_storage_dir = "index_storage"


Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=api_key
)

Settings.llm = Gemini(api_key=api_key, temperature=0)

docs=SimpleDirectoryReader("kbdata").load_data()
index=VectorStoreIndex.from_documents(docs)


if not os.path.exists(index_storage_dir):
    print("Creating new index...")
    documents=SimpleDirectoryReader("kbdata").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_storage_dir)
else:
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=index_storage_dir)
    index = load_index_from_storage(storage_context)
    
print("Index loaded successfully.")

def query_kb(index_i,query_str):
    """Search Knowledge Base or KB"""

    query_engine=index_i.as_query_engine()
    
    response= query_engine.query(query_str)
    responseAsText = str(response).strip()
    
    return responseAsText


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return "KB Search API "

@app.route("/kb", methods=['GET','POST'])
@cross_origin()
def get_bot_response():
  
    user_prompt = request.args.get('prompt')
    print("Received prompt:", user_prompt)
    user_query = f'Get the resolution or Background(if provided) and KB number for the Issue or Problem: {user_prompt}' 
    result = query_kb(index,user_query)
    print("Response: ",result)
    output_dict = {"response": result}
    output_json = json.dumps(output_dict)
    return output_json

if __name__ == "__main__":
    app.run()

