import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import dotenv

dotenv.load_dotenv()

df = pd.read_csv('data/menu.csv')
df['combined_text'] = df.apply(lambda x: f'Name: {x['name']}, Type: {x['Type']}, Price: {x['Price']}', axis=1)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = FAISS.from_texts(df['combined_text'].tolist(), embedding=embeddings)

db.save_local('menu_db')
