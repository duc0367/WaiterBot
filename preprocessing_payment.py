from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import dotenv

dotenv.load_dotenv()

loader = CSVLoader(file_path='data/payment.csv', source_column='Question')
data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = FAISS.from_documents(data, embedding=embeddings)

db.save_local('payment_db')
