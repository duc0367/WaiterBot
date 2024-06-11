from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import functools
import dotenv

from utils import create_agent_executor, agent_node

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0, top_p=0.85)

# tools
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local('../payment_db', embeddings=embeddings, allow_dangerous_deserialization=True)

payment_search_tool = create_retriever_tool(
    db.as_retriever(),
    "payment_search",
    "Please use this tool to search information about payment",
)
tools = [payment_search_tool]

manager_executor = create_agent_executor(llm, tools, 'You are a manager who can give the information about payment')

manager_node = functools.partial(agent_node, agent=manager_executor, name="Manager")

# out = manager_executor.invoke({'messages': [HumanMessage(content='What is the restaurant bank account?')]})
# print(out)
