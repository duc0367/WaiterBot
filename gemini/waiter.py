from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
import dotenv
from utils import create_agent_executor

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0, top_p=0.85)

# tools
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local('../menu_db', embeddings=embeddings, allow_dangerous_deserialization=True)

menu_search_tool = create_retriever_tool(
    db.as_retriever(),
    "menu_search",
    "Please use this tool to search information about restaurant menu",
)
tools = [menu_search_tool]

llm_math = load_tools(['llm-math'], llm=llm)

for tool in llm_math:
    tools.append(tool)

waiter_executor = create_agent_executor(llm, tools, 'You are a waiter in the restaurant.'
                                                    ' you can help make an order and calculate the price')

# out = waiter_executor.invoke({'messages': [HumanMessage(content='What is 4 times 3? Please use your available tool')]})
# print(out)
