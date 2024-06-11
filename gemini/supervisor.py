from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

members = ['Waiter', 'Manager']
options = ["FINISH"] + members

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members} and human. Given the following request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status."
    " Manager should only handle payment request. When finished or it is human turn"
    " respond with FINISH."
)

response_schema = [
    ResponseSchema(name="next", description="one of ['Waiter', 'Manager', 'FINISH']"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="""
                            Given the conversation above, who should act next?
                             Or should we FINISH?
                             Please use this format only: {format_instructions}.
                             Example: ```json\n{\n "next": "Waiter"}\n```
                             """)
    ]
).partial(options=str(options), members=", ".join(members), format_instructions=format_instructions)

supervisor_chain = (
        prompt | llm | output_parser
)
