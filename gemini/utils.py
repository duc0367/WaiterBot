from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage


def create_agent_executor(llm: ChatGoogleGenerativeAI, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name="messages"),

    ])

    llm_with_tools = llm.bind_tools(tools)
    agent = ({
        "messages": lambda x: x["messages"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    } | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["output"], name=name)]}
