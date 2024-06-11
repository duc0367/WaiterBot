from typing import Annotated, Sequence
import operator
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, BaseMessage

from waiter import waiter_node
from manager import manager_node
from supervisor import supervisor_chain, members


class RestaurantAgentState:
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


workflow = StateGraph(RestaurantAgentState)

workflow.add_node('Waiter', waiter_node)
workflow.add_node('Manager', manager_node)
workflow.add_node('Supervisor', supervisor_chain)

for member in members:
    workflow.add_edge(member, 'Supervisor')

conditional_map = {k: k for k in members}
conditional_map['FINISH'] = END

workflow.add_conditional_edges('Supervisor', lambda x: x['next'], conditional_map)

workflow.set_entry_point('Supervisor')

graph = workflow.compile()

config = {"recursion_limit": 20}

while True:
    user_input = input('User: ')
    if user_input.lower() == 'exit':
        break
    for s in graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=user_input)
                ]
            }, config=config
    ):
        if "__end__" not in s:
            print(s)
            print("----")
