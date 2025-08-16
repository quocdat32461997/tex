import logging
import os
from typing import Annotated

import dotenv
from langchain_core.tools import InjectedToolCallId, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

logger = logging.getLogger(__name__)
dotenv.load_dotenv()
print(os.getenv("GEMINI_API_KEY"))
# Create LLM class
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # replace with "gemini-2.0-flash"
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


# Handoffs
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)


# Simple agent tools
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


# Define agents
flight_assistant = create_react_agent(
    model=llm,
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a flight booking assistant",
    name="flight_assistant",
)
hotel_assistant = create_react_agent(
    model=llm,
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant",
)

# Define multi-agent graph
agent = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)

# Run the multi-agent graph
# for chunk in multi_agent_graph.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel",
#             }
#         ]
#     }
# ):
#     print(chunk)
#     print("\n")

"""
AIMessage(content="Here is the extracted information from the document, excluding the sensitive information:\n\n*   **Employer identification number (EIN):** 77-3995612\n*   **Employer's name, address, and ZIP code:** Amanda A. Scott, 4340 Fire Access Road, Charlotte, NC 28202\n*   **Employee's name and address:** Derek Cheater, 18 Yawkey Way, Boston, MA 55192\n*   **Control number:** R3D1\n*   **Wages, tips, other compensation:** 78,000.00\n*   **Social security wages:** 54,239.92\n*   **Medicare wages and tips:** 65,000.00\n*   **Federal income tax withheld:** 1,111.00\n*   **Social security tax withheld:** 1,111.00\n*   **Medicare tax withheld:** 1,111.00\n*   **Box 12a:** Code D, 1,234.00\n*   **Box 12b:** Code C, 123.45\n*   **State:** GA\n*   **Employer's state ID number:** 319-9921-4512\n*   **State wages, tips, etc.:** 78,000\n*   **State income tax:** 6,535\n*   **Local wages, tips, etc.:** 78,000\n*   **Local income tax:** 1,949\n*   **Locality name:** Atlanta\n*   **Year:** 2025", additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--b07c0684-6edb-4828-a9e3-f3689d0fc6a9-0', usage_metadata={'input_tokens': 1824, 'output_tokens': 387, 'total_tokens': 2211, 'input_token_details': {'cache_read': 0}})

{'messages': [HumanMessage(content='I want to agent to file tax form 1040.', additional_kwargs={}, response_metadata={}, id='8af45c8e-683d-4565-ae95-b263d7ee775b'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'transfer_to_form_1040', 'arguments': '{}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--69ae8a9a-dd43-4b1b-8cb3-912b800d1b9b-0', tool_calls=[{'name': 'transfer_to_form_1040', 'args': {}, 'id': 'c30e7c5d-cd73-4f5d-abc8-9293945c9f4c', 'type': 'tool_call'}], usage_metadata={'input_tokens': 52, 'output_tokens': 10, 'total_tokens': 62, 'input_token_details': {'cache_read': 0}}), ToolMessage(content='Successfully transferred to form_1040', name='transfer_to_form_1040', id='aedaac09-1add-4c4f-8f1f-67036e148d99', tool_call_id='c30e7c5d-cd73-4f5d-abc8-9293945c9f4c'), AIMessage(content="Here's the extracted information from the document, excluding the sensitive data:\n\n*   **Employer identification number (EIN):** 77-3995612\n*   **Employer's name, address, and ZIP code:**\n    *   Amanda A. Scott\n    *   4340 Fire Access Road\n    *   Charlotte, NC 28202\n*   **Control number:** R3D1\n*   **Employee's first name and initial:** Derek\n*   **Employee's last name:** Cheater\n*   **Employee's address and ZIP code:**\n    *   18 Yawkey Way\n    *   Boston, MA 55192\n*   **Wages, tips, other compensation:** 78,000.00\n*   **Federal income tax withheld:** 1,111.00\n*   **Social security wages:** 54,239.92\n*   **Social security tax withheld:** 1,111.00\n*   **Medicare wages and tips:** 65,000.00\n*   **Medicare tax withheld:** 1,111.00\n*   **12a Code:** D\n*   **12a Amount:** 1,234.00\n*   **12b Code:** C\n*   **12b Amount:** 123.45\n*   **State:** GA\n*   **Employer's state ID number:** 319-9921-4512\n*   **State wages, tips, etc.:** 78,000\n*   **State income tax:** 6,535\n*   **Local wages, tips, etc.:** 78,000\n*   **Local income tax:** 1,949\n*   **Locality name:** Atlanta\n*   **Form Year:** 2025", additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--95cb294d-6b63-47d9-b8b2-2b4b99b6c1fe-0', usage_metadata={'input_tokens': 1824, 'output_tokens': 438, 'total_tokens': 2262, 'input_token_details': {'cache_read': 0}})], 'forms': [], 'statments': []}
"""
