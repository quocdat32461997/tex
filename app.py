import mlflow
from langchain_core.messages import convert_to_messages

from tex.agents.tex_agent import TexAgent

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("auto-tracing-demo")
mlflow.langchain.autolog()


# asyncio.run(main())
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


agent_obj = TexAgent()
agent = agent_obj.get()

if __name__ == "__main__":
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I want to agent to file tax form 1040.",
                }
            ],
        },
    )
# for chunk in agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "I want to agent to file tax form 1040.",
#             },
#             # {
#             #     "role": "user",
#             #     "content": "I want to agent to file tax form 1040.",
#             # },
#         ],
#     },
#     subgraphs=True,
# ):
#     print("\n", "*******", chunk)
