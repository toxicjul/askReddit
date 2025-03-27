from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from users.models import ChatMemory

client_id='HO2fzJOdcIPsVyfgJT6sAA'
client_secret='o167oBShCm5OnmjnBSrArdZGMXohMQ'
user_agent='API'
openai_api_key = "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"


#-----------------------

query_llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini-2024-07-18", openai_api_key=openai_api_key)

query_prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
You are a helpful assistant that generates focused Reddit search queries based on a conversation.

Chat history:
{chat_history}

Current user message:
{input}

What would be a clean, specific Reddit search query to use for this input?
Reply only with the query, nothing else.
"""
)

query_chain = LLMChain(llm=query_llm, prompt=query_prompt)





#-----------------------
#template = """This is a conversation between a human and a bot:
#
#{chat_history}
#
#Write a summary of the conversation for {input}:
#"""

#prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)
#memory = ConversationBufferMemory(memory_key="chat_history")

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: "{input} , fetch at least 10 reddit posts related to the question. Always look up reddit. respond in a detailed paragraph answer based on the reddit posts and give url when you reference information from it in a seperate paragraph titles references."
{agent_scratchpad}"""

tools = [
    RedditSearchRun(
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=client_id,
            reddit_client_secret=client_secret,
            reddit_user_agent=user_agent,
        )
    )
]

prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=None, tools=tools,return_intermediate_steps=True
)



def get_agent_response(user_input, user):
    # Get or create ChatMemory instance for the user
    memory_obj, _ = ChatMemory.objects.get_or_create(user=user)
    chat_history = memory_obj.memory  # Should be a list of {"user": "...", "bot": "..."} dicts

    # Load chat history into LangChain memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg_pair in chat_history:
        memory.chat_memory.add_user_message(msg_pair["user"])
        memory.chat_memory.add_ai_message(msg_pair["bot"])

    raw_history = "\n".join([f"User: {pair['user']}\nBot: {pair['bot']}" for pair in chat_history[-5:]])
    reddit_query = query_chain.run(chat_history=raw_history, input=user_input).strip()
    print(reddit_query)
    # Inject memory into agent_chain for this request
    agent_chain.memory = memory

    # Run the agent with context
    result = agent_chain.invoke(input=reddit_query)

    observation_store=""
    #print(result)
    if result['intermediate_steps'] != []:
        for action, observation in result['intermediate_steps']:
            observation_store=observation
    answer= str(observation_store+"\n\n"+result['output'])
    response=result['output']
    # Append new messages
    chat_history.append({
        "user": user_input,
        "bot": answer
    })

    chat_history = chat_history[-6:]
    memory_obj.memory = chat_history
    memory_obj.save()
    

    return str(response)