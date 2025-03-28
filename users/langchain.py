from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from users.models import ChatMemory
import json
from pydantic import ValidationError
import os
from dotenv import load_dotenv

load_dotenv()  

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")
openai_api_key = os.getenv("OPENAI_API_KEY")


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

examples:
------------------------
Current user message: "what are good german dishes"
output: "search the r/germany subreddit for best germany dishes"

Current user message: "what are the average salaries in canada"
output: "search the r/canada subreddit for average salary"
-------------------------
if the user message does not need reddit for response or is asking about a past response, output "NO NOT USE THE REDDIT TOOL!" followed by the user message
"""
)

query_chain = LLMChain(llm=query_llm, prompt=query_prompt)

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following Reddit tool:"""
suffix = """Begin!"

{chat_history}
Question: "{input}" Fetch about 10 reddit posts related to the question using the tool. respond in a detailed paragraph answer based on the reddit posts and give url when you reference information from it in a seperate paragraph titles references. ONLY GIVE VALID URLS DO NOT PUT PLACE HOLDERS.
also when using the reddit tool IT"S VERY IMPORTANT to follow this format for action:

  "action": "reddit_search",
  "action_input": 
    "query": "",
    "sort": "",
    "time_filter": "",
    "subreddit": "",
    "limit": ""

{agent_scratchpad}"""

tools = [
    RedditSearchRun(
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=client_id,
            reddit_client_secret=client_secret,
            reddit_user_agent=user_agent,
        ),
        handle_tool_error=True,
        handle_validation_error=True
    )
]

prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0, openai_api_key=openai_api_key)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=None, tools=tools,return_intermediate_steps=True,handle_parsing_errors=True
)



def get_agent_response(user_input, user):
    try:
        memory_obj, _ = ChatMemory.objects.get_or_create(user=user)
        chat_history = memory_obj.memory  


        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg_pair in chat_history:
            memory.chat_memory.add_user_message(msg_pair["user"])
            memory.chat_memory.add_ai_message(str(str(msg_pair["context"])+"\n\n"+str(msg_pair["bot"])))
        raw_history=''
        if chat_history:
            raw_history = "\n".join([f"User: {str(pair['user'])}\nBot: {str(str(str(msg_pair["context"])+"\n\n"+str(msg_pair["bot"])))}" for pair in chat_history[-6:]])
        
        try:
            reddit_query = query_chain.run({"chat_history":raw_history, "input":user_input})
        except Exception as e:
            print(e)
        print("reddit "+str(reddit_query))

        agent_chain.memory = memory

        try:
            result = agent_chain.invoke({"input":str(reddit_query)})
        except Exception as e:
            print(e)

        observation_store=""
        if result['intermediate_steps'] != []:
            for action, observation in result['intermediate_steps']:
                observation_store=observation
        response=result['output']
        try:
            if (isinstance(json.loads(str('{'+response)), dict) and "action" in response) or response=="Agent stopped due to iteration limit or time limit.":
                print("Agent did not finish reasoning properly.")
                response = "Hmm, I couldn't generate a full answer. Want to rephrase or try again?"
        except:
            None
        
        
        chat_history.append({
            "user": user_input,
            "bot": response,
            "context":observation_store
        })

        chat_history = chat_history[-6:]
        memory_obj.memory = chat_history
        memory_obj.save()
    
    
        
        return str(response)
    except Exception as e:
            print("Error "+e)
            return ""