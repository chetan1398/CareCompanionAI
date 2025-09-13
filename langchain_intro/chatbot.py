# langchain_intro/chatbot.py

import dotenv
import time
import random
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain import hub

# Custom tool function
from langchain_intro.tools import get_current_wait_time

# Load environment variables
dotenv.load_dotenv()

# ─────────────────────────────────────────────────────────────
# VECTORSTORE & RETRIEVER
REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)
reviews_retriever = reviews_vector_db.as_retriever(search_kwargs={"k": 10})

# ─────────────────────────────────────────────────────────────
# REVIEW PROMPT SETUP
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template="""Your job is to use patient
reviews to answer questions about their experience at a hospital.
Use the following context to answer questions. Be as detailed as possible,
but don't make up any information that's not from the context. If you don't
know an answer, say you don't know.

{context}
"""
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
)

review_prompt_template = ChatPromptTemplate.from_messages(
    [review_system_prompt, review_human_prompt]
)

# ─────────────────────────────────────────────────────────────
# REVIEW CHAIN
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
output_parser = StrOutputParser()

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | output_parser
)

# ─────────────────────────────────────────────────────────────
# AGENT TOOL SETUP
tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about visit details like billing, treatment, or wait times.
        Pass the entire question as input. Example: "What do patients say about the nurses?"
        """
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times at a specific hospital.
        Returns wait time in minutes. Input must only be the hospital name (e.g., "B", not "Hospital B").
        Example: "What is the wait time at hospital C?" → input should be "C"
        """
    ),
]

# ─────────────────────────────────────────────────────────────
# AGENT SETUP
agent_chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

hospital_agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

# ─────────────────────────────────────────────────────────────
# TEST FUNCTIONS

def test_chatbot_review_chain(question: str) -> str:
    """Test review_chain independently."""
    return review_chain.invoke(question)

def test_agent(question: str) -> dict:
    """Test agent decision-making."""
    return hospital_agent_executor.invoke({"input": question})

# ─────────────────────────────────────────────────────────────
# MAIN

if __name__ == "__main__":
    # test 1: review chain
    print(test_chatbot_review_chain(
        "Were there any complaints about communication with hospital staff?"
    ))

    # test 2: agent routing
    print(test_agent("What is the wait time at hospital B?"))
    print(test_agent("What do patients say about cleanliness?"))
