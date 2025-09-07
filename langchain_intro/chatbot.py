# langchain_intro/chatbot.py

import dotenv
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

# Load environment variables
dotenv.load_dotenv()

# Paths
REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)
reviews_retriever = reviews_vector_db.as_retriever(search_kwargs={"k": 10})

# Prompt templates
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

# Chain setup
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
output_parser = StrOutputParser()

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | output_parser
)

# Optional test function
def test_chatbot(question: str) -> str:
    return review_chain.invoke(question)

# Run test
if __name__ == "__main__":
    question = "Were there any complaints about communication with hospital staff?"
    print(test_chatbot(question))
