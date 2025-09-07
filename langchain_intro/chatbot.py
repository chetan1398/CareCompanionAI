# langchain_intro/chatbot.py

import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

# Load .env environment variables
dotenv.load_dotenv()

# Step 1: Define the prompt templates
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

# Step 2: Set up chain
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
output_parser = StrOutputParser()
review_chain = review_prompt_template | chat_model | output_parser

# Step 3: Function to test chatbot
def test_chatbot(context: str, question: str) -> str:
    return review_chain.invoke({"context": context, "question": question})

# Step 4: Run test
if __name__ == "__main__":
    sample_context = "The nurses were kind and attentive throughout my stay."
    sample_question = "Were there any comments about the nursing staff?"
    print(test_chatbot(sample_context, sample_question))
