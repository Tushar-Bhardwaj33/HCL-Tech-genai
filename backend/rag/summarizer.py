from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

__all__ = [
    "SummarizerAndImageDescriber",
]

class SummarizerAndImageDescriber:
    def __init__(self):
        # Retrieve API keys from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")

        if not groq_api_key or not nvidia_api_key:
            raise ValueError("Missing required API keys in the environment variables.")

        # Set environment variables
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key

        # Text and table summarization setup
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text, keep in mind it should reflect every semantic meaning it has.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        """
        self.text_prompt = ChatPromptTemplate.from_template(prompt_text)

        self.text_model = ChatGroq(
            temperature=0.5,
            model="llama-3.1-8b-instant",
            api_key=groq_api_key
        )

        self.summarize_chain = {"element": lambda x: x} | self.text_prompt | self.text_model | StrOutputParser()

        # Image description setup
        image_prompt_template = """Describe the image in detail."""
        self.image_prompt = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": image_prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ])
        ])

        self.image_model = init_chat_model(
            "meta/llama-3.2-90b-vision-instruct",
            model_provider="nvidia",
            temperature=1
        )

        self.image_chain = self.image_prompt | self.image_model | StrOutputParser()

    def summarize_texts(self, texts: list, prompt_text: str = None):
        if prompt_text:
            self.text_prompt = ChatPromptTemplate.from_template(prompt_text)
            self.summarize_chain = {"element": lambda x: x} | self.text_prompt | self.text_model | StrOutputParser()
        else:
            self.summarize_chain = {"element": lambda x: x} | self.text_prompt | self.text_model | StrOutputParser()
        return self.summarize_chain.batch(texts, {"max_concurrency": 3})

    def summarize_tables(self, tables: list, prompt_text: str = None):
        if prompt_text:
            self.text_prompt = ChatPromptTemplate.from_template(prompt_text)
            self.summarize_chain = {"element": lambda x: x} | self.text_prompt | self.text_model | StrOutputParser()
        else:
            self.summarize_chain = {"element": lambda x: x} | self.text_prompt | self.text_model | StrOutputParser()
        tables_html = [table.metadata.text_as_html for table in tables]
        return self.summarize_chain.batch(tables_html, {"max_concurrency": 3})

    def describe_images(self, images: list, prompt_text: str = None):
        if prompt_text:
            self.image_prompt = ChatPromptTemplate.from_template(prompt_text)
            self.image_chain = self.image_prompt | self.image_model | StrOutputParser()
        else:
            self.image_chain = self.image_prompt | self.image_model | StrOutputParser()
        return self.image_chain.batch(images)