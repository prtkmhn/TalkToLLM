import os
import time
import subprocess
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
import speech_recognition as sr
import pyttsx3
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
load_dotenv()

class LanguageModelProcessor:
    def __init__(self,url):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.embed_model = "text-embedding-ada-002"
        self.embeddings = OpenAIEmbeddings(model=self.embed_model, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        self.loader = WebBaseLoader(url)
        self.docs = self.loader.load()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.documents = self.text_splitter.split_documents(self.docs)
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        
        self.prompt = ChatPromptTemplate.from_template('''
            Answer question based on provided context.
            <context>
            {context}
            </context>
            
            Question: {input}
        ''')

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retriever = self.vector_store.as_retriever()
        self.retrieve_chain = create_retrieval_chain(self.retriever, self.document_chain)

        self.conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.conversation_prompt,
            memory=self.memory
        )



    def process(self, text):
        self.memory.chat_memory.add_user_message(text)

        start_time = time.time()
        response = self.retrieve_chain.invoke({"input": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['answer'])

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['answer']}")
        return response['answer']

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Human: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        return ""

class ConversationManager:
    def __init__(self,url):
        self.llm = LanguageModelProcessor(url)
        self.tts = TextToSpeech()
        self.stt = SpeechToText()

    def main(self):
        while True:
            input("Press Enter to start listening...")
            transcription_response = self.stt.listen()

            if "goodbye" in transcription_response.lower():
                break

            if transcription_response.strip():
                llm_response = self.llm.process(transcription_response)
                self.tts.speak(llm_response)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Talk to your LLM')
    parser.add_argument('--url', type=str, default="https://en.wikipedia.org/wiki/OpenAI", help='URL of the website to load')
    args = parser.parse_args()

    manager = ConversationManager(args.url)
    manager.main()