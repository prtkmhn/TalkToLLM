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
import json
from langchain.chains import LLMChain
import speech_recognition as sr
import pyttsx3
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from duckduckgo_search import DDGS
from pynput import keyboard
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

load_dotenv()


class LanguageModelProcessor:
    def __init__(self, urls, max_tokens=256, temperature=0.7):
        self.llm = ChatGroq(temperature=temperature, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"), max_tokens=max_tokens)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()

        self.embed_model = "text-embedding-3-small"
        self.embeddings = OpenAIEmbeddings(model=self.embed_model, openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                self.documents.extend(text_splitter.split_documents(docs))
            except Exception as e:
                print(f"Error loading URL: {url}")
                print(f"Error message: {str(e)}")

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

    def stop(self):
        self.engine.stop()



class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        self.engine.stop()

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
    def __init__(self, query, max_tokens=256, temperature=0.7, toggle_internet=False):
        self.query = query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.toggle_internet = toggle_internet
        self.update_llm()

    def update_llm(self):
        if self.toggle_internet:
            results = DDGS().text(self.query, max_results=4)
            urls = [result['href'] for result in results]
        else:
            results = DDGS().text(self.query, max_results=2)
            urls = [result['href'] for result in results[:1]]

        self.llm = LanguageModelProcessor(urls, max_tokens=self.max_tokens, temperature=self.temperature)
        self.urls = urls

    def on_press(self, key):
        if key == keyboard.KeyCode.from_char('x'):
            self.stop_listening = True
            self.tts.stop()
            time.sleep(0.5)  # Add a small delay to ensure the speech is fully stopped
            self.stt.listen()

    def main(self):
        self.tts = TextToSpeech()
        self.stt = SpeechToText()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        while True:
            input("Press Enter to start listening...")
            transcription_response = self.stt.listen()

            if "goodbye" in transcription_response.lower():
                break

            if transcription_response.lower().startswith("new query"):
                self.query = transcription_response[10:].strip()
                self.update_llm()
                print(f"New search query: {self.query}")
                continue

            if self.toggle_internet:
                self.query = transcription_response
                self.update_llm()

            if transcription_response.strip():
                llm_response = self.llm.process(transcription_response)
                self.tts.speak(llm_response)
                print(f"URLs used by LLM: {', '.join(self.urls)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Talk to your LLM')
    parser.add_argument('--query', type=str, default="ImportantNewsToday", help='Search query for DuckDuckGo')
    parser.add_argument('--max_tokens', type=int, default=70, help='Maximum number of tokens for LLM output')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for LLM')
    parser.add_argument('--toggle_internet', action='store_true', help='Toggle internet search mode')
    args = parser.parse_args()

    manager = ConversationManager(args.query, max_tokens=args.max_tokens, temperature=args.temperature, toggle_internet=args.toggle_internet)
    manager.main()


