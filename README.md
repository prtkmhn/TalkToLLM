

# 🎙️ Voice-Enabled AI Conversation Manager 🤖

Welcome to the Voice-Enabled AI Conversation Manager! This project combines the power of speech recognition, language models, and text-to-speech to create an interactive conversational experience with an AI assistant.

## 🌟 Features

- 🗣️ Speech-to-Text: Transcribe your voice input into text using the `speech_recognition` library.
- 🧠 Language Model: Process the transcribed text using a powerful language model (Groq's `mixtral-8x7b-32768`) to generate intelligent responses.
- 🔊 Text-to-Speech: Convert the AI's responses into natural-sounding speech using the `pyttsx3` library.
- 💬 Conversation Management: Seamlessly manage the conversation flow, allowing you to engage in a continuous dialogue with the AI assistant.
- 📝 Customizable Prompts: Easily customize the system prompt by modifying the `system_prompt.txt` file to tailor the AI's behavior and personality.
- 🔄 Conversation Memory: Maintain context throughout the conversation using the `ConversationBufferMemory` from the `langchain` library.
- ⏰ Response Time Tracking: Monitor the response time of the language model to optimize performance.
- 👋 Graceful Exit: End the conversation by saying "goodbye" to the AI assistant.
- 🌐 Internet Search: Perform real-time searches on DuckDuckGo to retrieve relevant information for the conversation.
- 🔍 Retrieval Augmented Generation: Enhance the AI's responses by retrieving and incorporating relevant documents using the `langchain` library.
- 🔧 Configurable Parameters: Customize the maximum token length and temperature for the language model using command-line arguments.
- ⏸️ Interrupt Speech: Press the 'x' key to stop the AI's speech and start listening immediately.
- 🆕 Dynamic Search: Update the search query during the conversation by saying "new query" followed by the new query.
- 📊 URL References: Display the URLs used by the language model to generate its responses.

## 🚀 Getting Started

1. Clone the repository:

2. Install the required dependencies:

pip install -r requirements.txt

3. Set up your API keys:
- Create a `.env` file in the project root directory.
- Add your Groq API key in the following format: `GROQ_API_KEY=your-api-key`.
- Add your OpenAI API key in the following format: `OPENAI_API_KEY=your-api-key`.

4. Customize the system prompt (optional):
- Open the `system_prompt.txt` file.
- Modify the prompt to define the AI's behavior and personality.

5. Run the conversation manager:

python conversation_manager.py --query "your initial search query" [--max_tokens MAX_TOKENS] [--temperature TEMPERATURE] [--toggle_internet]

- `--query`: Specify the initial search query for DuckDuckGo (default: "OpenAI").
- `--max_tokens`: Set the maximum number of tokens for the language model's output (default: 256).
- `--temperature`: Set the temperature for the language model (default: 0.7).
- `--toggle_internet`: Enable real-time internet search mode.

6. Press Enter to start listening and speak your message.

7. The AI assistant will process your input and respond with both text and speech.

8. Continue the conversation by pressing Enter and speaking again.

9. To update the search query during the conversation, say "new query" followed by the new query.

10. Press the 'x' key to stop the AI's speech and start listening immediately.

11. To end the conversation, say "goodbye" to the AI assistant.

## 📋 Requirements

- Python 3.6 or above
- `langchain-groq`
- `speech_recognition`
- `pyttsx3`
- `python-dotenv`
- `duckduckgo_search`
- `pynput`

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Groq](https://groq.com/) for providing the powerful language model.
- [Langchain](https://github.com/hwchase17/langchain) for the conversation management, memory functionality, and retrieval augmented generation.
- [SpeechRecognition](https://github.com/Uberi/speech_recognition) for the speech-to-text capabilities.
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for the text-to-speech functionality.
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) for the real-time internet search functionality.

