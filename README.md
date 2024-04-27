
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

## 🚀 Getting Started

1. Clone the repository:

2. Install the required dependencies:

pip install -r requirements.txt

3. Set up your API keys:
- Create a `.env` file in the project root directory.
- Add your Groq API key in the following format: `GROQ_API_KEY=your-api-key`.

4. Customize the system prompt (optional):
- Open the `system_prompt.txt` file.
- Modify the prompt to define the AI's behavior and personality.

5. Run the conversation manager:

python conversation_manager.py

6. Press Enter to start listening and speak your message.

7. The AI assistant will process your input and respond with both text and speech.

8. Continue the conversation by pressing Enter and speaking again.

9. To end the conversation, say "goodbye" to the AI assistant.

## 📋 Requirements

- Python 3.6 or above
- `langchain-groq`
- `speech_recognition`
- `pyttsx3`
- `python-dotenv`

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [Groq](https://groq.com/) for providing the powerful language model.
- [Langchain](https://github.com/hwchase17/langchain) for the conversation management and memory functionality.
- [SpeechRecognition](https://github.com/Uberi/speech_recognition) for the speech-to-text capabilities.
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for the text-to-speech functionality.

Feel free to customize and enhance this README to best represent your project on GitHub! 😊
