
import streamlit as st
from datetime import datetime

def streamlit_run(chatbot_instance):
    """Main app function that accepts a chatbot instance"""
    # Page config
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="centered"
    )

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'bot' not in st.session_state:
        # Bot will be injected from main.py
        st.session_state.bot = chatbot_instance

    # Custom CSS for better styling
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .message-content {
            margin: 0.5rem 0;
        }
        .message-time {
            font-size: 0.75rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🤖 AI Chatbot")
    with col2:
        if st.button("🔄 Reset", help="Start a new conversation"):
            st.session_state.bot.reset_memory()
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            timestamp = message["timestamp"]

            if role == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div style="font-weight: bold; color: #1976d2;">You</div>
                        <div class="message-content">{content}</div>
                        <div class="message-time">{timestamp}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div style="font-weight: bold; color: #424242;">Bot</div>
                        <div class="message-content">{content}</div>
                        <div class="message-time">{timestamp}</div>
                    </div>
                """, unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Ask me anything..."
        )

    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)

    # Handle message sending
    if send_button and user_input:
        # Get current timestamp
        timestamp = datetime.now().strftime("%H:%M")

        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Get bot response
        with st.spinner("Thinking..."):
            bot_response = st.session_state.bot.chat(user_input)

        # Add bot message to chat
        st.session_state.messages.append({
            "role": "bot",
            "content": bot_response,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Rerun to update the chat
        st.rerun()

    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This is an AI-powered chatbot built with:
        - 🤖 LangGraph for conversation flow
        - 💬 Streamlit for the UI
        - 🧠 Memory persistence across messages

        **Tips:**
        - Click the Reset button to start a new conversation
        - All messages are saved in your current session
        """)

        st.markdown("---")
        st.markdown(f"**Messages in conversation:** {len(st.session_state.messages)}")

        if st.session_state.messages:
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.rerun()

