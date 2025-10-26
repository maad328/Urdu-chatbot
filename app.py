"""
Urdu Chatbot - Streamlit App
A simple, clean chatbot interface for your Urdu chatbot
"""
import streamlit as st
import sys
import os

# Add backend to path
sys.path.append('backend')

# Configure Streamlit page
st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean interface
st.markdown("""
<style>
    /* Hide Streamlit default styling */
    .stMarkdown {
        margin-bottom: 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load the chatbot model (cached for performance)"""
    try:
        # Create a new instance to avoid singleton issues
        from backend.inference import ChatbotInference
        chatbot_instance = ChatbotInference()
        chatbot_instance.initialize()
        return chatbot_instance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Error details: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def initialize_session_state():
    """Initialize chat history in session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot_loaded" not in st.session_state:
        st.session_state.chatbot_loaded = False


def main():
    """Main Streamlit app"""
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description - centered
    st.markdown("""
    <h1 style="text-align: center; font-size: 2.5rem; font-weight: bold; 
               margin-bottom: 0.5rem;">Urdu Chatbot</h1>
    <p style="text-align: center; font-size: 1.1rem; color: #666; 
              margin-bottom: 2rem;">اسلام وعلیکم! میں آپ کی مدد کر سکتا ہوں</p>
    """, unsafe_allow_html=True)
    
    # Load the chatbot (with caching)
    with st.spinner("Loading chatbot model..."):
        chatbot_instance = load_chatbot()
    
    if chatbot_instance is None:
        st.error("❌ Failed to load chatbot model. Please check your configuration.")
        st.stop()
    
    if not st.session_state.chatbot_loaded:
        st.session_state.chatbot_loaded = True
    
    # Sidebar with info
    with st.sidebar:
        st.header(" About")
        st.info("""
        This is an Urdu chatbot powered by a fine-tuned
        encoder-decoder transformer model.
        
        Ask questions in Urdu and get responses!
        """)
        
        st.divider()
        
        st.header(" Examples")
        example_questions = [
            "آپ کیسے ہیں؟",
            "آپ کا نام کیا ہے؟",
            "آج موسم کیسا ہے؟",
            "کیا آپ مجھے مدد کر سکتے ہیں؟",
            "آپ کہاں سے آئے ہیں؟"
        ]
        
        for example in example_questions:
            if st.button(example, key=example, use_container_width=True):
                st.session_state.example_clicked = example
    
    # Display chat messages with custom styling
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        # Create columns for positioning
        col1, col2, col3 = st.columns([1, 7, 1])
        
        if role == "user":
            # User messages on the right
            with col2:
                st.markdown(f"""
                <div style="text-align: right; margin-bottom: 10px;">
                    <div style="display: inline-block; background-color: #007bff; color: white; 
                                border-radius: 18px; padding: 12px 16px; max-width: 75%; 
                                box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Bot messages on the left
            with col2:
                st.markdown(f"""
                <div style="text-align: left; margin-bottom: 10px;">
                    <div style="display: inline-block; background-color: #f1f3f5; color: #212529; 
                                border-radius: 18px; padding: 12px 16px; max-width: 75%; 
                                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Handle example click
    if 'example_clicked' in st.session_state:
        question = st.session_state.example_clicked
        del st.session_state.example_clicked
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get bot response
        with st.spinner("Thinking..."):
            response = chatbot_instance.generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("اپنا سوال پوچھیں... (Ask your question in Urdu)"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        with st.spinner("Thinking..."):
            response = chatbot_instance.generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Clear chat button
    


if __name__ == "__main__":
    main()

