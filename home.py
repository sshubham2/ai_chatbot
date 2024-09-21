import streamlit as st

st.markdown("""
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px;
    }
    .custom-title {  /* Fixed the class name here */
        font-family: 'Arial', sans-serif;
        font-size: 40px;
        font-weight: bold;
        color: #3366cc;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);  /* Fixed 'rgbs' to 'rgba' */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-container">
        <h1 class="custom-title">Hi!! Welcome</h1>  <!-- Fixed the closing quote for class -->
    </div>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .centered {
        text-align: center;
        padding: 10px;
    }
    .justified {
        text-align: justify;
        padding: 10px;
    }
    .big-font {
        font-size: 24px !important;
    }
    .highlight { 
        background-color: #272e29;
        padding: 20px;
        border-radius: 5px;
        max-width: 800px;
        margin: 0 auto;
    }
    </style>   
""", unsafe_allow_html=True)

st.markdown("""
    <div class="highlight centered"> 
        <p class="big-font">I'm your <i>Personal Assistant</i>.🎢</p>
        <p class="jstified">I’m your AI assistant, skilled in both programming and natural languages.
        Whether you need help coding, debugging, or translating technical concepts into clear language, I’m here to assist you.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .centered-header {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        font-weight: bold;
        color: #d0afe0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 20px;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1)
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='centered-header'>Available Modules</h1>", unsafe_allow_html=True)

st.subheader("Programming Language Expert 💻")
with st.expander("What can I do?"):
    st.markdown("""
        I can assist you with a wide range of programming tasks and technical inquiries, including:

            1. Writing Code:  
                Help you craft code in languages like Python, Java, C++, and JavaScript.

            2.Debugging:  
                Identify and fix errors or bugs to ensure your program runs smoothly.

            3. Optimizing Code:  
                Suggest improvements to enhance performance and efficiency.

            4. Explaining Concepts:  
                Clarify programming concepts, algorithms, and data structures in an understandable way.

            5. Providing Examples:  
                Offer practical code examples for specific tasks or functionalities.

            6. Technical Documentation:  
                Create well-structured documentation outlining design, functionality, and usage, with detailed in-code comments.

            7. Code Review:  
                Review your code to ensure it follows best practices and is easy to maintain.

            8. Assisting with Projects:  
                Help you plan, design, and implement software projects of any size.

            9. Learning Resources:  
                Recommend resources for learning programming languages, including tutorials and online courses.

            10. Answering Questions:
                Address any questions related to programming, software development, and computer engineering.

    Feel free to ask for assistance with any specific task or inquiry you have!
""",unsafe_allow_html=True)
    
st.subheader("Natural Language Expert 💬")
with st.expander("What can I do?"):
    st.markdown("""
        I can assist you in several language-related tasks, including:
        
            1. Multilingual Translation:
                I can translate text accurately between multiple languages, taking cultural nuances and idiomatic expressions into account.
                
            2. Content Summarization:
                I can summarize complex information into concise overviews, focusing on key points and main ideas.   
                       
            3. Content Rewriting:
                I can rewrite text according to your specifications, adapting tone, style,and structure while maintaining coherence.  
                         
            4. Professional Email Composition:
                I can help you compose clear and polite professional emails for various contexts.
                           
            5. Resume Writing:
                I can create well-structured resumes that highlight your skills and experiences, tailored to specific job roles.

        If you have a specific task in mind or need assistance with something else, feel free to let me know!
""")
st.write('Use the sidebar to navigate between modules and start exploring!')

# Title of the Streamlit application
st.header("IMPORTANT! Clearing Chat History")
st.write("""
    If your next search is not related to your previous search,
    **PLEASE clear chat and continue with your next question.**
""")
# Expander for benefits of clearing chat history
with st.expander("Why is this important?"):
    st.markdown("""
        ```
        1. Improved Contextual Understanding:
            When starting a new question or topic, clearing chat history helps the model focus on the current context without being influenced by irrelevant
            previous conversations.
       
        2. Enhanced Clarity:
            By removing unrelated history, users can ensure that their inquiries are clear and straightforward, which can lead to more accurate responses.
       
        3. Reduced Cognitive Load:
            Users don’t need to sift through past messages to formulate their new questions, making the interaction more efficient and user-friendly.
       
        4. Better Performance:
            Clearing chat history can prevent the model from generating responses that might be influenced by older, irrelevant context, leading to higher
            quality outputs.
       
        5. Focused Interactions:
            Each interaction can be treated as a standalone session, allowing users to explore new topics without the baggage of previous discussions.
    """)
    
st.subheader("Tips for Better LLM Prompting")
st.info("""
- **Be Specific**:
    Clearly articulate what you're asking. Instead of vague queries, provide context and specifics to guide the model's response.

- **Use Examples**:
    Providing examples of what you're looking for can help the model understand your expectations better.

- **Ask Direct Questions**:
    Formulate your prompts as direct questions to elicit more focused and concise answers.

- **Limit Complexity**:
    Avoid overly complex questions that may confuse the model. Break down complex inquiries into simpler parts.

- **Iterate on Responses**:
    If the first response isn’t satisfactory, refine and rephrase your prompt based on the output to improve clarity and relevance.

- **Give Feedback**:
    If applicable, inform the model of what parts of its response were useful or not, which can help it generate better responses in the future.

- **Stay On Topic**:
    For the best responses, keep your questions related to the current topic and avoid jumping between unrelated subjects.
""")

st.markdown('____')
st.write('''
         <p style='text-align: center;'>Created with 💕 using Streamlit</p>
''', unsafe_allow_html=True)


