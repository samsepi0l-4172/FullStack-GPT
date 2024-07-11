import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.5,
    model="gpt-4o",
    openai_api_key="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",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    This is a multiple choice question.

    Question: What is the color of the ocean?
    type: multiple_choice
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital of Georgia?
    type: multiple_choice
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    type: multiple_choice
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    type: multiple_choice
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    -------------------------------------------------------------

    This is a subjective question.

    Question: What is the color of the sky?
    type: short_answer
    Answer: Blue

    Question: Who wrote 'To Kill a Mockingbird'?
    type: short_answer
    Answer: Harper Lee
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.

    You format exam questions into JSON format.
    Answers with (o) are the correct ones. Short answer questions will have a single "correct" field.

    Example Input:

    This is a multiple choice question.

    Question: What is the color of the ocean?
    type: multiple_choice
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital of Georgia?
    type: multiple_choice
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    type: multiple_choice
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    type: multiple_choice
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    -------------------------------------------------------------

    This is a subjective question.

    Question: What is the color of the sky?
    type: short_answer
    Answer: Blue

    Question: Who wrote 'To Kill a Mockingbird'?
    type: short_answer
    Answer: Harper Lee


    Example Output:

    ```json
    { "questions": [
            {
                "question": "What is the color of the ocean?",
                "type": "multiple_choice",
                "answers": [
                        {
                            "answer": "Red",
                            "correct": false
                        },
                        {
                            "answer": "Yellow",
                            "correct": false
                        },
                        {
                            "answer": "Green",
                            "correct": false
                        },
                        {
                            "answer": "Blue",
                            "correct": true
                        }
                ]
            },
            {
                "question": "What is the capital of Georgia?",
                "type": "multiple_choice",
                "answers": [
                        {
                            "answer": "Baku",
                            "correct": false
                        },
                        {
                            "answer": "Tbilisi",
                            "correct": true
                        },
                        {
                            "answer": "Manila",
                            "correct": false
                        },
                        {
                            "answer": "Beirut",
                            "correct": false
                        }
                ]
            },
            {
                "question": "When was Avatar released?",
                "type": "multiple_choice",
                "answers": [
                        {
                            "answer": "2007",
                            "correct": false
                        },
                        {
                            "answer": "2001",
                            "correct": false
                        },
                        {
                            "answer": "2009",
                            "correct": true
                        },
                        {
                            "answer": "1998",
                            "correct": false
                        }
                ]
            },
            {
                "question": "Who was Julius Caesar?",
                "type": "multiple_choice",
                "answers": [
                        {
                            "answer": "A Roman Emperor",
                            "correct": true
                        },
                        {
                            "answer": "Painter",
                            "correct": false
                        },
                        {
                            "answer": "Actor",
                            "correct": false
                        },
                        {
                            "answer": "Model",
                            "correct": false
                        }
                ]
            },
            {
                "question": "What is the color of the sky?",
                "type": "short_answer",
                "correct": "Blue"
            },
            {
                "question": "Who wrote 'To Kill a Mockingbird'?",
                "type": "short_answer",
                "correct": "Harper Lee"
            }
        ]
    }
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.file_cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)

    # Add debugging output
    st.write("Debugging: Displaying the response structure")
    st.json(response)  # Assuming response is a dictionary

    if "questions" not in response:
        st.error("Error: The response does not contain 'questions'")
    else:
        with st.form("questions_form"):
            for i, question in enumerate(response["questions"]):
                # Adding debug statements for the question structure
                st.write(f"Debugging: Question {i+1} structure")
                st.json(question)

                # Check and set default type if needed
                if "type" not in question:
                    st.error(
                        f"Error: Question {i+1} does not contain 'type'. Defaulting to 'short_answer'"
                    )
                    question["type"] = "short_answer"  # Default to 'short_answer'

                # Ensure the question is valid
                if "question" not in question:
                    st.error(
                        f"Error: Question {i+1} does not contain a 'question' key."
                    )
                    continue  # Skip this iteration if there's no 'question' key

                # Display the question
                st.write(question["question"])

                # Handle multiple choice questions
                if question["type"] == "multiple_choice":
                    if "answers" not in question:
                        st.error(
                            f"Error: Question {i+1} of type 'multiple_choice' does not contain 'answers'."
                        )
                        continue  # Skip this iteration if no 'answers' present

                    answers = question.get("answers", [])
                    answer_texts = [answer.get("answer", "N/A") for answer in answers]

                    value_multiple_choice = st.radio(
                        "Select an option.",
                        answer_texts,
                        index=None,
                        key=f"radio_{i}",  # Unique key for each radio button
                    )

                    # Check the correct answer
                    def is_correct(answer):
                        return answer.get("correct") == True

                    if any(
                        answer.get("answer") == value_multiple_choice
                        and is_correct(answer)
                        for answer in answers
                    ):
                        st.success("Correct!")
                    elif value_multiple_choice:
                        st.error("Wrong!")

                # Handle short answer questions
                elif question["type"] == "short_answer":
                    correct_answer = question.get("correct", None)
                    if not correct_answer:
                        st.error(
                            f"Error: Question {i+1} of type 'short_answer' does not contain 'correct'."
                        )
                        continue  # Skip this iteration if no 'correct' present

                    value_subjective = st.text_input(
                        "Enter your answer.",
                        key=f"text_input_{i}",  # Unique key for each text input
                    )

                    if value_subjective == correct_answer:
                        st.success("Correct!")
                    elif value_subjective:
                        st.error("Wrong!")

            # Add the submit button at the end of the form
            submit_button = st.form_submit_button("Submit All")

            # Handle the form submission
            if submit_button:
                st.success("Quiz submitted successfully!")
