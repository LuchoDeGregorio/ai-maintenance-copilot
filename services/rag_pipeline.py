from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_qa_chain(vectorstore):

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever()

    def qa_chain(question):

        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an industrial maintenance assistant.

        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = llm.invoke(prompt)

        return response.content

    return qa_chain