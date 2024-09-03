from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
vectorstore = Chroma.from_documents(
    documents=text_splitter.split_documents(docs),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

query = "agent memory"
retrived_docs = retriever.invoke(query)
relevant_docs = []
for doc in retrived_docs:
    parser = JsonOutputParser()
    retrieval_relevance_chain = (
        {
            "context": RunnableLambda(lambda _: [doc]) | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )
    relevance = retrieval_relevance_chain.invoke(f"""Is the context relevant to the user query?
true or false in relevance variable.
{parser.get_format_instructions()}
query: {query}
""")
    #print(relevance)
    if relevance['relevance']:
        relevant_docs.append(doc)

print(f'{len(relevant_docs)} relevant documents.')
if len(relevant_docs) == 0:
    print("No relevant documents.")
    exit()

answer_chain = (
    {"context": RunnableLambda(lambda _: relevant_docs) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = answer_chain.invoke(query)
print(f'Answer: {answer}')

parser = JsonOutputParser()
hallucination_chain = (
    {
        "context": RunnableLambda(lambda _: relevant_docs) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)
hallucination = hallucination_chain.invoke(f"""Does the answer have hallucination?
true or false in hallucination variable.
{parser.get_format_instructions()}
answer: {answer}
query: {query}
""")

print(f'Hallucination: {hallucination}')
