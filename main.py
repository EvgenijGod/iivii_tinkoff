from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.schema import Document

from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

parsed_data = []
data = pd.read_csv('tinkoff-terms/cards.csv')
for i in range(len(data)):
    row = data.iloc[i]
    tmp = list(row)
    parsed_data.append(
        Document(
            page_content=f"Сервис: {tmp[0]}, Условие: {tmp[1]}, Тариф: {tmp[2]}", 
            metadata={'source': 'tinkoff-terms/cards.csv'}
        )
    )

from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter

input_docs = ['tinkoff-terms/doc1.pdf', 'tinkoff-terms/doc2.pdf']

for doc_loader in [UnstructuredPDFLoader(path) for path in input_docs]:
    doc_parsed = doc_loader.load()
    
    tmp = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=1024,
        length_function=len,
    ).transform_documents(doc_parsed)
    #strings += list(map(lambda x: x.page_content, tmp))
    parsed_data += tmp
    tmp = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=0,
        length_function=len,
    ).transform_documents(doc_parsed)
    parsed_data += tmp
    #strings += list(map(lambda x: x.page_content, tmp))

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2", model_kwargs = {'device': 'cpu'})
from langchain.vectorstores import FAISS

db = FAISS.from_documents(parsed_data, embeddings)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate


prompt = PromptTemplate(
    template = (
        "Ты — помощник в банковских документах, твоя задача ответить пользователю на его вопрос, используя информацию из юридических документов.\n"
        "Отвечай строго по данным из документов, ничего лишнего не придумывай. Будь вежлива и отвечай только на русском языке.\n"
        "Твой ответ должен быть коротким: не более пяти предложений, а также не должен использовать английские слова\n\n"
        "Данные:\n{context}\n"
        "Вопрос: {question}\n"
        "Ответ:"
    ),
    input_variables=['context', 'question']
)
# Make sure the model path is correct for your system!
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.0,
    max_tokens=5000,
    n_ctx=2000,
    top_p=0.98,
    n_batch=1,
    callback_manager=callback_manager, 
    verbose=True,
)
qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    callbacks=[StreamingStdOutCallbackHandler()],
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def make_answer(question):
    answer = qa_with_sources_chain({'query': question})
    return answer['result'].split('Вопрос')[0].strip()

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    message: str
    user_id: str

@app.post("/message/")
def answer_question(input: Input):
    print("processing", input.user_id)
    answer = make_answer(input.message)
    print(answer)
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
