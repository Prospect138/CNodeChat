
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from ollama import ChatResponse, chat, Message, Options
import uvicorn



def use_retriever(model_query: str) -> str:
    response_from_retriever = retriever.invoke(model_query)
    result = ''


    for doc in response_from_retriever:
        result += doc.metadata['file_path'] + doc.metadata['full_code']
        logger.debug(f"called funсs: {doc.metadata['called_functions']}\n{doc.metadata['full_code']}")
    return result


def run_retriever_agent(model_query: str) -> str:
    use_retriever_tool = {
        'type': 'function',
        'function': {
            'name': 'use_retriever',
            'description': 'Use RAG retriever to get extend context with snippets of code base.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'model_query': {'type': 'string', 'description': 'Query for RAG retriever'},
                },
                'required': ['model_query']
            },
        },
    }

    available_functions = {
        'use_retriever': use_retriever
    }

    messages = []
    system_prompt = Message(role='system', content='Ты — RAG агент. Твоя задача — использовать инструмент use_retriever, чтобы помочь Чат Агенту.\
                                                    Ищи 5 раз, можешь ходить по связанным функциям (например, из поля called_functions), менять поисковый запрос. \
                                                    Когда соберешь полный контекст, он вернeтся Чат Агенту. Удачи.')
    messages.append(system_prompt)
    messages.append(Message(role='user', content=str(model_query)))
    max_tool_calls = 5
    tool_call_count = 0
    final_response = ''

    while tool_call_count < max_tool_calls:
        response: ChatResponse = chat(
            model_name,
            messages=messages,
            tools=[use_retriever_tool],
            options=Options(num_ctx=131072, temperature=0.2, num_predict=8192)
        )
        tool_call_count += 1
        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_call_message = f"🧠 Вызываю инструмент: {tool.function.name} с аргументами {tool.function.arguments}"
                messages.append(Message(role='assistant', content=str(tool_call_message)))
                logger.debug(f"tool call: {tool.function.arguments}")

                if function_to_call := available_functions.get(tool.function.name):
                    output = function_to_call(**tool.function.arguments)
                    messages.append(Message(role='tool', content=str(output)))
                    logger.debug(f"tool answer: {tool_call_message}")
                    final_response += output

    #final_response: ChatResponse = chat(
    #    model_name,
    #    messages=messages
    #)

    return final_response



# --- FastAPI integration ---
app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    request: str
    history: list = []  # List of {role, content}

class ChatResponseModel(BaseModel):
    response: str
    history: list
    
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AirChat")

def run_chat(user_message: str, history: list):
    run_retriever_agent_tool = {
    run_retriever_agent_tool = {
        'type': 'function',
        'function': {
            'name': 'run_retriever_agent',
            'description': 'Use RAG retriever Agent to get extend context with snippets of code base.',
            'name': 'run_retriever_agent',
            'description': 'Use RAG retriever Agent to get extend context with snippets of code base.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'model_query': {'type': 'string', 'description': 'Query for RAG retriever agent'},
                    'model_query': {'type': 'string', 'description': 'Query for RAG retriever agent'},
                },
                'required': ['model_query']
            },
        },
    }

    available_functions = {
        'run_retriever_agent': run_retriever_agent
        'run_retriever_agent': run_retriever_agent
    }

    messages = []
    system_prompt = Message(role='system', content='Ты — помощник программиста. Получи вопрос пользователя и автоматически передай его RAG-агенту для поиска информации в C/C++ кодовой базе. \
        "                                           RAG-агент может находить строчки C++ кода по: file name, namespace. \
                                                    Не описывай процесс поиска. Найди информацию через RAG и сразу дай понятный ответ на русском языке.')
    messages.append(system_prompt)
    
    for msg in history:
        messages.append(Message(role=msg['role'], content=msg['content']))

    response: ChatResponse = chat(
        model=model_name,
        messages=messages,
        tools=[run_retriever_agent_tool],
        options=Options(num_ctx=131072, temperature=1.3)
    )

    if response.message.tool_calls:
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            tool_call_message = f"🧠 Вызываю инструмент: {tool.function.name} с аргументами {tool.function.arguments}"
            messages.append(Message(role='assistant', content=str(tool_call_message)))
            tool_call_message = f"🧠 Вызываю инструмент: {tool.function.name} с аргументами {tool.function.arguments}"
            messages.append(Message(role='assistant', content=str(tool_call_message)))
            if function_to_call := available_functions.get(tool.function.name):
                output = function_to_call(**tool.function.arguments)
                messages.append(Message(role='tool', content=str(output)))

        response: ChatResponse = chat(
            model_name,
            messages=messages
        )
                messages.append(Message(role='tool', content=str(output)))

        response: ChatResponse = chat(
            model_name,
            messages=messages
        )

    final_answer = response.message.content
    new_history = [{'role': m.role, 'content': m.content} for m in messages[1:]]

    return final_answer, new_history

@app.post("/chat", response_model=ChatResponseModel)
def chat_endpoint(req: ChatRequest):
    logger.debug("Получен новый POST-запрос /chat")
    user_message = req.request
    history = req.history if req.history else []
    response, new_history = run_chat(user_message, history)
    return ChatResponseModel(response=response, history=new_history)


db_path = "/home/prospect/oia5g2/air_chat/service/code_vector_db"
embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 10, 'similarity_score_threshold': 0.8})
model_name = 'devstral:24b'
retriever = vectorstore.as_retriever(search_kwargs={'k': 10, 'similarity_score_threshold': 0.8})
model_name = 'devstral:24b'

if __name__ == "__main__":
    uvicorn.run("oll_chat:app", host="0.0.0.0", port=21666, reload=True)
