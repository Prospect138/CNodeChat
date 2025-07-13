# CNodeChat

LLM RAG chat for VS code for C/C++ projects.

## Modules

- exstension - source files for vs code exstension
- service - python backend, that act as http server and interact with ollama service AND script for build faiss database.

## Demo

![gif](https://github.com/Prospect138/CNodeChat/demo.gif)

## Diagram
```
1.                                         ---source_codes--->[ProjectParser]--VectorDB--->[RAG]

2. [VS Code extension] -user_query-> [Uvicorn server] -user_query-> [Ollama] -llm_query--------> [RAG Agent] --query_to_db--> [RAG]
                                                                            (optional tool call)          (mandatory tool call)

3. [VS Code extension] <-answer------[Uvicorn server] <-answer----- [Ollama] <-filtered_answer-- [RAG Agent] <-long_answer--- [RAG]
```

## How build:
- install conda enviroment from enviroment.yml
- start ollama service:
```
ollama serve
```
- build database with create_database.py, don't forget to choose dir 
- start oll_chat.py
- open air_chat_exstension/ with vs code, then start it

## TODO
- ~~Refactoring~~
- ~~clang AST based context enlargement~~
- more refactoring
- fix frontend
- containerization for service
- distribution for exstension
