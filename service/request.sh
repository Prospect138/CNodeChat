curl -X POST "http://127.0.0.1:21666/chat" \
     -H "Content-Type: application/json" \
     -d '{
           "request": "Привет! Можешь объяснить, как...",
           "history": []
         }'
