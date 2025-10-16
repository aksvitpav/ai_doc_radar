build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

pull:
	docker exec -it ollama ollama pull $(EMBEDDING_MODEL)
	docker exec -it ollama ollama pull $(CHAT_MODEL)