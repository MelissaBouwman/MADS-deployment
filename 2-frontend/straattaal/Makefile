# Makefile voor Straattaal Docker Deployment
# Automatiseert: training, wheel build, Docker build, en deployment

# ===== CONFIGURATIE =====
WHEEL_DIR = dist
MODEL_DIR = artefacts
ASSETS_DIR = assets
IMAGE_NAME = straattaal
CONTAINER_NAME = straattaal-container
PORT = 80

# Vind de wheel file (nieuwste eerst)
WHEEL_FILE = $(shell ls -t $(WHEEL_DIR)/*.whl 2>/dev/null | head -n1)

# Kleuren voor output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# ===== PHONY TARGETS =====
.PHONY: help train wheel check-wheel check-model build run stop clean logs status all shell inspect

# ===== DEFAULT TARGET =====
help: ## 📖 Toon deze help message
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Straattaal Docker Deployment            ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Beschikbare commando's:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Quick start:$(NC) make all"
	@echo ""

# ===== CHECKS =====
check-wheel: ## ✅ Check of wheel bestaat, build zo niet
	@echo "$(YELLOW)Checking voor wheel bestand...$(NC)"
	@if [ ! -d "$(WHEEL_DIR)" ] || [ -z "$(WHEEL_FILE)" ]; then \
		echo "$(RED)❌ Geen wheel gevonden in $(WHEEL_DIR)$(NC)"; \
		echo "$(YELLOW)📦 Building wheel...$(NC)"; \
		$(MAKE) wheel; \
	else \
		echo "$(GREEN)✅ Wheel gevonden: $(WHEEL_FILE)$(NC)"; \
	fi

check-model: ## ✅ Check of model getraind is, train zo niet
	@echo "$(YELLOW)Checking voor getraind model...$(NC)"
	@if [ ! -d "$(MODEL_DIR)" ] || [ -z "$$(ls -A $(MODEL_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)❌ Geen model gevonden in $(MODEL_DIR)$(NC)"; \
		echo "$(YELLOW)🤖 Training model...$(NC)"; \
		$(MAKE) train; \
	else \
		echo "$(GREEN)✅ Model gevonden in $(MODEL_DIR)$(NC)"; \
		ls -lh $(MODEL_DIR); \
	fi
	@if [ ! -d "$(ASSETS_DIR)" ] || [ ! -f "$(ASSETS_DIR)/straattaal.txt" ]; then \
		echo "$(RED)❌ straattaal.txt niet gevonden!$(NC)"; \
		echo "$(YELLOW)🤖 Training model...$(NC)"; \
		$(MAKE) train; \
	else \
		echo "$(GREEN)✅ Assets gevonden in $(ASSETS_DIR)$(NC)"; \
	fi

# ===== BUILDING =====
train: ## 🤖 Train het model
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║            Model Training                    ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Training model... Dit kan even duren...$(NC)"
	python src/slanggen/main.py
	@echo ""
	@echo "$(GREEN)✅ Model training voltooid!$(NC)"
	@echo "$(BLUE)Generated bestanden:$(NC)"
	@ls -lh $(ASSETS_DIR)/ 2>/dev/null || echo "$(RED)⚠️  Assets folder niet gevonden$(NC)"
	@ls -lh $(MODEL_DIR)/ 2>/dev/null || echo "$(RED)⚠️  Artefacts folder niet gevonden$(NC)"

wheel: ## 📦 Build de wheel package
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║            Building Wheel                    ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Building wheel package...$(NC)"
	uv build --clean
	@echo ""
	@echo "$(GREEN)✅ Wheel gebouwd!$(NC)"
	@ls -lh $(WHEEL_DIR)/

build: check-wheel check-model ## 🏗️  Build Docker image (controleert eerst wheel & model)
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Building Docker Image                ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Building Docker image '$(IMAGE_NAME)'...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo ""
	@echo "$(GREEN)✅ Docker image succesvol gebouwd!$(NC)"
	@docker images $(IMAGE_NAME)

# ===== RUNNING =====
run: ## 🚀 Start Docker container op port $(PORT)
	@echo "$(BLUE)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          Starting Container                  ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Stopping oude container (indien actief)...$(NC)"
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "$(YELLOW)Starting nieuwe container...$(NC)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):80 \
		--restart unless-stopped \
		$(IMAGE_NAME)
	@echo ""
	@echo "$(GREEN)✅ Container succesvol gestart!$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)🌐 Applicatie draait op: http://127.0.0.1:$(PORT)$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "Handige commando's:"
	@echo "  $(YELLOW)make logs$(NC)   - Bekijk logs"
	@echo "  $(YELLOW)make status$(NC) - Check status"
	@echo "  $(YELLOW)make stop$(NC)   - Stop container"

stop: ## 🛑 Stop en verwijder container
	@echo "$(YELLOW)Stopping container...$(NC)"
	@docker stop $(CONTAINER_NAME) 2>/dev/null || echo "$(RED)Container was niet actief$(NC)"
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "$(GREEN)✅ Container gestopt en verwijderd$(NC)"

# ===== MONITORING =====
logs: ## 📋 Bekijk container logs (live)
	@echo "$(BLUE)Logs van $(CONTAINER_NAME):$(NC)"
	@echo "$(YELLOW)(Druk Ctrl+C om te stoppen)$(NC)"
	@echo ""
	@docker logs -f $(CONTAINER_NAME)

status: ## 📊 Check container status
	@echo "$(BLUE)Container Status:$(NC)"
	@docker ps -a | grep $(CONTAINER_NAME) || echo "$(RED)Container niet gevonden$(NC)"
	@echo ""
	@echo "$(BLUE)Image Info:$(NC)"
	@docker images | grep $(IMAGE_NAME) || echo "$(RED)Image niet gevonden$(NC)"

shell: ## 🐚 Open shell in draaiende container
	@echo "$(YELLOW)Opening shell in container...$(NC)"
	docker exec -it $(CONTAINER_NAME) /bin/bash

inspect: ## 🔍 Inspecteer container details
	@docker inspect $(CONTAINER_NAME)

# ===== CLEANUP =====
clean: stop ## 🧹 Verwijder alle gegenereerde files en images
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@echo "Removing wheel..."
	@rm -rf $(WHEEL_DIR)
	@echo "Removing model artefacts..."
	@rm -rf $(MODEL_DIR)
	@echo "Removing assets..."
	@rm -rf $(ASSETS_DIR)
	@echo "Removing Docker image..."
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup compleet!$(NC)"

# ===== TESTING =====
test-local: ## 🧪 Test backend lokaal (zonder Docker)
	@echo "$(YELLOW)Starting backend lokaal...$(NC)"
	@echo "$(BLUE)Test op: http://127.0.0.1:80$(NC)"
	@cd backend && python app.py

test-curl: ## 🧪 Test Docker container met curl
	@echo "$(YELLOW)Testing container...$(NC)"
	@curl -s http://127.0.0.1:$(PORT) || echo "$(RED)❌ Container niet bereikbaar$(NC)"

# ===== ALL-IN-ONE =====
all: check-wheel check-model build run ## 🎯 Doe alles: check, build, run
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║              🎉 SUCCESS! 🎉                  ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)✅ Alles is klaar en draait!$(NC)"
	@echo ""
	@echo "$(BLUE)🌐 Applicatie: http://127.0.0.1:$(PORT)$(NC)"
	@echo ""
	@echo "Volgende stappen:"
	@echo "  1. Test de applicatie in je browser"
	@echo "  2. Bekijk logs met: $(YELLOW)make logs$(NC)"
	@echo "  3. Deploy naar SURF!"
	@echo ""