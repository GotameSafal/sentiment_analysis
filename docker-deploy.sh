#!/bin/bash

# Sentiment Analysis Docker Deployment Script
# This script provides a complete Docker deployment workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sentiment-analysis"
API_IMAGE="${PROJECT_NAME}-api"
FRONTEND_IMAGE="${PROJECT_NAME}-frontend"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "All dependencies are installed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_warning "Created .env file from .env.example. Please update the values."
        else
            log_error ".env.example file not found. Please create environment configuration."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p logs models data monitoring/grafana/dashboards monitoring/grafana/datasources
    
    log_success "Environment setup complete"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    log_info "Building API image..."
    docker build -t $API_IMAGE .
    
    # Build frontend image
    if [ -d "web" ]; then
        log_info "Building frontend image..."
        docker build -t $FRONTEND_IMAGE ./web
    fi
    
    log_success "Docker images built successfully"
}

start_services() {
    local profile=${1:-""}
    
    log_info "Starting services..."
    
    if [ -n "$profile" ]; then
        docker-compose --profile $profile up -d
    else
        docker-compose up -d
    fi
    
    log_success "Services started successfully"
}

stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."
    docker-compose restart
    log_success "Services restarted"
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo ""
    log_info "Service logs (last 20 lines):"
    docker-compose logs --tail=20
}

show_logs() {
    local service=${1:-""}
    
    if [ -n "$service" ]; then
        log_info "Showing logs for $service:"
        docker-compose logs -f $service
    else
        log_info "Showing logs for all services:"
        docker-compose logs -f
    fi
}

health_check() {
    log_info "Performing health checks..."
    
    # Check API health
    local api_port=$(grep API_PORT .env | cut -d'=' -f2 || echo "5000")
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:$api_port/api/health &> /dev/null; then
            log_success "API is healthy"
            break
        else
            if [ $attempt -eq $max_attempts ]; then
                log_error "API health check failed after $max_attempts attempts"
                return 1
            fi
            log_info "Waiting for API to be ready... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done
    
    # Check frontend health
    local frontend_port=$(grep FRONTEND_PORT .env | cut -d'=' -f2 || echo "3000")
    if curl -f http://localhost:$frontend_port/health &> /dev/null; then
        log_success "Frontend is healthy"
    else
        log_warning "Frontend health check failed"
    fi
    
    # Check Redis health
    if docker-compose exec redis redis-cli ping &> /dev/null; then
        log_success "Redis is healthy"
    else
        log_warning "Redis health check failed"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (optional)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup complete"
}

backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating backup in $backup_dir..."
    
    mkdir -p $backup_dir
    
    # Backup models
    if [ -d "models" ]; then
        cp -r models $backup_dir/
    fi
    
    # Backup logs
    if [ -d "logs" ]; then
        cp -r logs $backup_dir/
    fi
    
    # Backup Redis data
    docker-compose exec redis redis-cli BGSAVE
    docker cp $(docker-compose ps -q redis):/data/dump.rdb $backup_dir/redis_dump.rdb
    
    log_success "Backup created in $backup_dir"
}

show_help() {
    echo "Sentiment Analysis Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                 Build Docker images"
    echo "  start [profile]       Start services (optional profile: monitoring, production)"
    echo "  stop                  Stop services"
    echo "  restart               Restart services"
    echo "  status                Show service status"
    echo "  logs [service]        Show logs (optional: specific service)"
    echo "  health                Perform health checks"
    echo "  cleanup               Clean up containers and images"
    echo "  backup                Backup data and models"
    echo "  deploy                Full deployment (build + start + health check)"
    echo "  help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Full deployment"
    echo "  $0 start monitoring          # Start with monitoring stack"
    echo "  $0 logs sentiment-api        # Show API logs"
    echo "  $0 health                    # Check service health"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_dependencies
        setup_environment
        build_images
        ;;
    "start")
        check_dependencies
        start_services $2
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs $2
        ;;
    "health")
        health_check
        ;;
    "cleanup")
        cleanup
        ;;
    "backup")
        backup_data
        ;;
    "deploy")
        check_dependencies
        setup_environment
        build_images
        start_services
        sleep 10
        health_check
        log_success "Deployment complete!"
        echo ""
        log_info "Access the application:"
        echo "  API: http://localhost:$(grep API_PORT .env | cut -d'=' -f2 || echo "5000")"
        echo "  Frontend: http://localhost:$(grep FRONTEND_PORT .env | cut -d'=' -f2 || echo "3000")"
        echo "  API Docs: http://localhost:$(grep API_PORT .env | cut -d'=' -f2 || echo "5000")/api/status"
        ;;
    "help"|*)
        show_help
        ;;
esac
