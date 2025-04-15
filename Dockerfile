# Используем официальный образ Go для сборки
FROM golang:1.22-alpine AS builder

# Устанавливаем зависимости для сборки
RUN apk add --no-cache git

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы модулей
COPY go.mod go.sum ./
RUN go mod download

# Копируем исходный код
COPY *.go ./

# Собираем приложение
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/euler_search .

# Используем минимальный образ alpine для финального контейнера
FROM alpine:3.18

# Копируем собранный бинарник из builder
COPY --from=builder /app/euler_search /app/euler_search

# Создаем директорию для выходных файлов
RUN mkdir -p /app/output

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда для запуска приложения
CMD ["./euler_search"]