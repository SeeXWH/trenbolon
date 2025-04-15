FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY *.go ./

RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/euler_search .


FROM alpine:latest


RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY --from=builder /app/euler_search /app/euler_search

RUN chown appuser:appgroup /app && chown appuser:appgroup /app/euler_search

USER appuser


ENTRYPOINT ["/app/euler_search"]

# --- Важное примечание по файлу вывода ---
# Приложение будет пытаться создать/открыть файл 'answer_go_minimal.txt'
# внутри контейнера в директории /app.
# Чтобы сохранить этот файл на вашем хост-компьютере, необходимо использовать
# монтирование тома (volume mount) при запуске контейнера командой 'docker run'.
# Пример:
# docker run -v $(pwd)/results:/app your_image_name
# Это создаст директорию 'results' на вашем хосте (если ее нет) и
# свяжет ее с директорией '/app' внутри контейнера. Файл
# 'answer_go_minimal.txt' появится в './results' на хосте.
