FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o euler_search .

FROM alpine:3.18
WORKDIR /app
RUN mkdir -p /app/output && chmod 777 /app/output
COPY --from=builder /app/euler_search .
CMD ["/app/euler_search"]