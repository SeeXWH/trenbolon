FROM ubuntu:latest
LABEL authors="debosh"

ENTRYPOINT ["top", "-b"]