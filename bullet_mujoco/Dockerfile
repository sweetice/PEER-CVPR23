FROM python:3.8

COPY docker.sh .
RUN chmod +x docker.sh
RUN ./docker.sh

# RUN /docker.sh

ENV CODE_DIR /root/code

COPY ./ $CODE_DIR/peer
WORKDIR $CODE_DIR/peer

COPY ./.mujoco /home/
