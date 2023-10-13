ARG JVECTOR_VERSION=annbench3
FROM ann-benchmarks

ARG JVECTOR_VERSION
WORKDIR /home/app

# Install JDK
RUN apt-get update
RUN apt-get install -y wget apt-transport-https gnupg software-properties-common
RUN wget -O- https://apt.corretto.aws/corretto.key | apt-key add -
RUN add-apt-repository 'deb https://apt.corretto.aws stable main'
RUN apt-get update
RUN apt-get install -y java-21-amazon-corretto-jdk


# Install JVector src
RUN wget https://github.com/tjake/jvector/archive/refs/heads/${JVECTOR_VERSION}.tar.gz
RUN tar -xzf ${JVECTOR_VERSION}.tar.gz
WORKDIR /home/app/jvector-${JVECTOR_VERSION}
RUN ./mvnw clean compile -DskipTests
RUN ./mvnw compile exec:exec@annbench -Dexec.args="just compile this" || true


ENV JVECTOR_VERSION=${JVECTOR_VERSION}
WORKDIR /home/app

# Custom entrypoint that also starts the Elasticsearch server.
RUN echo 'set -eux' >> entrypoint.sh
RUN echo 'sh -c "cd jvector-${JVECTOR_VERSION} && nohup ./mvnw compile exec:exec@annbench-1core > nohup.out 2>&1 &"' >> entrypoint.sh
RUN echo 'python3 -u run_algorithm.py "$@" || cat jvector-${JVECTOR_VERSION}}/nohup.out' >> entrypoint.sh

ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
