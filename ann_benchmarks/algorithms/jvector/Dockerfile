ARG JVECTOR_VERSION=annbench3
FROM ann-benchmarks

ARG JVECTOR_VERSION
WORKDIR /home/app

# Install temurin JDK
RUN apt-get update
RUN apt-get install -y wget apt-transport-https gnupg
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update
RUN apt-get install -y temurin-20-jdk 

# Install JVector src
RUN wget https://github.com/tjake/jvector/archive/refs/heads/${JVECTOR_VERSION}.tar.gz
RUN tar -xzf ${JVECTOR_VERSION}.tar.gz
ENV JAVA_HOME=/usr/lib/jvm/temurin-20-jdk-amd64
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
