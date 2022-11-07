#FROM gcr.io/monzo-build/python-jobs/python3:latest
FROM python:3.8
WORKDIR /root
#ARG JOB_NAME
#ARG GIT_BRANCH
#ARG GIT_SHA

## Requirements and dependencies
#COPY ./libraries/monzo_models ./libraries/monzo_models
#COPY ./libraries/monzo_datasets ./libraries/monzo_datasets
#COPY ./${JOB_NAME}/requirements.txt ./${JOB_NAME}/requirements.txt
#COPY ./${JOB_NAME}/bin ./${JOB_NAME}/bin
COPY . .

## Add Python install location to $PATH
#ENV PATH="/root/.local/bin:${PATH}"
#
## Make environment variables available in container (Git branch and commit hash)
#ENV GIT_BRANCH=${GIT_BRANCH}
#ENV GIT_SHA=${GIT_SHA}

# Dependencies
# Note doing this as a single step means we don't store intermediate image
# layers we don't want. this saves us many megabytes (>300mb saved)
#RUN apt-get update \
# && ./${JOB_NAME}/bin/setup_python \
# && apt-get remove -y build-essential \
# && apt-get autoremove -y
RUN apt-get update  \
    && pip install cmake  \
    && pip install dlib  \
    && ./install_dependencies.sh \
    && pip install tf_slim

## The pipeline
#COPY ./${JOB_NAME}/run.sh ./${JOB_NAME}/run.sh
#COPY ./${JOB_NAME}/src ./${JOB_NAME}/src
#COPY ./${JOB_NAME}/main.py ./${JOB_NAME}/main.py
#
## Unit tests
#COPY ./${JOB_NAME}/tests ./${JOB_NAME}/tests
#RUN ./${JOB_NAME}/bin/run_tests

#ENTRYPOINT ["./job.train-account-takeover-at-login-model/run.sh"]
