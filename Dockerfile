FROM python:3.8.13-slim-buster as image

MAINTAINER Patryk Orzechowski

# Install required packages
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
ADD docker/generate.py generate.py
ADD docker/myMethod.py myMethod.py

#Here argument required
#ARG dataset

#Setting specific value
ARG dataset=digen8

#Setting specific seed
ARG seed=4426

# Set the sizes of the dataset. By default DIGEN has size 1000 x 10
ARG rows=1000
ARG columns=10

# Create all dataset in csv format and save them into current directory in CSV format
RUN echo "$(python generate.py -d $dataset -r $rows -c $columns -s $seed)"

# In order to benchmark a new method against DIGEN and maintain reproducibility, please update a myMethod.py file, according to the instructions:
RUN echo "$(python myMethod.py -d $dataset)"


# Reproduce all the results for all the datasets and all classifiers (uncomment below). This may take several weeks!
#RUN echo "$(python reproduce.py $dataset $rows $columns)"


# Export all the files to the host
FROM scratch as export
COPY --from=image /app ./


#### HOW TO USE: ####
# Please build the image with the following command:
#docker build --no-cache --progress=plain --file Dockerfile --output . .

# You can pass additional arguments to build a specific dataset or shape of the dataset.
# If no parameters are passed, it is assumed that all the datasets should be run. The following command benchmarks myMethod for a single dataset only:
#docker build --no-cache --progress=plain --file Dockerfile --output . . --build-arg 'dataset=digen8_4426' --build-arg 'rows=1000'


# Acknowledgements
# Thanks to Ben T for useful answer at StackOverflow.
