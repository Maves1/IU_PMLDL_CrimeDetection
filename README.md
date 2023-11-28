# IU_PMLDL_JokeGeneration

## Project Structure

### Notebooks

`notebooks` folder contains Jupyter Notebooks that are used to develop the models
and conduct experiments.

### Models

We trained two models: the first one was trained from scratch, and the second one
was already pre-trained, and we also trained it on the same dataset.

This readme is more about the application, so if you have a desire to learn more
about models, please check out the report in `reports/report.pdf`

## Application

![bla](https://github.com/timuraiz/IU_PMLDL_JokeGeneration/blob/main/reports/photo_2023-11-29_00-07-46.jpg)

This project involves creating a FastAPI application that generates jokes and deploying it using Docker on Yandex Cloud. The FastAPI app exposes two endpoints: the root endpoint ("/home") for a welcome message and "/generate-joke". The application is packaged into a Docker container for easy deployment.

This project demonstrates the combination of FastAPI for building a RESTful API, Docker for containerization, and Yandex Cloud for scalable and reliable deployment of the FastAPI Jokes App.

## References
1. [GPT From Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1694s)
2. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
3. [How to train GPT](https://habr.com/ru/companies/neoflex/articles/722584/)
4. [kaggle gpt2 training notebook](https://www.kaggle.com/code/suraj520/pytorch-train-gpt2-and-generate-text-from-it)
5. [GPT2 HuggingFace](https://huggingface.co/gpt2)
6. [JokeGeneration](https://github.com/Maves1/IU_NLP_Joke_Generator/tree/main)
