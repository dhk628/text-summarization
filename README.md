(This project is in progress)

# Motivation

As a PhD student and a data scientist, I have to read numerous papers, and so a LLM that can summarize a research paper effectively would be quite beneficial. Of course, all papers come with an abstract and an introduction, but these may not always be a suitable summary for readers. (Perhaps the abstract is too short, and the introduction focuses too much on the background.) The end goal of this project would be to fine-tune a LLM so that when I input a PDF file, it returns a reasonable summary. A nice feature would be the ability to instruct the model which part of the PDF to summarize, e.g. "Summarize Section 3 of this paper."

# Fine-Tuning a LLM

For the first step, I will practice fine-tuning a LLM. For this section, I will roughly follow this [Kaggle tutorial](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models) and fine-tune BART on dialogues. This is done in [`dialogue.ipynb`](dialogue.ipynb). The outputs of the notebook with the Plotly plots can be viewed [here](https://dhk628.github.io/text-summarization/). The training script is in [`src/models/train_model.py`](src/models/train_model.py) and the evaluation script is in [`src/models/eval_model.py`](src/models/eval_model.py).

The model can be containerized in a docker using the command
```
docker build -f docker/Dockerfile -t dialogue-summarizer .
```
The docker then can be run using
```
docker run --name dialogue-summarizer -p 8501:8501 dialogue-summarizer
```
This makes the Streamlit app available on `localhost:8501/`.

Here is an example of the results of the model. The original dialogue is:
>Beatrice: I am in town, shopping. They have nice scarfs in the shop next to the church. Do you want one?
Leo: No, thanks
Beatrice: But you don't have a scarf.
Leo: Because I don't need it.
Beatrice: Last winter you had a cold all the time. A scarf could help.
Leo: I don't like them.
Beatrice: Actually, I don't care. You will get a scarf.
Leo: How understanding of you!
Beatrice: You were complaining the whole winter that you're going to die. I've had enough.
Leo: Eh.

The reference summary provided in the dataset is:
>Beatrice wants to buy Leo a scarf, but he doesn't like scarves. She cares about his health and will buy him a scarf no matter his opinion.

The model summary *before fine-tuning* is:
>An exchange between Leo and Beatrice, who are both suffering from colds.

The model summary *after fine-tuning* is:
>Beatrice is in town, shopping. She will buy a scarf for Leo.