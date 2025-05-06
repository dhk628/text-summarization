(This project is in progress)

# Motivation

As a PhD student and a data scientist, I have to read numerous papers, and so a LLM that can summarize a research paper effectively would be quite beneficial. Of course, all papers come with an abstract and an introduction, but these may not always be a suitable summary for readers. (Perhaps the abstract is too short, and the introduction focuses too much on the background.) The end goal of this project would be to fine-tune a LLM so that when I input a PDF file, it returns a reasonable summary. A nice feature would be the ability to instruct the model which part of the PDF to summarize, e.g. "Summarize Section 3 of this paper."

# Fine-Tuning a LLM

For the first step, I will practice fine-tuning a LLM. For this section, I will roughly follow this [Kaggle tutorial](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models) and fine-tune BART on dialogues. This is (being) done in `dialogue.ipynb`.