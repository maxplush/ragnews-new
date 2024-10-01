# Retrieval Augmented Generation (RAG) News Project ![](https://github.com/maxplush/ragnews-new/workflows/tests/badge.svg)

This project involves Retrieval Augmented Generation (RAG), a method that combines retrieval and generation models to enhance the ability to generate responses based on retrieved information. Specifically, this project focuses on answering questions about the US election in 2024 using a database of articles from various news sources.

## Prerequisites

- Before running the script, make sure you have the packages listed in the requirements.txt file.
- A `.env` file in the same directory containing your  [Groq API KEY](https://groq.com). The file should have the following format:
  
  ```env
  GROQ_API_KEY=your_groq_api_key_here

## Usage

To use ragnews.py, follow these steps:

1. Ensure your .env file is configured correctly with your Groq API key. Connect your .env by running the command.

```
$ export $(cat .env)
```

2. Run the ragnews.py script within your virtual environment:

```
python3 ragnews.py
```

3. The system will prompt with:

```
ragnews>
```

4. Ask a question, for example:

```
What is Trump's stance on abortion compared to Harris?
```

```
I'm happy to help you with that! To answer your question, I'll briefly review the articles provided to me.

According to the articles, on the topic of abortion, Donald Trump, the 45th President of the United States, held a strong anti-abortion stance throughout his presidency. In 2019, he reiterated his stance by stating that he thinks, "abortion is a terrible thing" and "we have to get rid of Roe v. Wade."

On the other hand, Kamala Harris, the 49th Vice President of the United States, has consistently taken a pro-choice stance. During her 2020 presidential campaign, she advocated for the protection of Roe v. Wade, the landmark Supreme Court decision that legalized abortion in the United States, and pledged to repeal any state laws that restrict access to abortion.

In short, the articles suggest that Trump has a strong anti-abortion stance, aiming to reverse Roe v. Wade, while Harris, as a pro-choice candidate, strives to protect and safeguard reproductive rights.
```

## Example Usage of Evaluate 
```
python3 ragnews/evaluate.py
```

```
Number Correct: 95
Total Test Cases: 127
Accuracy: 74.80%
```

![Screenshot of my project](images/Screenshot_2024-09-30_at_11.30.48_PM.png)
![Screenshot of my project](images/Screenshot_2024-09-30_at_11.31.37_PM.png)

