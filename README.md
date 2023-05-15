# Psychotherapy AI Assistant: Prompt Engineering

Be sure you checked the usage in your [dashboard](https://platform.openai.com/account/usage), the generation using `gpt-4` cost a lot D:

## Description

This project utilizes OpenAI's GPT model to generate multiple fictitious answers to a list of questions, particularly for running psychotherapy report summarization test in Traditional Chinese. It is designed to generate creative responses by emulating a character in need of psychotherapy treatment. The script uses threading to handle multiple requests concurrently, and it also handles API rate limits and other exceptions for a smooth user experience.
Using techniques introduced here: [INSTRUCT](https://medium.com/@ickman/instruct-making-llms-do-anything-you-want-ff4259d4b91), we designed a prompt that can perform accurate fictitious answer generation and psychotherapy report summarization.

## Installation

- To install, you need Python installed on your system. The version used for developing this script is Python 3.8, but it should be compatible with other Python 3 versions as well.

- To get started, first clone the repository to your local system.
- Next, navigate to the directory containing the script. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

### General Arguments

To use this 2 scripts, you need to have an active OpenAI API key. You can either pass this as an argument (`--openai_api_key <your api key>`) or set it as an environment variable (`OPENAI_API_KEY`).

The scripts supports several command line arguments:

- `--model`: Name of the model to use (default is "gpt-4").
- `--temperature`: Controls randomness of the model's output (default is 0.7).
- `--max_tokens`: Maximum number of tokens for the model to generate (default is 1000).
- `--thread_num`: Number of threads to run in parallel (default is 2).
- `--questions_path`: Path to the JSON file containing the list of questions (default is "questions.json").

### `generate_answer.py`

- `--request_num`: Number of fictitious answer requests (default is 8).

```bash
python generate_answer.py --model gpt-4 --temperature 0.7 --max_tokens 1000 --thread_num 2 --request_num 8 --questions_path questions.json --openai_api_key <your-api-key>
```

Generate 100 answers cost about $6

### `generate_report.py`

- `--answers_path`: Path to the directory containing the generated answers (default is "cleaned_answers").

```bash
python generate_report.py --model gpt-4 --temperature 0.7 --max_tokens 1000 --thread_num 2 --questions_path questions.json --answers_path ./cleaned_answers/ --openai_api_key <your-api-key>
```

Generate 100 reports cost about $8


## Example

Let's assume you have a JSON file named `questions.json` that contains an array of questions. The content of `questions.json` might look like this:

```json
[
    "你以前有過諮商的經驗，或者接受過任何形式的心理治療嗎？",
    "目前你有任何疾病或是用藥的情況嗎？",
    "你的童年過得怎麼樣？有沒有什麼重大的事件特別印象深刻？",
    "你能告訴我一下你現在的工作或學校狀況嗎？"
]
```

You can run `generate_answer.py` as follows:

```bash
python generate_answer.py --model gpt-4 --temperature 0.7 --max_tokens 1000 --thread_num 2 --request_num 10 --questions_path questions.json --openai_api_key <your_api_key>
```

The script will generate 10 responses to these questions, emulating the persona of a character in need of psychotherapy treatment. The responses will be saved in directories: `answers`, `cleaned_answers`, and `extracted_failed_answers`.

Then you can run `generate_report.py` to generate reports from the generated answers:

```bash
python generate_report.py --model gpt-4 --temperature 0.7 --max_tokens 1000 --thread_num 2 --questions_path questions.json --answers_path ./cleaned_answers/ --openai_api_key <your-api-key>
```

Finally you'll see the report in `cleaned_reports` directory. An example like this:

```txt
## 個人資料
- 林先生，42歲，教師。

## 想要解決的問題
- 無法應對工作和個人生活壓力。

## 問題的緣由
- 教學工作繁重，壓力持續幾個月。

## 疾病歷史
- 無諮商或心理治療經驗。
- 無疾病或用藥情況。

## 個人和社交背景
- 擅長教學和溝通，喜歡閱讀和運動。
- 與朋友的聯繫變得比較少。

## 家庭背景
- 與家人關係良好。
- 無心理健康問題病史。

## 治療目標
- 學會更好地應對壓力，改善心理健康狀況。
```

Any result that is not valid will be saved in `extracted_failed_answers` and `extracted_failed_reports` directory.

---

This project is not a replacement for professional psychotherapists. It's a tool designed to assist in the process of psychotherapy treatment and should be used as such.