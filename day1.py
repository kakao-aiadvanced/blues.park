import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def gpt4omini(input):
    completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": input,
        }
        ],
        model="gpt-4o-mini",
        )
    print(completion.choices[0].message.content)

import ollama

def llama3(input):
    response = ollama.chat(
        model='llama3.1',
        messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            },
        ],
    )
    print(response['message']['content'])

def prompt(input): gpt4omini(input)

# get_chat_response("""
# You are a English to Korean translator. For example:
# 1. 'bird' is '새'
# 2. 'cow' is '소'
# 3. 'horse' is '말'
# Request: "dog"
# Result:
# """)


# get_chat_response("""
# Convert the following natural language requests into SQL queries:
# 1. "Write your Prompt": SELECT * FROM employees WHERE salary > 50000;
# 2. "Write your Prompt": SELECT * FROM products WHERE stock = 0;
# 3. "Write your Prompt": SELECT name FROM students WHERE math_score > 90;
# 4. "Write your Prompt": SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
# 5. "Write your Prompt": SELECT city, COUNT(*) FROM customers GROUP BY city;

# Request: "Find the average salary of employees in the marketing department."
# SQL Query:
# """)

prompt(
"""
Solve the following problem step-by-step: 23 + 47

Step-by-step solution:
1. 1의 자리에 있는 3과 7을 더하면 10이 된다.
2. 10의 자리에 있는 2와 4를 더하면 6이 된다.
3. 1의 자리에서 넘어온 1을 10의 자리에 더하면 7이 된다.
4. 따라서 70이 된다.

Answer:

Solve the following problem step-by-step: 123 - 58

Step-by-step solution:
1. 1의 자리에 있는 3에서 8을 뺄 수 없다. 따라서 10의 자리에서 1을 빌려온다.
2. 13에서 8을 빼면 5다.
3. 10의 자리에 남은 1에서 5를 뺄 수 없다. 따라서 100의 자리에서 1을 빌려온다.
4. 11에서 5를 빼면 6이다.
5. 따라서 65가 된다.

Answer:

Solve the following problem step-by-step: 345 + 678 - 123

Answer:
"""
)
