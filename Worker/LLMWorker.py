import pandas as pd
import numpy as np
import re
import time
import threading
import queue
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class LLM_brain():
    def __init__(self, model_name):
        self.llm = OpenAI(api_key='lm-studio', base_url="http://localhost:1234/v1")
        self.model_name = model_name

    def _validate_response(self,response: str) -> int:
        numbers = re.findall(r'\b[1-5]\b', response)

        if not numbers:
            raise ValueError(f"No valid number found in response: {response}")
        if len(numbers) > 1:
            raise ValueError(f"Multiple candidates detected: {numbers}")
        result = int(numbers[0])
        if not (1 <= result <= 5):
            raise ValueError(f"Value {result} out of valid range")

        return result

    def get_rating(self, prompt: str, temperature: float = 0.0) -> int:
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role':'system', 'content':'Return ONLY an integer between 1-5'},
                {'role':'user', 'content':prompt}
            ],
            temperature=temperature,
            stream=False
        )
        response_content = completion.choices[0].message.content.strip()
        #print(response_content)
        return self._validate_response(response_content)


class LLM_worker(LLM_brain):
    def __init__(self, model_name, MAX_CONCURRENT_REQUESTS, temperature):
        super(LLM_worker, self).__init__(model_name)
        self.task_queue = queue.Queue()
        self.results_lock = threading.Lock()
        self.MAX_CONCURRENT_REQUESTS = MAX_CONCURRENT_REQUESTS
        self.temperature = temperature
        self.demographic = {}
        self.tasks = []
        self.results = {}

    def change_model(self, model_name):
        self.model_name = model_name

    def _generate_demographic(self, distrib_df: pd.DataFrame):
        for col in ['gender', 'race', 'age', 'occupation', 'education']:
            self.demographic[col] = np.random.choice(distrib_df[col].dropna().unique())

    def _make_prompt(self, comment_text: str) -> str:
        if not self.demographic:
            demographic_str = "general human"
        else:
            parts = []
            for k, v in self.demographic.items():
                parts.append(f"{k.capitalize()}: {v}")
            demographic_str = " | ".join(parts)

        prompt = f"""
You are asked to simulate a real human user's response to the following question-answering task.
Please assume the user has the following demographics: {demographic_str}.

Please read the text carefully, then give the rating:
A difficulty rating (1-5) of answering the question, where:
   1 = Not difficult at all
   2 = Slightly difficult
   3 = Moderately difficult
   4 = Very difficult
   5 = Extremely difficult
Example: 3
Text to read:
{comment_text}

Your response MUST follow this exact format:
Difficulty: [1-5]
Do not provide a justification! Output ONLY the number, with no explanation. If your answer contains anything other than a single digit, your output is invalid.
ps: if the llm is Qwen:
/no_think
                """
        return prompt.strip()

    def _task_gathering(self, df_A, distrib_df, multi_persona):
        for idx, row in df_A.iterrows():
            if multi_persona:
                self._generate_demographic(distrib_df)
                prompt_text = self._make_prompt(comment_text=row['text'])
            else:
                prompt_text = self._make_prompt(comment_text=row['text'])
            self.task_queue.put((idx, self.get_rating, (prompt_text,), {'temperature':self.temperature}, self.results, idx))


    def _task_process(self):
        while True:
            task = self.task_queue.get()
            if task is None: 
                self.task_queue.task_done()
                break

            task_id, func, args, kwargs, result_store, result_key = task
            try:
                result = func(*args, **kwargs)
                with self.results_lock:
                    result_store[result_key] = result
            except Exception as e:
                with self.results_lock:
                    result_store[result_key] = e
            finally:
                self.task_queue.task_done()

    def give_answer(self, df_A, distrib_df, multi_persona):
        df_A['llm_rating'] = pd.Series(dtype='Int64')
        df_A['validation_status'] = ''
        self.tasks = []
        for _ in range(self.MAX_CONCURRENT_REQUESTS):
            t = threading.Thread(target=self._task_process)
            t.daemon = True
            t.start()
            self.tasks.append(t)
        self._task_gathering(df_A, distrib_df, multi_persona)
        self.task_queue.join()
        for idx in self.results:
            result = self.results[idx]
            if isinstance(result, Exception):
                df_A.at[idx, 'validation_status'] = f'INVALID: {str(result)}'
            else:
                df_A.at[idx, 'llm_rating'] = result
                df_A.at[idx, 'validation_status'] = 'VALID'
        # df_A_sub = df_A.iloc[:,:1].copy()
        # df_A_sub['llm_rating'] = df_A['llm_rating']
        self.task_exit()
        return df_A['llm_rating'].to_numpy()

    def task_exit(self):
        for _ in range(len(self.tasks)):
            self.task_queue.put(None)
        for t in self.tasks:
            t.join()
