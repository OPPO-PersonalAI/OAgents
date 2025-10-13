#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are modifications by OPPO PersonalAI Team.
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import os
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import logging
import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from typing import Dict, List

from scripts.scorer import question_scorer
from scripts.reformulator import prepare_response
from scripts.searcher import SearchTool
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.audio_inspector_tool import AudioInspectorTool
from scripts.visual_inspector_tool import VisualInspectorTool
from scripts.async_web_crawler import (
    CrawlerReadTool,
    CrawlerArchiveSearchTool,
    SimpleCrawler,
)
from scripts.automodel import get_api_model, process_selected_tasks_param, prepare_model_kwargs

from oagents.memory import ActionStep, PlanningStep, TaskStep
from tqdm import tqdm

from oagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
)

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil"
]


parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
env_path = os.path.join(parent_dir, '.env')

load_dotenv(dotenv_path=env_path, override=True)
login(os.getenv("HF_TOKEN"))

logger = logging.getLogger(__name__)

jsonl_lock = threading.Lock()

logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="gpt-4.1")
    parser.add_argument("--model_id_search", type=str, default="gpt-4.1")
    parser.add_argument("--run_name", type=str, default="init_run")
    parser.add_argument("--debug", default=False, action="store_true")
    # infer params
    parser.add_argument('--planning_interval', type=int, default=1, help='Number of rollouts per state.')
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps for ReAct agent.")
    parser.add_argument("--temperature", default=None, type=float, help= "The temperature for llm generation.")
    parser.add_argument('--top_p', default=None, type=float, help="The top_p for llm generation.")
    parser.add_argument('--reflection', action='store_true', default=True, help='Enable reflection')
    # data selection
    parser.add_argument("--split", type=str, default="validation", choices=['validation','test'])
    parser.add_argument("--level", type=str, default="all", choices=["all", "1", "2", "3"])
    parser.add_argument("--selected-tasks", default=['32102e3e-d12a-4209-9163-7b3a104efe5d'], nargs='*', help="Tasks to run: specify single or multiple indices (--selected-tasks 1 or --selected-tasks 1 2 5), a single task ID, or a path to a text file with one task ID per line")
    # search params
    parser.add_argument('--search_tool_reflection', action='store_true', default=False, help='Enable search tool reflection')
    # plan params
    parser.add_argument('--subtask', action='store_true', default=False, help='Enable subtask')
    parser.add_argument('--static_plan', action='store_true', default=False, help='Use static plan')
    parser.add_argument('--dynamic_update_plan', action='store_true', default=False, help='Use dynamic update plan')
    # TTS params
    parser.add_argument('--n_rollouts', type=int, default=1, help='Number of rollouts per state.')
    parser.add_argument('--search_type', type=str, choices=['BON-wise','Beam-Search','Tree-Search','BON','default'], default='default', help='Type of search algorithm to use.')
    parser.add_argument('--reflection_threshold', type=int, default=2, help='Number of rollouts per state.')
    parser.add_argument('--verify_type', type=str, choices=['list-wise','scoring'], default='list-wise', help='Type of search algorithm to use.')
    parser.add_argument('--result_merging_type', type=str, choices=['list-wise','scoring','voting'], default='list-wise', help='Type of search algorithm to use.')
    # memory params
    parser.add_argument('--summary', action='store_true', default=False, help='Summarize the current step memory')
    parser.add_argument('--use_long_term_memory', action='store_true', default=False, help='Use long-term memory')
    parser.add_argument('--retrieve_key_memory', action='store_true', default=False, help='Retrieve key memory')
    
    return parser.parse_args()

def load_gaia_dataset(args):
    eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", trust_remote_code=True)[args.split]
    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{args.split}/" + row["file_name"]
        return row

    eval_ds = eval_ds.map(preprocess_file_paths)
    eval_df = pd.DataFrame(eval_ds)
    return eval_df

def create_agent_hierarchy(model: Model, model_search: Model, args, debug=False):
    crawler = SimpleCrawler(serpapi_key=os.getenv("SERP_API_KEY"))
    text_limit = 100000

    search_types = ['wiki', 'google', 'baidu', 'bing', 'duckduckgo']
    search_tools = [SearchTool(search_type=st, reflection=args.search_tool_reflection) for st in search_types]
    
    WEB_TOOLS = [
        CrawlerReadTool(crawler),
        CrawlerArchiveSearchTool(crawler),
        TextInspectorTool(model, text_limit),
        VisualInspectorTool(model, text_limit), 
        AudioInspectorTool(model, text_limit), 
    ]
    WEB_TOOLS += search_tools
    manager_agent = CodeAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=args.max_steps,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=args.planning_interval,
        debug=debug,
        subtask=args.subtask,
        static_plan=args.static_plan,
        dynamic_update_plan=args.dynamic_update_plan,
        reflection=args.reflection,
        reflection_threshold=args.reflection_threshold,
        verify_type=args.verify_type,
        result_merging_type=args.result_merging_type,
        n_rollouts=args.n_rollouts,
        search_type=args.search_type,
        summary=args.summary,
        use_long_term_memory=args.use_long_term_memory,
        retrieve_key_memory=args.retrieve_key_memory,
    )
    return manager_agent

def append_answer(entry: dict, jsonl_file: str, file_lock) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entry) + "\n"
    with file_lock:
        with open(jsonl_file, "a", encoding="utf-8") as fp:
            fp.write(data)
    assert os.path.exists(jsonl_file), "File not found!"
    logger.info("Answer exported to file: {}".format(jsonl_file.resolve()))

def extract_intermediate_steps(agent):

    intermediate_steps = []
    for memory_step in agent.memory.steps:
        memory_step.model_input_messages = None
        step_dict = memory_step.dict()
        if isinstance(memory_step, ActionStep):
            step_dict['step_type'] = 'action'
            step_dict.pop('model_output_message', None)
        elif isinstance(memory_step, TaskStep):
            step_dict['step_type'] = 'task'
        elif isinstance(memory_step, PlanningStep):
            step_dict['step_type'] = 'planning'
            step_dict.pop('model_output_message_facts', None)
            step_dict.pop('model_output_message_plan', None)
        else:
            step_dict['step_type'] = 'unknown'
        intermediate_steps.append(step_dict)
    return intermediate_steps


def answer_single_question(example, args, model_id, model_id_search, answers_file, debug=False):

    text_limit = 100000
    model_name, key, url, model_wrapper = get_api_model(model_id)
    model_name_search, key_search, url_search, model_wrapper_search = get_api_model(model_id_search)

    kwargs = prepare_model_kwargs(model_id, args)
    kwargs_search = prepare_model_kwargs(model_id_search, args)

    model = model_wrapper(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key,
        api_base=url,
        **kwargs
    )

    model_search = model_wrapper_search(
        model_name_search,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        api_key=key_search,
        api_base=url_search,
        **kwargs_search
    )

    document_inspection_tool = TextInspectorTool(model, text_limit)
    audio_inspection_tool = AudioInspectorTool(model, text_limit)
    visual_inspection_tool = VisualInspectorTool(model, text_limit)

    agent = create_agent_hierarchy(model, model_search, args, debug)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). 
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool, audio_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        final_result = agent.run(augmented_question)
        output = str(final_result)

        intermediate_steps = extract_intermediate_steps(agent)
        
        intermediate_steps_check = [str(step) for step in agent.memory.steps]
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps_check]) else False
        
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        logger.error(f"Error on task {example['task_id']}\n{e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
        
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "true_answer": example["true_answer"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
    }
    append_answer(annotated_example, answers_file, jsonl_lock)


def get_examples_to_answer(answers_file, eval_df, selected_tasks=None, level='all', debug=False) -> List[dict]:
    logger.info(f"Loading answers from {answers_file}...")
    try:
        answer_df = pd.read_json(answers_file, lines=True)
        done_questions = answer_df.get("task_id", []).tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.info("Error when loading records: ", e)
        logger.info("No usable records! ▶️ Starting new.")
        done_questions = []

    if level == 'all':
        filtered_df = eval_df
    else:
        filtered_df = eval_df[eval_df['task'] == level]

    if selected_tasks:
        if isinstance(selected_tasks[0], int):
            filtered_df = eval_df.iloc[selected_tasks]
        else:
            filtered_df = eval_df[eval_df['task_id'].isin(selected_tasks)]
    
    if debug:
        done_questions = []
    return [row.to_dict() for idx, row in filtered_df.iterrows() if row["task_id"] not in done_questions]

def main():
    args = parse_args()
    logger.info(f"Starting run with arguments: {args}")
    answers_file = f"output/{args.split}/{args.run_name}.jsonl"

    eval_df = load_gaia_dataset(args)

    selected_tasks = process_selected_tasks_param(args.selected_tasks)
    level = args.level
    tasks_to_run = get_examples_to_answer(answers_file, eval_df, selected_tasks, level, args.debug)
    if args.debug or args.concurrency == 1:
        for example in tasks_to_run:
            answer_single_question(example, args, args.model_id, args.model_id_search, answers_file, args.debug)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
            futures = [
                exe.submit(answer_single_question, example, args, args.model_id, args.model_id_search, answers_file, args.debug)
                for example in tasks_to_run
            ]
            for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

    logger.info("All tasks processed.")


if __name__ == "__main__":
    main()