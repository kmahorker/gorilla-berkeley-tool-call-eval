import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import convert_to_openai_messages

from agents.packages.core.tool_search.ToolSearchClient import ToolSearchClient
from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
    convert_system_prompt_into_user_prompt,
    combine_consecutive_user_prompts,
)
from openai import OpenAI


class OceanWaterHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.model = ChatOpenAI(model=model_name, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))        
        self.tool_search_client = ToolSearchClient(api_key="")

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            return default_decode_ast_prompting(result, language)
        else:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result):
        if "FC" not in self.model_name:
            return default_decode_execute_prompting(result)
        else:
            function_call = convert_to_function_call(result)
            return function_call

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        
        # Instantiate Dynamic Tool Selection Agents
        tool_selection_agent, initial_state = self.tool_search_client.get_tool_selection_agent(self.model)
        initial_state["messages"] = repr(message)
        
        return tool_selection_agent.invoke(initial_state)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        
        #Parse API response from the tool selection agent
        
        # Get all the messages with
        openai_messages = []
        input_tokens = 0
        output_tokens = 0
        
        for msg in api_response["messages"]:
            message = convert_to_openai_messages(msg)
            input_tokens += message.usage_metadata["input_tokens"]
            output_tokens += message.usage_metadata["output_tokens"]
            
            try:
                model_responses = []
                for func_call in message.tool_calls:
                    if func_call["function"]["name"] == "select_tools":
                        continue # Skip the select_tools tool call
                    if func_call["function"]["name"] == "selected_dynamic_tool_call":
                        model_responses.append({
                            func_call["function"]["arguments"]["selectedFunctionId"]: func_call["function"]["arguments"]["selectedFunctionArguments"]
                        })
                tool_call_ids = [
                    func_call["id"] for func_call in message.tool_calls if func_call["function"]["name"] != "select_tools"
                ]
                
                # If valid message, convert to OpenAI format (i.e. not select_tools)
                openai_messages.append(message)

            except:
                model_responses = message.content
                tool_call_ids = []
                openai_messages.append(message)


        # model_responses_message_for_chat_history = openai_messages
        
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": openai_messages, #TODO: Check if this is correct
            "tool_call_ids": tool_call_ids,
            "input_token": input_tokens,
            "output_token": output_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        # These two models have temperature fixed to 1
        # Beta limitation: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        temperature = 1 if "o1-preview" in self.model_name or "o1-mini" in self.model_name else self.temperature
        m = ChatOpenAI(model=self.model_name, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
        tool_selection_agent, initial_state = self.tool_search_client.get_tool_selection_agent(m)
        initial_state["messages"] = repr(inference_data["message"])
        api_response = tool_selection_agent.invoke(initial_state)

        return api_response

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        # Special handling for o1-preview and o1-mini as they don't support system prompts yet
        if "o1-preview" in self.model_name or "o1-mini" in self.model_name:
            for round_idx in range(len(test_entry["question"])):
                test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                    test_entry["question"][round_idx]
                )
                test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                    test_entry["question"][round_idx]
                )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data
