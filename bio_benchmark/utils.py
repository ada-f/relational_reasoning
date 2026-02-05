from openai import AzureOpenAI, OpenAI
import anthropic
from google import genai
import time
from ete3 import Tree

class LLMCaller():
        def __init__(self, config):
            self.backend = config['backend']
            self.azure_openai_api_key = config.get('azure_openai_api_key')
            self.azure_openai_endpoint = config.get('azure_openai_endpoint')
            self.openai_api_key = config.get('openai_api_key')
            self.anthropic_api_key = config.get('anthropic_api_key')
            self.gemini_api_key = config.get('gemini_api_key')
            self.model = config['model']
        
        def call_openai(self, 
                        conversation: str, 
                        temp: float = 0.000000001, 
                        model = 'gpt-4o',
                        max_tokens: int = 1000):
                
                if type(conversation) == str:
                    conversation = [{"role": "user", "content": conversation}]

                openai_backend = self.backend
                while True:
                    if openai_backend == 'azure':
                        client = AzureOpenAI(
                                azure_endpoint = self.azure_openai_endpoint,
                                api_key=self.azure_openai_api_key,
                                api_version="2025-03-01-preview")

                        if model == 'o1':
                            response = client.chat.completions.create(
                                    model=model,
                                    messages = conversation,
                            )
                        else:
                            print("Launching LLM call")
                            start = time.time()
                            response = client.chat.completions.create(
                                    model=model,
                                    messages = conversation,
                                    # temperature=temp,
                                    # max_tokens=max_tokens,
                            )
                            llm_call_time = time.time() - start
                            print("LLM call finished, Time taken:", llm_call_time)

                    elif openai_backend == 'openai':
                        self.model = 'gpt-5.2'
                        client = OpenAI(
                            api_key=self.openai_api_key 
                        )
                        start = time.time()
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=conversation,
                            temperature=temp,
                            # max_tokens=max_tokens,
                        )
                        llm_call_time = time.time() - start
                        print("LLM call finished, Time taken:", llm_call_time)
                    
                    elif openai_backend == 'anthropic':
                        self.model = 'claude-opus-4-20250514'
                        client = anthropic.Anthropic(
                            api_key=self.anthropic_api_key
                        )
                        # Convert conversation format for Anthropic
                        # Anthropic expects messages without system role mixed in
                        system_prompt = None
                        anthropic_messages = []
                        for msg in conversation:
                            if msg['role'] == 'system':
                                system_prompt = msg['content']
                            else:
                                anthropic_messages.append({
                                    'role': msg['role'],
                                    'content': msg['content']
                                })
                        
                        print("Launching LLM call (Anthropic)")
                        start = time.time()
                        
                        kwargs = {
                            'model': self.model,
                            'max_tokens': max_tokens,
                            'messages': anthropic_messages,
                        }
                        if system_prompt:
                            kwargs['system'] = system_prompt
                        
                        response = client.messages.create(**kwargs)
                        llm_call_time = time.time() - start
                        print("LLM call finished, Time taken:", llm_call_time)
                        
                        # Extract text from Anthropic response
                        response_text = response.content[0].text
                        return response_text, llm_call_time
                    
                    elif openai_backend == 'gemini':
                        self.model = 'gemini-3-pro-preview'
                        client = genai.Client(api_key=self.gemini_api_key)
                        
                        # Convert conversation to Gemini format
                        gemini_contents = []
                        system_instruction = None
                        
                        for msg in conversation:
                            if msg['role'] == 'system':
                                system_instruction = msg['content']
                            elif msg['role'] == 'user':
                                gemini_contents.append({
                                    'role': 'user',
                                    'parts': [{'text': msg['content']}]
                                })
                            elif msg['role'] == 'assistant':
                                gemini_contents.append({
                                    'role': 'model',
                                    'parts': [{'text': msg['content']}]
                                })
                        
                        print("Launching LLM call (Gemini)")
                        start = time.time()
                        
                        # Build config
                        generate_config = {}
                        if system_instruction:
                            generate_config['system_instruction'] = system_instruction
                        
                        generate_config['thinking_config'] = {
                            'thinking_level': 'low'  # Can be 0-24576, or use 'none', 'low', 'medium', 'high'
                            }
                        
                        response = client.models.generate_content(
                            model=self.model,
                            contents=gemini_contents,
                            config=generate_config if generate_config else None
                        )

                        # import pdb; pdb.set_trace()
                        
                        llm_call_time = time.time() - start
                        print("LLM call finished, Time taken:", llm_call_time)
                        
                        response_text = response.text
                        return response_text, llm_call_time
                    
                    else:
                        raise ValueError(f"Invalid openai_backend: {openai_backend}")

                    if "I'm sorry, I can't assist with that" in response.choices[0].message.content or "I'm unable to view the image" in response.choices[0].message.content or "I'm unable to provide a definitive answer" in response.choices[0].message.content:
                            print("Failed to generate response, trying again")
                            continue
                    else:
                            response = response.choices[0].message.content
                            return response, llm_call_time

    
