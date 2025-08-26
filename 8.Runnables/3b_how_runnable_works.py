# Creating Dummy LLM and Prompt Template Chain
# This demonstrates combining a prompt template with a dummy LLM into a single chain
# After this, you can simply pass the topic and get a simulated response


import random


# Component 1: Dummy LLM
class NakliLLM:
    def __init__(self):
        # Print message when object is initialized
        print('LLM Created')
    
    def predict(self, prompt):
        # Simulate LLM output using a list of predefined responses
        response_list = [
            'Lucknow Capital of UP',
            'T20 World Cup 2024 India',
            'Physics Based Neural Network'
        ]
        # Return one random response as a dictionary
        return {'response': random.choice(response_list)}


# Component 2: Dummy Prompt Template
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        # Store the template string with placeholders
        self.template = template
        # Store the list of variables expected in the template
        self.input_variables = input_variables
    
    def format(self, input_dict):
        # Replace placeholders in template with actual values from input_dict
        return self.template.format(**input_dict)


# Component 3: Combine LLM and Prompt Template
class NakliLLMChain:
    def __init__(self, llm, prompt):
        # Store LLM and prompt template objects
        self.llm = llm
        self.prompt = prompt
    
    def run(self, input_dict):
        # Step 1: Format the prompt with input values
        final_prompt = self.prompt.format(input_dict)
        print("Final Prompt:", final_prompt)
        # Step 2: Pass formatted prompt to the LLM and get response
        result = self.llm.predict(final_prompt)
        # Step 3: Return only the text part of the response
        return result['response']


# Usage Example
template = NakliPromptTemplate(
    template="Write a short note on India's {topic}",
    input_variables=['topic']
)

llm = NakliLLM()                     # Create dummy LLM object
chain = NakliLLMChain(llm, template) # Combine LLM with the prompt template

# Run the chain with input dictionary containing the topic 
result = chain.run({'topic': 'Peacock'})
print("Result:", result)
