# ---------------------------------
# Creating Dummy (LLM + Prompt Template) Use Case Chain
# ---------------------------------
# This is a simple simulation of how a PromptTemplate and an LLM
# can be connected together into a small pipeline.
# We are not using any real LLM here, just dummy classes.

import random


# --------------------------
# Component 1: Dummy LLM
# --------------------------
class NakliLLM:
    def __init__(self):
        # Message to show when the object is created
        print('LLM Created')   
    
    def predict(self, prompt):
        """
        Simulate LLM response.
        Instead of real AI, we just return one random answer
        from a predefined list.
        """
        response_list = [
            'Lucknow Capital of UP',
            'T20 World Cup 2024 India',
            'Physics Based Neural Network'
        ]
        # Randomly choose one response
        return {'response': random.choice(response_list)}


# ✅ Create dummy LLM instance and test it
llm = NakliLLM()
prompt = "What is Capital of UP"
print(llm.predict(prompt=prompt))  # Call predict with a prompt


# --------------------------
# Component 2: Dummy Prompt Template
# --------------------------
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        # Save the template string (with placeholders like {topic})
        self.template = template
        # Save the list of variables required to fill the template
        self.input_variables = input_variables   
    
    def format(self, input_dict):
        """
        Fill the template with actual values from input_dict.
        Example: 
        template = "Write about India's : {topic}"
        input_dict = {'topic': 'Wildlife'}
        Result → "Write about India's : Wildlife"
        """
        return self.template.format(**input_dict)


# ✅ Create dummy prompt template and test it
template = NakliPromptTemplate(
    template="Write about India's : {topic}",
    input_variables=['topic']
)
print(template.format({'topic': 'Wildlife'}))  # Replace {topic} with "Wildlife"


# --------------------------
# Component 3: Combine PromptTemplate with Dummy LLM
# --------------------------

# Step 1: Create a new prompt template
templateNakli = NakliPromptTemplate(
    template="Write notes on India's: {topic}",
    input_variables=['topic']
)

# Step 2: Format the template with a real input
# Input dictionary → {'topic': 'BushFires in India'}
# This will give: "Write notes on India's: BushFires in India"
prompt = templateNakli.format({'topic': 'BushFires in India'})

# Step 3: Create LLM instance again
llm = NakliLLM()

# Step 4: Pass the formatted prompt to LLM and get response
result = llm.predict(prompt)

# ✅ Final Output (simulated LLM response)
print(result)
