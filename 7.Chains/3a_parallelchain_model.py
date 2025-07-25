from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1=ChatOpenAI()

model2=ChatAnthropic(model_name='claude-3.5 sonnet')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
  template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz ->{quiz}',
  input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
  'notes':prompt1 | model1| parser,
  'quiz': prompt2 | model2 | parser
 })

merge_chain=prompt3 | model1 | parser

chain=parallel_chain | merge_chain

text="""Linear regression is a type of supervised machine-learning algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. It assumes that there is a linear relationship between the input and output, meaning the output changes at a constant rate as the input changes. This relationship is represented by a straight line.

For example we want to predict a student's exam score based on how many hours they studied. We observe that as students study more hours, their scores go up. In the example of predicting exam scores based on hours studied. Here

Independent variable (input): Hours studied because it's the factor we control or observe.
Dependent variable (output): Exam score because it depends on how many hours were studied. """

# result=chain.invoke({'text':text})
# print(result)
chain.get_graph().print_ascii()