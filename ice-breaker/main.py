from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
load_dotenv()

def ice_breaker_with(name:str)->str:
  linkedin_username = linkedin_lookup_agent(name=name)
  linkedin_data= scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
  summary_template = """

        given the information {information} about a person from i want you to create:
        1.a short summary 
        2. two interesting facts about them

        """
    
  summary_prompt_template = PromptTemplate(input_variables="information",template=summary_template)

  llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
  chain = summary_prompt_template | llm
  res = chain.invoke(input={"information": linkedin_data})
  print(res)


if __name__ == "__main__":
  print("Ice Breaker Enter")
  ice_breaker_with(name="jyoti pal esstart")

  
 

