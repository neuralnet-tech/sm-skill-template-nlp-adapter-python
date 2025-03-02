"""
These functions use Promises and setTimeouts to mock HTTP requests to a third part NLP service
and should be replaced with the actual HTTP calls when implementing.
"""

#ref https://docs.soulmachines.com/skills-api/getting-started/nlp-adapter-skill#advanced-concepts

from typing import List
from smskillsdk.models.common import Memory, MemoryScope, Intent
import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Part, FinishReason, Tool, Content
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import grounding
from typing import List, Optional


def get_nonstreaming_text_response (response):
    return response.candidates[0].content.parts[0]._raw_part.text

safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.3, #0.5,
    "top_p": 0.9, #0.5, #0.5 better than 0.95
    "top_k": 40,
}

MODEL_STR = "gemini-1.5-flash-002"

system_instruction = ["""You are an expert and customer fronting service agent for an Association called NS Chinese Chamber of Commerce. 
                      You will ground your answers using context from the homepage https://nsccci.org.my/ (and exclude https://nsccabout.gbs2u.com/ as a reference) whenever it is relevant to the user query. 
                      Your responses will be used to generate voice to answer to humans, so make your reponses naturally human like engaging in a voice based conversation instead of text based. 
                       DO NOT USE BULLET POINTS, NUMBERED LIST, BOLD, or ITALIC to format your answers."""]


BOT_WELCOME_MESSAGE = "Hello! I am your virtual assistant 小美. How can I assist you today?"



class Chatbot:
    def __init__(self, history: Optional[List["Content"]] = None, model: Optional[str] = "gemini-1.5-flash-002", use_search=False):
        self.model = GenerativeModel(
            model,
            system_instruction=system_instruction)
        self.chat = self.model.start_chat(history=history)
        self.grounding_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

    """
    def use_rag_tool(self, user_prompt):
        return  self.chat.send_message(
                        user_prompt, 
                        tools=[tool],
                        generation_config=generation_config,
                        #safety_settings=safety_settings,
                        stream=False
                        )"""
    
    def use_search(self, prompt):
        return  self.chat.send_message(
                        #f"Contexts: {contexts}. Message from User: {user_prompt}", 
                        [prompt],
                        tools=[self.grounding_tool],
                        generation_config=generation_config,
                        #safety_settings=safety_settings,
                        stream=False
                        )



    def generate_response(self, user_prompt=""):
        #prompt = user_prompt
        prompt = user_prompt
    
        response = self.use_search(prompt)
        
        return get_nonstreaming_text_response(response)

vertexai.init(project="neuralnet-manforce", location="us-central1")

class Agent:
    def __init__(self):
        self.chatbot = None

    def allocated_resources(self):
        self.chatbot = Chatbot()

agent = Agent()

def init_actions():
    """
    Example of an action performed by the Initalize ednpoint
    """
    agent.allocated_resources()
    print("resource initialized. . .")


def init_resources(session_id: str) -> List[Memory]:
    """
    Example of an action performed by the Session ednpoint
    """

    private_memory = Memory(**{
        "session_id": session_id,
        "name": "private json memory",
        "value": { "example": "object" },
        "scope": MemoryScope.PRIVATE,
    })
    public_memory = Memory(**{
        "session_id": session_id,
        "name": "public string memory",
        "value": "This is to be persisted",
        "scope": MemoryScope.PUBLIC,
    })

    return [private_memory, public_memory]

def get_welcome_response():
    # standard welcome message
    response = BOT_WELCOME_MESSAGE

    intent = Intent(
        name="Welcome",
        confidence=1,
    )

    cards, annotations =  None, None
    return response, cards, intent, annotations

def get_response(user_input: str):
    """
    Example of an action performed by the Execute ednpoint
    """

    print(f"User said: {user_input}")

    # Response to be spoken by your Digital Person
    response = agent.chatbot.generate_response(user_input) #"Hello! @showcards(card) Here is a kitten."

    cards, intent, annotations =  None, None, None
    """
    # Add your Cards as required
    cards = {
        "card": {
            "type": "image",
            "data": {
                "url": "https://placekitten.com/200/200",
                "alt": "An adorable kitten",
            },
        },
    }

    # Add your Intent as required
    intent = Intent(
        name="Welcome",
        confidence=1,
    )
  
    # If applicable, add your conversation annotations to see metrics for your Skill on Studio Insights
    annotations = {
        "conv_tag": "Skill.BaseTemplate", 
        "conv_id": intent.name, 
        "conv_intent": intent.name, 
        "conv_type": "Entry",
    }"""

    return response, cards, intent, annotations