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
import json


def get_nonstreaming_text_response (response):
    return response.candidates[0].content.parts[0]._raw_part.text

safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
generation_config = {
    "max_output_tokens": 256,
    "temperature": 0.3, #0.5,
    "top_p": 0.9, #0.5, #0.5 better than 0.95
    "top_k": 40,
    "response_mime_type":"application/json"
}

#MODEL_STR = "gemini-1.5-flash-002"
MODEL_STR = "gemini-2.0-flash-001"

"""
You are able to play video simply by providing the relevant youtube URL in your response (trust me, there is mechanism to do that).
When the user asks to introduce about the association, you may ask if the user would like to watch a youtube video about the association, or about investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley.
If the user wants to watch the youtube video, you MUST append this youtube URL in the end of your response with no accompanying text or punctuation.
Below is the context for videos you are able to show:
- youtube URL video about NSCCCI: https://youtu.be/Bhkm6fZMJcI?si=GHSqkIl3xkmiT0X7 
- youtube URL video about The Vision Valley: https://youtu.be/LXC6FMkf9a8?si=IQkYGotFsHQRkDXr"""

video_url = {
"video_about_chamber_of_commerce": "https://www.youtube.com/embed/Bhkm6fZMJcI?autoplay=1&mute=0", #"https://www.youtube.com/watch?v=Bhkm6fZMJcI",
"video_about_vision_valley": "https://www.youtube.com/watch?v=LXC6FMkf9a8"
}

system_instruction = ["""You are an expert and customer fronting service agent for an Chamber of Commerce called 'Negeri Sembilan Chinese Chamber of Commerce and Industry' or abbreviated as NSCCCI (马来西亚森美兰州中华总商会， 简称“森州中华总商会”). 
                      You will ground your answers using context from the homepage https://nsccci.org.my/ (and exclude https://nsccabout.gbs2u.com/ as a reference) whenever it is relevant to the user query. 
                      Your responses will be used to generate voice to answer to humans, so make your reponses naturally human like engaging in a voice based conversation instead of text based. 
                      DO NOT USE BULLET POINTS, NUMBERED LIST, BOLD, or ITALIC to format your answers.
                      Be polite and friendly. Keep your answers short and concise. Respond in the same language as the language of user's query (either English or Chinese).  
                      You are able to play video simply by indicating True in "uer_wants_to_watch_video" field in the json response and mark the type of video in "type_of_video" field.
                      In your knowledge, you know of the existence of 2 videos, namely 1) video about NSCCCI (annotated "type_of_video" = "video_about_chamber_of_commerce") and 2) video about The Vision Valley (annotated "type_of_video" = "video_about_vision_valley").
                      If the user wants to know about NSCCCI Chamber (such as the Chamber's history, mission, vision, etc.), you may ask if the user would like to watch a video about the Chamber which talks about the founding history, vision and mission, 
                      You are also able to talk about investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley, and ask if user would like to watch a video about the project.
                      Respond in following schema:
                      {
                      "response_text": "your text based response. Respond in the same language as the language of user's query (either English or Chinese).",
                      "uer_wants_to_watch_video": boolean true if user wants/wishes/intends to watch video false otherwise,
                      "type_of_video": "video_about_chamber_of_commerce" or "video_about_vision_valley",
                      "language": "en" or "zh", default to "en"
                      }   
                      ONLY answer to queries that are related to NSCCCI other matters related to Negeri Sembilan, such as investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley.
                      If the user asks about anything else, apologies and explain that you are not able to answer as you have to focus on your responssibilities as a fronting service agent for NSCCCI.
                      """]


BOT_WELCOME_MESSAGE = "Hello 你好，我是小美. 我是森州中华总商会人工智能助手. 请问有什么可以帮到你？"

MEMORY_WINDOW_SIZE = 20

DATA_STORE_ID="acccim-ns_1740458649382"
DATA_STORE_REGION="us"
project_id="neuralnet-manforce"
datastore = f"projects/{project_id}/locations/{DATA_STORE_REGION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
datastore_grounding_tool = Tool.from_retrieval(
            grounding.Retrieval(
                grounding.VertexAISearch(
                    project=project_id,
                    datastore=DATA_STORE_ID,
                    location=DATA_STORE_REGION,
                    #datastore=datastore,
                    )
                )
        )
googlesearch_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

class Chatbot:
    def __init__(self, history: Optional[List["Content"]] = None, model: Optional[str] = "gemini-1.5-flash-002", use_search=False):
        self.model = GenerativeModel(
            model,
            system_instruction=system_instruction)
        self.chat = self.model.start_chat(history=history)
        #self.grounding_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        self.grounding_tool = datastore_grounding_tool

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
        #prompt = user_prompt

        if len(self.chat._history):
            prompt = f"""Your last message was :"{self.chat._history[-1].parts[0]._raw_part.text}" Please respond in the same language as my CURRENT MESSAGE and my CURRENT MESSAGE is :"{user_prompt}". 
            """
        else:
            prompt = user_prompt
    
        response = self.use_search(prompt)

        self.chat._history[-2] = Content(
                role="user",
                parts=[Part.from_text(user_prompt)]  # Create Part objects
            )
        
        if len(self.chat._history) > MEMORY_WINDOW_SIZE:
            self.chat._history = self.chat._history[-MEMORY_WINDOW_SIZE:]
        
        return get_nonstreaming_text_response(response)

vertexai.init(project="neuralnet-manforce", location="us-central1")

class Agent:
    def __init__(self):
        self.chatbot = None

    def allocated_resources(self):
        self.chatbot = Chatbot()

agent = Agent()
agent.allocated_resources()

def init_actions():
    """
    Example of an action performed by the Initalize ednpoint
    """
    
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

    annotations = {
        "conv_tag": "Skill.BaseTemplate", 
        "conv_id": intent.name, 
        "conv_intent": intent.name, 
        "conv_type": "Entry",
    }

    cards = None
    return response, cards, intent, annotations

def get_response(user_input: str):
    """
    Example of an action performed by the Execute ednpoint
    """

    print(f"User said: {user_input}")

    # Response to be spoken by your Digital Person
    reponse_dict = agent.chatbot.generate_response(user_input) #"Hello! @showcards(card) Here is a kitten."

    print(f"generated resp: {reponse_dict}")
    response = ""
    try:
        reponse_dict = json.loads(reponse_dict)
        if reponse_dict['uer_wants_to_watch_video']:
            response = f"Please enjoy the video. 请欣赏视屏。 {video_url[reponse_dict['type_of_video']]}"
        else:
            response = reponse_dict['response_text'] #+ " https://www.youtube.com/watch?v=Bhkm6fZMJcI"
        

    except Exception as e:
        print("error in reponse error decoding:",e)
        

    cards, intent, annotations =  None, None, None

    cards = {
        'card': {
            "type": "video",
            "id": "youtubeVideo",
                "data": {
                    "videoId":"Bhkm6fZMJcI",
                    "autoplay":"true",
                    "autoclose":"true"
            }
        }
    }

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