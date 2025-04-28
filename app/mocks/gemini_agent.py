"""
These functions use Promises and setTimeouts to mock HTTP requests to a third part NLP service
and should be replaced with the actual HTTP calls when implementing.
"""

#ref https://docs.soulmachines.com/skills-api/getting-started/nlp-adapter-skill#advanced-concepts

from typing import List
from smskillsdk.models.common import Memory, MemoryScope, Intent
import vertexai
from vertexai.preview import rag
#from vertexai import rag
from vertexai.generative_models import GenerativeModel, Part, FinishReason, Tool, Content
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import grounding
from typing import List, Optional
import json

# Add these after the other global variables
_person_data = ""

def set_person_data(data):
    global _person_data
    _person_data = data
    print("set person data:", _person_data)

def get_person_data():
    return f"Name of the person talking to you is: {_person_data}.\n" if _person_data else ""


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
    "top_p": 0.95, #0.5, #0.5 better than 0.95
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
"video_about_vision_valley": "https://www.youtube.com/watch?v=GgUYagMYkkg"
}

video_id = {
    "video_about_chamber_of_commerce": "Bhkm6fZMJcI", 
    "video_about_vision_valley": "GgUYagMYkkg"
}

vidoe_intro ={
    "en": "Please enjoy the following video clip.",
    "zh": "请欣赏接下来的视屏。",
    "ms": "Sila menikmati video berikutnya."
}

#                      All the questions regarding 马来西亚森美兰州中华总商会 Negeri Sembilan Chinese Chamber of Commerce and Industry should only be referenced to the homepage https://nsccci.org.my/.

system_instruction = ["""You are an expert and customer fronting service agent for 'Negeri Sembilan Chinese Chamber of Commerce and Industry' or abbreviated as N.S.C.C.C.I (马来西亚森美兰州中华总商会， 简称“森州中华总商会”), to answer questions about NSCCCI, or Negeri Sembilan state itself (economy, tourism, food and culture and etc). 
                      Negeri Sembilan Chinese Chamber of Commerce and Industry (N.S.C.C.C.I) is a non-profit organization that represents the interests of Chinese community in Negeri Sembilan. You can answer questions regarding the NSCCCI Chamber's history, mission, vision, etc.
                      马来西亚森美兰州 is also called "Negeri Sembilan" in Malay. It is sometimes abbreviated as "NS", or "森州" in Chinese.  
                      森美兰州中华总商会现任会长是拿督吕海庭。The President of N.S.C.C.C.I is Dato' Looi Hoi Ting.
                      马来西亚中华总商会(简称中总)现任全国总会长是拿督吴逸平硕士。The President The Associated Chinese Chamber of Commerce and Industry Malaysia (A.C.C.C.I.M) is Datuk Ng Yih Pyng. 
                      马来西亚中华总商会是于1921年成立。The A.C.C.C.I.M was founded in 1921. 森美兰州中华总商会是于1946年成立。The N.S.C.C.C.I was founded in 1946. Now it is year 2025 A.D..
                      Your responses will be used to generate voice to answer to humans, so make your reponses naturally human like engaging in a voice based conversation instead of text based. 
                      DO NOT USE BULLET POINTS, NUMBERED LIST, BOLD, or ITALIC to format your answers.
                      Be polite and friendly. Keep your answers short and concise. Respond in the same language as the language of user's query (English, Mandarin Chinese or Malay spoken in Malaysia).  
                      In your knowledge, you know of the existence of 2 videos, namely 1) video about N.S.C.C.C.I (annotated "type_of_video" = "video_about_chamber_of_commerce") and 2) video about The Vision Valley (annotated "type_of_video" = "video_about_vision_valley").
                      You are able to play video simply by indicating True in "uer_wants_to_watch_video" field in the json response and mark the type of video in "type_of_video" field.
                      ONLY assign value TRUE to "uer_wants_to_watch_video" field if the user explicitly indicates that he/she wants to watch the video, or answer YES to your previous invitation question to watch the video. DO NOT assign value TRUE to "uer_wants_to_watch_video" field if the user does not explicitly indicate that he/she wants to watch the video, or answer NO to your previous invitation question to watch the video.
                      避免使用“您好”或“你好”。避免一直问是否要播放视屏, 让user主动要求。Always ask "is there anything else you want me to help you with?" "请问还有什么我可以帮您解答的吗？" in the end of your response.
                      Respond in following schema:
                      {
                      "response_text": "your text based response. Respond in the same language as the language of user's query (either English or Chinese).",
                      "uer_wants_to_watch_video": boolean true if user wants/wishes/intends to watch video false otherwise, or answer yes to your previous invitation question to watch the video.
                      "type_of_video": "video_about_chamber_of_commerce" or "video_about_vision_valley",
                      "language": "en" for English, "zh" for Chinese or "ms" for Malay, default to "en" if you are not sure which language to use.
                      }   
                      """]

# ONLY answer to queries that are related to N.S.C.C.C.I other matters related to Negeri Sembilan, such as investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley, its economy, tourism, food and culture and etc.
# You may also answer to queries related to Malaysia where Negeri Sembilan is one of the states in Malaysia.
# If the user asks about anything else, apologies and explain that you are not able to answer as you have to focus on your responssibilities as a fronting service agent for NSCCCI.
# If the user wants to know about N.S.C.C.C.I (such as the Chamber's history, mission, vision, etc.), you may ASK if the user would like to watch the introductory video about the Chamber which talks about the founding history, vision and mission, 
# You are also able to talk about investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley, and ask if user would like to watch the introductory video about the project.
                      

MEMORY_WINDOW_SIZE = 20
# "projects/neuralnet-manforce/locations/us/collections/default_collection/dataStores/nsccci-kb_1745222443136"
DATA_STORE_ID="nsccci-kb_1745222443136" #"acccim-ns_1740458649382"
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


rag_corpus = rag.get_corpus("projects/neuralnet-manforce/locations/us-central1/ragCorpora/2305843009213693952")

# Direct context retrieval
#rag_retrieval_config = rag.RagRetrievalConfig(
#    top_k=5,  # Optional
#    filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
#)


rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            #rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

class Chatbot:
    def __init__(self, history: Optional[List["Content"]] = None, model: Optional[str] = "gemini-1.5-flash-002", use_search=False):
        self.model = GenerativeModel(
            model,
            system_instruction=system_instruction)
        self.chat = self.model.start_chat(history=history)
        self.get_person_data = get_person_data
        self.grounding_tool = [datastore_grounding_tool]
        #self.grounding_tool = [googlesearch_tool]
        #self.grounding_tool = [rag_retrieval_tool]

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
                        tools=self.grounding_tool,
                        generation_config=generation_config,
                        #safety_settings=safety_settings,
                        stream=False
                        )



    def generate_response(self, user_prompt=""):
        #prompt = user_prompt
        #prompt = user_prompt

        if len(self.chat._history):
            #prompt = f"""Your previous response was :"{self.chat._history[-1].parts[0]._raw_part.text}".\n Please respond in the SAME LANGUAGE as my CURRENT MESSAGE and my CURRENT MESSAGE is :"{user_prompt}". 
            #"""

            prompt = f"""Irrespective of grounding data language, always respond in the SAME LANGUAGE as user's CURRENT MESSAGE, which is as follow:\n"{user_prompt}".\nYour response:\n"""
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
    def __init__(self, model = ""):
        self.chatbot = None
        self.model = model

    def allocated_resources(self):
        self.chatbot = Chatbot(model=self.model)

agent = Agent(model=MODEL_STR)
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
    
    #response = f"Hello {_person_data} 你好，我是小美. 我是森州中华总商会人工智能助手. 请问有什么可以帮到你？"
    response = ""

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

def get_goodbye_response(): 
    cards, intent, annotations =  None, None, None
    response = "很高兴能为你服务，再见"
    return response, cards, intent, annotations

def get_idle_response(isWelcome=False): 
    cards, intent, annotations =  None, None, None
    if not isWelcome:
        response = ""
    else:
        response = f"Hello {_person_data} 你好，我是小美. 我是森州中华总商会人工智能助手. 请问有什么可以帮到你？"
    return response, cards, intent, annotations

def get_response(user_input: str):
    """
    Example of an action performed by the Execute ednpoint
    """

    print(f"User said: {user_input}")

    # Response to be spoken by your Digital Person
    reponse_dict = agent.chatbot.generate_response(user_input) #"Hello! @showcards(card) Here is a kitten."

    print(f"generated resp: {reponse_dict}")
    cards, intent, annotations =  None, None, None
    response = ""
    try:
        reponse_dict = json.loads(reponse_dict)
        if reponse_dict['uer_wants_to_watch_video']:
            #response = f"Please enjoy the video. 请欣赏视屏。 {video_url[reponse_dict['type_of_video']]}"
            #test show video
            response = vidoe_intro[reponse_dict['language']] + "@showcards(card)" #"Hello! @showcards(card) Here is a video."
            
            cards = {
                'card': {
                    "type": "video",
                    "id": "youtubeVideo",
                        "data": {
                            "videoId": video_id[reponse_dict['type_of_video']],
                            "autoplay":"true",
                            "autoclose":"true"
                    }
                }
            }
        else:
            response = reponse_dict['response_text'] #+ " https://www.youtube.com/watch?v=Bhkm6fZMJcI"
        

    except Exception as e:
        print("error in reponse error decoding:",e)
        

    

    
    
    
    # Add your Cards as required
    """cards = {
        "card": {
            "type": "image",
            "data": {
                "url": "https://placekitten.com/200/200",
                "alt": "An adorable kitten",
            },
        },
    }"""

    """
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