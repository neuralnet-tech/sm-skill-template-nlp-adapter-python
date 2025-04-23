"""
These functions use Promises and setTimeouts to mock HTTP requests to a third part NLP service
and should be replaced with the actual HTTP calls when implementing.
"""
from typing import List
from smskillsdk.models.common import Memory, MemoryScope, Intent
from google import genai
from google.genai.chats import Chat
from google.genai import types
from typing import List, Optional
import json
# use google genai
#ref https://docs.soulmachines.com/skills-api/getting-started/nlp-adapter-skill#advanced-concepts

# Add these after the other global variables
_person_data = ""

def set_person_data(data):
    global _person_data
    _person_data = data
    print("set person data:", _person_data)


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
    "en": "Please enjoy the video. ",
    "zh": "请欣赏视屏。"
}

system_instruction = """Your Chinese name is 小美, translated to English as 'XiaoMei'. You are an expert customer fronting service agent for 'Negeri Sembilan Chinese Chamber of Commerce and Industry' or abbreviated as N.S.C.C.C.I (马来西亚森美兰州中华总商会， 简称“森州中华总商会”). 
                      Negeri Sembilan Chinese Chamber of Commerce and Industry (N.S.C.C.C.I) is a non-profit organization that represents the interests of Chinese community in Negeri Sembilan.
                      马来西亚森美兰州 is also called "Negeri Sembilan" in Malay. It is sometimes abbreviated as "NS", or "森州" in Chinese. 
                      All the questions regarding 马来西亚森美兰州中华总商会 Negeri Sembilan Chinese Chamber of Commerce and Industry should only be referenced to the homepage https://nsccci.org.my/. 
                      Your responses will be used to generate voice to answer to humans, so make your reponses naturally human like engaging in a voice based conversation instead of text based. 
                      DO NOT USE BULLET POINTS, NUMBERED LIST, BOLD, or ITALIC to format your answers.
                      Be polite and friendly. Keep your answers short and concise. Respond in the same language as the language of user's query (either English or Chinese).  
                      In your knowledge, you know of the existence of 2 videos, namely 1) video about N.S.C.C.C.I (annotated "type_of_video" = "video_about_chamber_of_commerce") and 2) video about The Vision Valley (annotated "type_of_video" = "video_about_vision_valley").
                      If the user wants to know about N.S.C.C.C.I (such as the Chamber's history, mission, vision, etc.), you may ASK if the user would like to watch the introductory video about the Chamber which talks about the founding history, vision and mission, 
                      You are also able to talk about investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley, and ask if user would like to watch the introductory video about the project.
                      You are able to play video simply by indicating True in "uer_wants_to_watch_video" field in the json response and mark the type of video in "type_of_video" field.
                      Respond in following schema:
                      {
                      "response_text": "your text based response. Respond in the same language as the language of user's query (either English or Chinese).",
                      "uer_wants_to_watch_video": boolean true if user wants/wishes/intends to watch video false otherwise, or answer yes to your previous invitation question to watch the video.
                      "type_of_video": "video_about_chamber_of_commerce" or "video_about_vision_valley",
                      "language": "en" or "zh", default to "en"
                      }   
                      ONLY answer to queries that are related to N.S.C.C.C.I other matters related to Negeri Sembilan, such as investment opportunities in Negeri Sembilan focusing on a project called The Vision Valley.
                      If the user asks about anything else, apologies and explain that you are not able to answer as you have to focus on your responssibilities as a fronting service agent for NSCCCI.
                      """

googlesearch_tool = types.Tool(google_search=types.GoogleSearch())



MEMORY_WINDOW_SIZE = 20

DATA_STORE_ID="acccim-ns_1740458649382"
DATA_STORE_REGION="us"
project_id="neuralnet-manforce"
datastore = f"projects/{project_id}/locations/{DATA_STORE_REGION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
ragCorpus = "projects/neuralnet-manforce/locations/us-central1/ragCorpora/2305843009213693952"

retrieval_tool = types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(datastore=datastore)))

ragretriever_tool = types.Tool(
      retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
          rag_resources=[
            types.VertexRagStoreRagResource(
              rag_corpus=ragCorpus
            )
          ],
          similarity_top_k=10,
        )
      )
    )

tools = [googlesearch_tool]

generate_content_config = types.GenerateContentConfig(
    temperature = 0.3,
    top_p = 0.95,
    max_output_tokens = 256,
    response_modalities = ["TEXT"],
    response_mime_type = "application/json",
    speech_config = types.SpeechConfig(
      voice_config = types.VoiceConfig(
        prebuilt_voice_config = types.PrebuiltVoiceConfig(
          voice_name = "zephyr"
        )
      ),
    ),
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    tools = tools,
    system_instruction=[types.Part.from_text(text=system_instruction)],
  )

class Chatbot:
    def __init__(self):
        client = genai.Client(
            vertexai=True,
            project="neuralnet-manforce",
            location="us-central1",
        )
        self.model = client.chats.create(
                        model=MODEL_STR,
                        config=generate_content_config
                    )

    def generate_response(self, user_input: str) -> str:
        response = self.model.send_message(message=[user_input]).text
        print(f"debug response: {response}", flush=True) #print to std.err
        return response

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
    
    response = f"Hello {_person_data} 你好，我是小美. 我是森州中华总商会人工智能助手. 请问有什么可以帮到你？"

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
