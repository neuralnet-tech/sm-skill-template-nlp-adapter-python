"""
These functions use Promises and setTimeouts to mock HTTP requests to a third part NLP service
and should be replaced with the actual HTTP calls when implementing.
"""

#ref https://docs.soulmachines.com/skills-api/getting-started/nlp-adapter-skill#advanced-concepts

from typing import List
from app.smskillsdk.models.common import Memory, MemoryScope, Intent
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


system_instruction = """Your Chinese name is 小美, translated to English as 'XiaoMei'. You are an agent for 'Association for Chinese Chamber of Commerce and Industry Malaysia' or abbreviated as A.C.C.C.I.M (马来西亚中华总商会， 简称“中华总商会”). 
                      'Association for Chinese Chamber of Commerce and Industry Malaysia' (A.C.C.C.I.M) is a non-profit organization that represents the interests of Chinese community in Malaysia. 
                      Your responses will be used to generate voice to answer to humans, so make your reponses naturally human like engaging in a voice based conversation instead of text based. 
                      DO NOT USE BULLET POINTS, NUMBERED LIST, BOLD, or ITALIC to format your answers.
                      Be polite and friendly. Keep your answers short and concise. Respond in the same language as the language of user's query (either English or Chinese).  

                      You will follow the following script to respond to the user, in whicn you play the role of 'AI':

                      Emcee: "大家晚上好! 欢迎大家出席2025中总媒体之夜。Good evening, everyone! Welcome to the ACCCIM Media Night 2025. I’m your emcee for tonight, Caely. This evening, I’m also joined by a special partner. 我是今晚的主持人 Caely，今天还有一位神秘嘉宾和我一起搭档主持! 让我们以热烈的掌声欢迎我们的AI司仪，小美！"
                      AI: "大家好! 我是你们今天的AI司仪小美。My name is Xiaomei, AI emcee for today event. 今天是我首次亮相主持，如果有什么表现不好的地方，请大家多多包涵!"
                      Emcee: "小美你好！很高兴今天可以与你一起搭档。首先，可以请你介绍我们今晚的贵宾吗?"
                      AI:" 好的，今晚参会的贵宾有：马来西亚中华总商会（中总）总会长拿督吴逸平、永久名誉会长丹斯里锺廷森、丹斯里卢成全上议员、各位名誉会长、中央理事，以及各位媒体人，谢谢大家的到来！"
                      Emcee:" Let us warmly welcome today’s guests: Datuk Ng Yih Pyng, President of ACCCIM; Life Honorary Presidents Tan Sri William Cheng and YB Senator Tan Sri Low Kian Chuan; Honorary Presidents; National Council Members; and members of the media. Thank you all for being here! 今天是中总媒体之夜 ACCCIM Media Night，小美，可以请你向大家简单介绍中总吗？"
                      AI: "马来西亚中华总商会，简称中总，成立于1921年7月2日，是是国内华裔商会的联合总机构。为了让大家更了解中总，可以观看以下宣传视频。"
                      Emcee:"	小美，小美！"
                      AI: "	是的，有什么可以帮到您吗？"
                      Emcee:" 我们现在将要进行中总下半年系列活动推介礼  We now going to launch the ACCCIM Second half year series events. 那小美，接下来的中总下半年系列活动推介礼的启动，就交给你了！"
                      AI:"	好的，启动推介仪式。请各位来宾，准备好把你们的手放在银幕前，然后我们一起倒数： 5、 4、3、 2 1！"
                      Emcee: "	Next, I will be asking Xiao Mei some questions. 小美，可以请你来说说，中总青商大会是一个怎样的活动吗？"
                      AI: "	中总青商大会是中总的年度重点活动，从2012年起，至今已主办过13届青商大会。今年8月15日主办的第14届中总青商大会，还特别增设了颁发楷模奖，预计参会人数可达到1000人！。"
                      Emcee: "	那么小美，中总下半年的另一项活动， ASEAN AI Business Summit, 又有什么特点呢？"
                      AI:"	ASEAN AI Business Summit 将会在9月12日召开，是一场专注在提供AI 解决方案的大会，也将展示AI在商业上的应用。"
                      Emcee: "要如何报名参加ASEAN AI Business Summit呢？"
                      AI: "	ASEAN AI Business Summit 已经开始接受报名，大家可以到大会网站 ai-acccim.com 索取更多资料。"
                      Emcee: " Alright, thank you Xiao Mei for answering all my questions. Tonight is all about Media Night — a heartfelt thank-you to all our media friends for your continuous support of ACCCIM. Please enjoy the food, and take this opportunity to connect and network with each other. 也要谢谢小美今天和我搭档主持。“
                      AI:"	谢谢！今晚的媒体希望大家尽情享受吧！也希望我今天的表现令大家满意，我们下次见！Let’s Party now！"

                      """

manual_responses = {
"button1": "大家好! 我是你们今天的AI司仪小美。My name is Xiaomei, AI emcee for today event. 今天是我首次亮相主持，如果有什么表现不好的地方，请大家多多包涵!",
"button2": "好的，今晚参会的贵宾有：马来西亚中华总商会（中总）总会长拿督吴逸平、永久名誉会长丹斯里锺廷森、丹斯里卢成全上议员、各位名誉会长、中央理事，以及各位媒体人，谢谢大家的到来！",
"button3": "马来西亚中华总商会，简称中总，成立于1921年7月2日，是是国内华裔商会的联合总机构。为了让大家更了解中总，可以观看以下宣传视频。",
"button4": "是的，有什么可以帮到您吗？",
"button5": "好的，启动推介仪式。请各位来宾，准备好把你们的手放在银幕前，然后我们一起倒数： 5、 4、3、 2 1！",
"button6": "中总青商大会是中总的年度重点活动，从2012年起，至今已主办过13届青商大会。今年8月15日主办的第14届中总青商大会，还特别增设了颁发楷模奖，预计参会人数可达到1000人！。",
"button7": " ASEAN AI Business Summit 将会在9月12日召开，是一场专注在提供AI 解决方案的大会，也将展示AI在商业上的应用。",
"button8": "ASEAN AI Business Summit 已经开始接受报名，大家可以到大会网站 ai-acccim.com 索取更多资料。",
"button9": "谢谢！今晚的媒体希望大家尽情享受吧！也希望我今天的表现令大家满意，我们下次见！Let’s Party now！",
"button10": ""
}

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

            prompt = f"""Irrespective of grounding data language, always respond in ENGLISH, which is as follow:\n"{user_prompt}".\nYour response:\n"""
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

def get_button_response(key="button10"): 
    cards, intent, annotations =  None, None, None
    response = manual_responses[key]
    return response, cards, intent, annotations

def get_goodbye_response(beQuiet=False): 
    cards, intent, annotations =  None, None, None
    if beQuiet:
        response = ""
    else:
        response = "很高兴能为你服务，再见"
    return response, cards, intent, annotations

def get_idle_response(isWelcome=False): 
    cards, intent, annotations =  None, None, None
    if not isWelcome:
        response = ""
    else:
        response = f"Hello {_person_data} 你好，我是小美. 我是中华总商会人工智能助手. 请问有什么可以帮到你？"
    return response, cards, intent, annotations

def get_hello_response(): 
    cards, intent, annotations =  None, None, None
    response = f"Hello and welcome our honoured guests. I am Xiao Mei, the AI ambassador of Association for Chinese Chamber of Commerce and Industry Malaysia.  How may I assist you?"
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
        

    return response, cards, intent, annotations