"""
These functions use Promises and setTimeouts to mock HTTP requests to a third part NLP service
and should be replaced with the actual HTTP calls when implementing.
"""
from typing import List
from app.smskillsdk.models.common import Memory, MemoryScope, Intent
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

googlesearch_tool = types.Tool(google_search=types.GoogleSearch())


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
    #response_mime_type = "application/json",
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
        #reponse_dict = json.loads(reponse_dict)
        #response = reponse_dict['response_text'] #+ " https://www.youtube.com/watch?v=Bhkm6fZMJcI"
        response = reponse_dict

    except Exception as e:
        print("error in reponse error decoding:",e)
        



    return response, cards, intent, annotations
