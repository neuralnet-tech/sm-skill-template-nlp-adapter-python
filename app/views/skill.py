from fastapi import APIRouter
from operator import itemgetter, attrgetter
from smskillsdk.utils.memory import get_memory_value, set_memory_value
from ..services.fake_nlp_service import FakeNLPService

# Add these near the top of the file with other imports
from fastapi import Request, HTTPException
import sys

from ..mocks.gemini_agent import set_person_data


from smskillsdk.models.api import (
    InitRequest,
    SessionRequest,
    SessionResponse,
    ExecuteRequest,
    ExecuteResponse,
    Output,
    Variables,
)

router = APIRouter(
    tags=["Skill"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)


@router.post("/face-detection", status_code=200)
async def handle_face_detection(request: Request):
    try:
        data = await request.json()
        # Process the data
        person_data = data["person"]

        set_person_data(person_data)
        print("Received face detection data:", person_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/init", status_code=204)
async def init(request: InitRequest):
    """
    Init Endpoint
    https://docs.soulmachines.com/skills/api#tag/Init
    
    Runs when a DDNA Studio project is deployed with this Skill configured
    """

    # 1. Extract relevant data
    skill_config = request.config

    print("skill config body:", skill_config)

    # 1a. Extract relevant credentials from config
    #credentials = itemgetter("first_credentials", "second_credentials")(skill_config)
    credentials = ("me","you")

    # 2. Make request to third party service to initialize 
    # any configuration, data storage, or pre-training on the NLP service before executing this Skill
    fake_nlp_service = FakeNLPService(*credentials)
    fake_nlp_service.init_actions()


@router.post("/session", status_code=200, response_model=SessionResponse, response_model_exclude_unset=True)
async def session(request: SessionRequest) -> SessionResponse:
    """
    Session Endpoint
    https://docs.soulmachines.com/skills/api#tag/Session
    
    Runs before the very first interaction between a user and a DP using this Skill
    Note that if this endpoint is mapped in skill definition file, the execute endpoint
    will not contain config in the SessionRequest
    """
  
    # 1. Extract relevant data
    session_id, skill_config, skill_memory = attrgetter("sessionId", "config", "memory")(request)

    # 1a. Extract relevant credentials from config
    #credentials = itemgetter("first_credentials", "second_credentials")(skill_config)
    credentials = ("me","you")

    # 2. Make request to third party service to initialize session-specific resources
    fake_nlp_service = FakeNLPService(*credentials)

    # 3. Extract relevant response data from the third party service
    memory_resources = fake_nlp_service.init_session_resources(session_id)
    memory_credentials = fake_nlp_service.persist_credentials(session_id)

    skill_memory.extend(memory_resources)
    set_memory_value(memories=skill_memory, **memory_credentials)
    
    # 4. Construct SM-formatted response body
    response = SessionResponse(memory=skill_memory)

    return response

@router.post("/execute", status_code=200, response_model=ExecuteResponse, response_model_exclude_unset=True)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute Endpoint

    https://docs.soulmachines.com/skills/api#tag/Execute
    Runs when user input is forwarded to this Skill
    Note that if the session endpoint is mapped in skill definition file, this endpoint
    will not contain config in the SessionRequest
    """

    # 1. Extract relevant data
    user_intent, skill_config, skill_memory, context = attrgetter("intent", "config", "memory", "context")(request)

    print("Request (intent, skill_config, memory. context):", user_intent, skill_config, skill_memory, context )

    # 1a. when using stateless skill, extract relevant credentials from config
    # credentials = itemgetter("first_credentials", "second_credentials")(skill_config)

    # 1b. when using stateful skill, extract relevant credentials elsewhere (eg. memory) as config will not be present here
    _, credentials = get_memory_value(memories=skill_memory, key="credentials")

    # 3. Make request to third party service
    fake_nlp_service = FakeNLPService(*credentials)

    # 2. Extract user input
    user_input = request.text

    # 4. Extract relevant response data from the third party service
    spoken_response, cards, intent, annotations = fake_nlp_service.send(user_input)

    # 5. Construct SM-formatted response body
    if (annotations is not None) and (cards is not None):
        #print("show card1")
        variables = Variables(public=cards, **annotations)
    elif cards is not None:
        #print("show card2")
        variables = Variables(public=cards)
    else:
        #print("show card3")
        variables = None

    output = Output(
        intent=intent,
        text=spoken_response,
        variables=variables
    )

    response = ExecuteResponse(
        output=output,
        endConversation=False,
    )

    return response

@router.delete("/delete/{project_id}", status_code=204)
async def delete(project_id: str):
    """
    Delete Endpoint
    https://docs.soulmachines.com/skills/api#tag/Delete
        
    Use this endpoint to implement any clean-up for a Skill when it is no longer used by a project.
    
    Skills which make use of the init endpoint may find the delete endpoint particularly useful for
    cleaning up any long-running tasks or stored data associated with the provided projectId.
    
    The delete endpoint will be called every time a DDNA Studio project using this Skill is deleted.
    It will also be called when a project using the Skill removes it, and is then redeployed.
    """

   # Initiate any cleaning up of data or processes for this project
    print(f"Cleaned up project - {project_id}")
