from fastapi import HTTPException
#from ..mocks.mock_request import mock_get_response, mock_init_resources, mock_init_actions
from ..mocks.gemini_agent import get_response, init_resources, init_actions, get_welcome_response, get_goodbye_response, get_idle_response, get_hello_response
#from ..mocks.gemini_agent_2 import get_response, init_resources, init_actions, get_welcome_response
from smskillsdk.models.common import MemoryScope, Intent


_fake_nlp_state = "idle"

def get_fake_nlp_state():
    return _fake_nlp_state
def set_fake_nlp_state(state):
    global _fake_nlp_state
    _fake_nlp_state = state


class FakeNLPService:
    first_credentials: str
    second_credentials: str


    def __init__(self, first_credentials, second_credentials):
        self.first_credentials = first_credentials
        self.second_credentials = second_credentials
        self.__authenticate()
        self.get_fake_nlp_state = get_fake_nlp_state
        self.set_fake_nlp_state = set_fake_nlp_state

  
    def __authenticate(self):
        """
        Example of using credentials to authenticate
        """

        if (not (self.first_credentials and self.second_credentials)):
          raise HTTPException(status_code = 401, detail = "Unauthenticated")
        print("Authenticated!")

    def init_actions(self):
        """
        Example of initializing Skill-specific actions on third party NLP call
        """

        return init_actions()

    def init_session_resources(self, session_id: str):
        """
        Example of initializing resources with third party NLP call 
        """
        
        return init_resources(session_id)


    def persist_credentials(self, session_id: str):
        """
        Example of persisting credentials during session endpoint with third party NLP call 
        """

        credentials = {
          "first_credentials": self.first_credentials, "second_credentials": self.second_credentials
        }

        credentials_memory = {
          "key": "credentials",
          "value": credentials,
          "session_id": session_id,
          "scope": MemoryScope.PRIVATE,
        }

        return credentials_memory

    def send(self, user_input):
        """
        Example of sending input to the third party NLP call 
        """
        print(f"State: {self.get_fake_nlp_state()} User said: {user_input}")

        if user_input == "Welcome":
            self.set_fake_nlp_state("idle")
            return get_idle_response(isWelcome=False)
        elif user_input == "Mayday1234":
            self.set_fake_nlp_state("active")
            return get_hello_response()
        else:
            #manage state here
            if self.get_fake_nlp_state() == "idle":
                # check if user_input contains wake words [""]
                if ("小美" in user_input or "小米" in user_input) and "你好" in user_input:
                    self.set_fake_nlp_state("active")
                    return get_idle_response(isWelcome=True)
                else:
                    return get_idle_response(isWelcome=False)
            elif self.get_fake_nlp_state() == "active":
                # check if user_input contains wake words ["小美", "你好"]
                if ("小美" in user_input or "小米" in user_input) and "再见" in user_input:
                    self.set_fake_nlp_state("idle")
                    return get_goodbye_response()
                return get_response(user_input)
        


        #if user_input == "Welcome":
        #    return get_welcome_response()
        #else:
        #    return get_response(user_input)

        