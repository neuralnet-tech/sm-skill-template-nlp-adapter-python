#for pydantic > 2

import sys
import importlib.util
import importlib.machinery
import types
from typing import List
from pydantic import RootModel

# Define our fixed class
class FixedConversationHistory(RootModel):
    root: List  # Will be properly typed later

# Function to patch the module
def patch_api_module():
    # Get the spec
    spec = importlib.util.find_spec('smskillsdk.models.api')
    if not spec:
        raise ImportError("Module smskillsdk.models.api not found")
    
    # Create a new module object
    module = types.ModuleType('smskillsdk.models.api')
    
    # Add it to sys.modules early
    sys.modules['smskillsdk.models.api'] = module
    
    # Load the source code as a string
    source = spec.loader.get_source('smskillsdk.models.api')
    
    # Modify the source code to remove or fix the problematic class
    modified_source = source.replace(
        "class ConversationHistory(BaseModel):",
        "# Original ConversationHistory commented out"
    ).replace(
        "    __root__: List[HistoryItem]",
        "    # __root__: List[HistoryItem]"
    )
    
    # Compile and execute the modified source
    code = compile(modified_source, spec.origin, 'exec')
    exec(code, module.__dict__)
    
    # Now inject our fixed class
    module.ConversationHistory = FixedConversationHistory
    
    return module 

# Patch the module before anyone else imports it
patched_module = patch_api_module()

from fastapi import FastAPI
from .views import skill

def create_app():
  app = FastAPI(
    title="Skill Hooks",
    version="0.0.0",
    description="Your description goes here",
  )

  # Add routes
  app.include_router(router=skill.router)

  return app