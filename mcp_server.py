# mcp_server.py
import os
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from mem0 import Memory
import uvicorn
from types import SimpleNamespace

# --- This helper is still needed for the nested vector_store config ---
def dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    converted_dict = {k: dict_to_namespace(v) for k, v in d.items()}
    return SimpleNamespace(**converted_dict)

# --- Pydantic Models for Type Safety ---
class MCPState(BaseModel):
    user_id: Optional[str] = None

class InvokePayload(BaseModel):
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)

# --- The Core MCP Component Logic ---
class Mem0MCPComponent:
    def __init__(self):
        storage_path = "/tmp/mem0_storage"
        os.makedirs(storage_path, exist_ok=True)
        print(f"💾 Using TEMPORARY storage at: {storage_path}.")

        # --- THE DEFINITIVE CONFIGURATION ---
        # This configuration satisfies all of the library's initialization quirks:
        # 1. Provides `vector_store` for our custom path.
        # 2. PROVIDES the `custom_..._prompt` keys as `None` to prevent the first type of AttributeError.
        # 3. OMITS `llm` and `embedder` keys entirely, so the library creates its own defaults
        #    instead of crashing on `None.provider`.
        config_dict = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": storage_path
                }
            },
            # These keys MUST exist, even if they are None.
            "custom_fact_extraction_prompt": None,
            "custom_update_memory_prompt": None,
            "custom_summarization_prompt": None,
        }
        
        config_object = dict_to_namespace(config_dict)
        
        print("🔧 Passing the definitive 'hybrid' config to mem0...")

        # Reminder: You still need the OPENAI_API_KEY environment variable in Render
        # for the default LLM and embedder to work.
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ WARNING: OPENAI_API_KEY env var not set. mem0 will likely fail.")
        
        self.mem0 = Memory(config=config_object)
        
        self.state = MCPState()
        print("✅✅✅ Mem0 MCP Component Initialized SUCCESSFULLY!")

    def get_state(self) -> MCPState:
        return self.state

    def update_state(self, new_state: MCPState):
        self.state = new_state
        return {"status": "success", "new_state": self.state.dict()}

    def invoke(self, invoke_data: InvokePayload) -> Dict[str, Any]:
        action = invoke_data.action
        payload = invoke_data.payload
        user_id = self.state.user_id

        if not user_id:
            raise HTTPException(status_code=400, detail={"error": "user_id is not set. Please update state first."})

        try:
            if action == "add":
                content = payload.get("content")
                if not content:
                    raise HTTPException(status_code=400, detail={"error": "Missing 'content'."})
                self.mem0.add(content=content, user_id=user_id)
                return {"result": "Memory added successfully."}

            elif action == "search":
                query = payload.get("query")
                if not query:
                    raise HTTPException(status_code=400, detail={"error": "Missing 'query'."})
                results = self.mem0.search(query=query, user_id=user_id)
                return {"result": results}

            else:
                raise HTTPException(status_code=400, detail={"error": f"Unknown action: {action}"})
        except Exception as e:
            print(f"🔥 Error during invoke: {e}")
            raise HTTPException(status_code=500, detail={"error": str(e)})

# --- FastAPI App Setup ---
app = FastAPI(title="Mem0 MCP Server")
component = Mem0MCPComponent()

@app.post("/get_state")
def get_state_endpoint():
    return component.get_state()

@app.post("/update_state")
def update_state_endpoint(state: MCPState):
    return component.update_state(state)

@app.post("/invoke")
def invoke_endpoint(invoke_data: InvokePayload):
    return component.invoke(invoke_data)

# --- Run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

