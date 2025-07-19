# mcp_server.py
import os
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from mem0 import Memory
import uvicorn
from types import SimpleNamespace # <-- 关键导入！

# --- 辅助函数：将字典递归转换为 SimpleNamespace 对象 ---
def dict_to_namespace(d):
    """
    Recursively converts a dictionary to a SimpleNamespace object.
    This allows attribute access (e.g., config.key) instead of item access (e.g., config['key']).
    """
    if not isinstance(d, dict):
        return d
    
    # 递归转换所有嵌套的字典
    converted_dict = {k: dict_to_namespace(v) for k, v in d.items()}
    
    return SimpleNamespace(**converted_dict)

# --- Pydantic Models for Type Safety ---
class MCPState(BaseModel):
    user_id: Optional[str] = None

class InvokePayload(BaseModel):
    action: str  # "add", "search"
    payload: Dict[str, Any] = Field(default_factory=dict)

# --- The Core MCP Component Logic ---
class Mem0MCPComponent:
    def __init__(self):
        storage_path = "/tmp/mem0_storage"
        os.makedirs(storage_path, exist_ok=True)
        
        print(f"💾 Using TEMPORARY storage at: {storage_path}. Data will be lost on restart.")

        # --- 根本性修复：创建一个字典，然后将其转换为对象 ---
        # 这是我们原始的配置字典
        config_dict = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": storage_path
                }
            },
            "custom_fact_extraction_prompt": None
        }
        
        # 将字典转换为 mem0 库需要的对象格式
        config_object = dict_to_namespace(config_dict)
        
        # 将转换后的对象传递给 Memory
        print("🔧 Passing configuration object to mem0 library...")
        self.mem0 = Memory(config=config_object)
        
        self.state = MCPState()
        print("✅ Mem0 MCP Component Initialized successfully!")

    def get_state(self) -> MCPState:
        print(f"➡️ Getting state: {self.state.dict()}")
        return self.state

    def update_state(self, new_state: MCPState):
        print(f"🔄 Updating state from {self.state.dict()} to {new_state.dict()}")
        self.state = new_state
        return {"status": "success", "new_state": self.state.dict()}

    def invoke(self, invoke_data: InvokePayload) -> Dict[str, Any]:
        action = invoke_data.action
        payload = invoke_data.payload
        user_id = self.state.user_id

        print(f"🚀 Invoking action '{action}' for user_id '{user_id}' with payload: {payload}")

        if not user_id:
            raise HTTPException(status_code=400, detail={"error": "user_id is not set. Please update state first."})

        try:
            if action == "add":
                content = payload.get("content")
                if not content:
                    raise HTTPException(status_code=400, detail={"error": "Missing 'content' in payload for 'add' action."})
                self.mem0.add(content=content, user_id=user_id)
                return {"result": "Memory added successfully (temporarily)."}

            elif action == "search":
                query = payload.get("query")
                if not query:
                    raise HTTPException(status_code=400, detail={"error": "Missing 'query' in payload for 'search' action."})
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

