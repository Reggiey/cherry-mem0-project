# mcp_server.py
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from mem0 import Mem0
import uvicorn

# --- Pydantic Models for Type Safety ---
class MCPState(BaseModel):
    user_id: Optional[str] = None

class InvokePayload(BaseModel):
    action: str  # "add", "search", "get_all"
    payload: Dict[str, Any] = Field(default_factory=dict)

# --- The Core MCP Component Logic ---
class Mem0MCPComponent:
    def __init__(self):
        # Render 会将持久化磁盘挂载到 /var/data 目录
        # 我们在这里创建一个子目录来存放数据库
        storage_path = "/var/data/mem0_storage"
        os.makedirs(storage_path, exist_ok=True)
        
        print(f"💾 Using persistent storage at: {storage_path}")
        self.mem0 = Mem0(vector_store_path=storage_path) # 指定数据库路径
        
        self.state = MCPState()
        print("✅ Mem0 MCP Component Initialized.")

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
                return {"result": "Memory added successfully."}

            elif action == "search":
                query = payload.get("query")
                if not query:
                    raise HTTPException(status_code=400, detail={"error": "Missing 'query' in payload for 'search' action."})
                results = self.mem0.search(query=query, user_id=user_id)
                # results 是一个字典列表，我们直接返回它
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
    # 我们将服务器运行在 8001 端口
    uvicorn.run(app, host="0.0.0.0", port=8001)


