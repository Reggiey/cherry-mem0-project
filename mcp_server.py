# mcp_server.py
import os
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from mem0 import Memory
import uvicorn
from types import SimpleNamespace

# 辅助函数依然需要，因为 vector_store 内部还是嵌套的
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

        # --- 终极方案：只提供必需的配置，让库处理其他所有默认值 ---
        config_dict = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": storage_path
                }
            }
        }
        
        config_object = dict_to_namespace(config_dict)
        
        print("🔧 Passing MINIMAL config, letting mem0 use its defaults...")
        # 你需要设置你的 OpenAI API 密钥作为环境变量
        # 在 Render.com 的 Environment 选项卡中，添加一个环境变量
        # Key: OPENAI_API_KEY
        # Value: sk-YourActualApiKey
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ WARNING: OPENAI_API_KEY environment variable not set. Default mem0 LLM may fail.")

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
                    raise HTTPException(status_code=400, detail={"error": "Missing 'content' in payload for 'add' action."})
                # 注意：mem0 的 add 方法现在可能会调用 LLM 进行事实提取
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

