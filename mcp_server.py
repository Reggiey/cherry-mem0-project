# mcp_server.py
import os
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from mem0 import Memory
import uvicorn

# --- Pydantic Models for Type Safety ---
class MCPState(BaseModel):
    user_id: Optional[str] = None

class InvokePayload(BaseModel):
    action: str  # "add", "search"
    payload: Dict[str, Any] = Field(default_factory=dict)

# --- The Core MCP Component Logic ---
class Mem0MCPComponent:
    def __init__(self):
        # --- 免费套餐的妥协方案 ---
        # Render 免费套餐没有持久化磁盘，我们使用 /tmp 目录。
        # 警告：/tmp 目录下的所有数据会在服务重启或休眠后丢失！
        storage_path = "/tmp/mem0_storage"
        os.makedirs(storage_path, exist_ok=True)
        
        print(f"💾 Using TEMPORARY storage at: {storage_path}. Data will be lost on restart.")

        # --- 关键修改：使用配置字典来初始化 Memory ---
        config = {
            "vector_store": {
                "provider": "qdrant",  # mem0 默认的本地存储提供商
                "config": {
                    "path": storage_path # 在这里指定路径
                }
            }
        }
        self.mem0 = Memory(config=config) # 正确的初始化方式
        
        self.state = MCPState()
        print("✅ Mem0 MCP Component Initialized (in temporary mode).")

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
                # 在新版 mem0 中, add 方法返回添加的记忆对象，我们可以忽略它
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
# 在Render这样的生产环境中，我们依赖 Gunicorn/Uvicorn worker，所以这部分代码不会被执行
# 但保留它对于本地测试是好的
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

