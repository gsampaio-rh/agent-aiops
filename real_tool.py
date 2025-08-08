#!/usr/bin/env python3
"""
Real Tool Permission Demo — Rewritten

Goals of this rewrite:
- Keep the chat stream clean (only user prompts and a single, final assistant answer per turn)
- Render ALL tool-permission UI (analysis, permission ask, execution progress, results) OUTSIDE the chat
- Avoid appending meta/system/tool messages to the chat log
- Provide robust state management with explicit fields in st.session_state
- Preserve your existing services (OllamaService, WebSearchService) and logger utilities
- Maintain the original feature set (approve/deny/modify tool usage; retry; debug mode; reset)

Key differences vs. original:
1) Dedicated containers `permission_area` and `results_area` for all tool UI.
2) No assistant messages are added during the tool flow. Only after the flow completes we add ONE final assistant message.
3) Introduces `tool_context` state to pass tool results to the model for the final answer.
4) Removes the forced-tool behavior (demo override). The agent’s decision stands, but you can still modify/approve/deny.
5) Uses compact helpers for state transitions and rendering, with early returns to simplify control flow.
"""

import streamlit as st
import time
import uuid
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# External services (kept as-is, expected to exist in your project)
from services.ollama_service import OllamaService
from services.search_service import WebSearchService
from utils.logger import get_logger, request_context


# =============================
# Enums & Data Classes
# =============================
class PermissionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    MODIFIED = "modified"


class ToolExecutionStatus(Enum):
    NOT_STARTED = "not_started"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolRequest:
    id: str
    tool_name: str
    tool_description: str
    query: str
    reasoning: str
    original_user_query: str
    permission_status: PermissionStatus
    execution_status: ToolExecutionStatus
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


# =============================
# RealAgent (selection + tools)
# =============================
class RealAgent:
    """A real agent that thinks, reasons, and optionally uses tools with permission."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.ollama_service = OllamaService()
        self.search_service = WebSearchService()
        self.model = model
        self.logger = get_logger(__name__)

        # Available tools catalogue
        self.available_tools = {
            "web_search": {
                "name": "web_search",
                "description": (
                    "Search the web for current information, facts, news, and general knowledge. "
                    "Use this when you need up-to-date information or when the user asks about current events, "
                    "recent developments, or specific factual information."
                ),
                "function": self._execute_web_search,
            }
        }

        # System prompt for tool selection
        self.system_prompt = (
            "You are a helpful AI assistant. Analyze the user's query and respond with ONLY valid JSON - no other text.\n\n"
            "Available tools:\n- web_search: Search the web for current information, facts, news, and general knowledge\n\n"
            "Rules:\n"
            "- If the user asks about current events, recent news, weather, stock prices, or anything requiring up-to-date information: use web_search\n"
            "- If the user asks about general knowledge, explanations, how-to guides that don't require current data: answer directly\n"
            "- Always respond with ONLY valid JSON, no additional text\n\n"
            "For tool usage, respond with:\n"
            "{\"needs_tool\": true, \"tool_name\": \"web_search\", \"reasoning\": \"brief reason\", \"search_query\": \"simple search terms\"}\n\n"
            "For direct answers, respond with:\n"
            "{\"needs_tool\": false, \"direct_answer\": \"your answer here\"}\n\n"
            "Examples:\n"
            "User: \"What's the weather today?\" → {\"needs_tool\": true, \"tool_name\": \"web_search\", \"reasoning\": \"Current weather requires real-time data\", \"search_query\": \"current weather forecast\"}\n"
            "User: \"Explain Python loops\" → {\"needs_tool\": false, \"direct_answer\": \"Python loops allow you to repeat code...\"}\n\n"
            "Remember: ONLY return valid JSON, nothing else."
        )

    # -------- Query analysis (tool selection) --------
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        self.logger.info("Analyzing user query", query=user_query[:200])

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            full_response = ""
            for chunk in self.ollama_service.chat_stream(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            ):
                if chunk.get("content"):
                    full_response += chunk["content"]

            self.logger.info("Agent analysis completed", response_length=len(full_response))
            self.last_raw_response = full_response

            # Parse JSON strictly, with a tolerant pre-clean
            cleaned = full_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                json_content = cleaned[start:end]
                analysis = json.loads(json_content)

                if "needs_tool" not in analysis:
                    raise ValueError("Missing 'needs_tool' field")
                if analysis.get("needs_tool") and not analysis.get("tool_name"):
                    raise ValueError("Tool needed but no tool_name specified")

                return analysis

            raise ValueError("No valid JSON structure found")

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback heuristic
            self.logger.warning("Failed to parse JSON; using heuristic", error=str(e))
            q = user_query.lower()
            current_keywords = [
                "weather", "current", "today", "now", "latest", "recent", "news",
                "stock", "price", "forecast", "happening", "update", "this week",
                "this month", "2024", "2025",
            ]
            needs_search = any(k in q for k in current_keywords)
            if needs_search:
                return {
                    "needs_tool": True,
                    "tool_name": "web_search",
                    "reasoning": "Query likely requires fresh information beyond training data.",
                    "search_query": user_query,
                }
            return {
                "needs_tool": False,
                "direct_answer": (
                    "I can answer based on my knowledge. "
                    "(Note: analysis JSON parsing failed, so falling back to a direct answer path.)"
                ),
            }
        except Exception as e:
            self.logger.error("Error analyzing query", error=str(e))
            return {
                "needs_tool": False,
                "direct_answer": f"I hit an error analyzing your query: {str(e)}",
            }

    # -------- Tool: web_search --------
    def _execute_web_search(self, query: str) -> Dict[str, Any]:
        self.logger.info("Executing web search", query=query)
        try:
            result = self.search_service.search(query, provider="duckduckgo", max_results=5)
            if result.get("status") == "success" and result.get("results"):
                formatted = []
                for i, res in enumerate(result["results"][:5], 1):
                    formatted.append(
                        f"{i}. {res['title']}\n   {res['snippet']}\n   Source: {res['url']}"
                    )
                return {"success": True, "results": "\n\n".join(formatted), "metadata": result}
            return {"success": False, "error": result.get("error", "No results found")}
        except Exception as e:
            self.logger.error("Web search execution failed", error=str(e))
            return {"success": False, "error": str(e)}


# =============================
# UI helpers
# =============================

def get_permission_styles() -> str:
    return """
    <style>
        .main .block-container { max-width: 1000px; padding-top: 1rem; }
        .analysis-card, .tool-permission-card, .tool-result-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.9) 100%);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 16px; padding: 1.25rem; margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); backdrop-filter: blur(18px); position: relative;
        }
        .analysis-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #34C759 0%, #30D158 100%); border-radius: 16px 16px 0 0; }
        .tool-permission-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #FF9500 0%, #FF9F0A 100%); border-radius: 16px 16px 0 0; }
        .tool-result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #34C759 0%, #30D158 100%); border-radius: 16px 16px 0 0; }
        .stButton > button { border-radius: 12px !important; border: none !important; font-weight: 600 !important;
            padding: 0.7rem 1.1rem !important; font-size: .9rem !important; transition: all .2s ease !important; margin: .25rem !important; }
    </style>
    """


def render_analysis_card(analysis: Dict[str, Any]):
    if analysis.get("needs_tool"):
        icon = "🔧"; title = "Tool Likely Needed"; color = "#FF9500"
        content = (
            f"<strong>Reasoning:</strong> {analysis.get('reasoning', 'Tool usage recommended')}<br>"
            f"<strong>Proposed Tool:</strong> {analysis.get('tool_name', 'unknown')}"
        )
    else:
        icon = "💭"; title = "Direct Answer Available"; color = "#34C759"
        content = f"<strong>Response:</strong> {analysis.get('direct_answer', 'I can answer directly.')}"

    st.markdown(
        f"""
        <div class="analysis-card" style="position:relative;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:.75rem;">
            <div style="font-size:1.2rem;background:{color};color:white;width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;">{icon}</div>
            <div>
              <div style="font-size:1.05rem;font-weight:700;color:#1D1D1F;">{title}</div>
              <div style="color:#8E8E93;font-size:.8rem;">Agent analysis</div>
            </div>
          </div>
          <div style="color:#424245;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_permission_card(req: Dict[str, Any]):
    st.markdown(
        f"""
        <div class="tool-permission-card">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:.75rem;padding-bottom:.5rem;border-bottom:1px solid rgba(0,0,0,.06);">
            <div style="font-size:1.2rem;background:linear-gradient(135deg,#FF9500 0%,#FF9F0A 100%);color:white;width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;">🛡️</div>
            <div>
              <div style="font-size:1.05rem;font-weight:700;color:#1D1D1F;">Tool Permission Required</div>
              <div style="color:#8E8E93;font-size:.8rem;">The agent wants to use: {req['tool_name']}</div>
            </div>
          </div>
          <div style="margin:.5rem 0; color:#424245;">
            <div style="font-weight:700;margin-bottom:.25rem;">🔍 {req['tool_name']}</div>
            <div style="font-size:.9rem;">{req['tool_description']}</div>
          </div>
          <div style="margin:.5rem 0; color:#424245; font-size:.9rem;"><strong>Original Query:</strong><br><em>"{req['original_user_query']}"</em></div>
          <div style="margin:.5rem 0; color:#424245; font-size:.9rem;"><strong>Agent's Reasoning:</strong><br><em>"{req['reasoning']}"</em></div>
          <div style="background:rgba(255,149,0,.06);border:1px solid rgba(255,149,0,.2);border-radius:8px;padding:.6rem;margin-top:.6rem;">
            <div style="color:#8E8E93;font-size:.75rem;margin-bottom:.25rem;">PROPOSED SEARCH QUERY:</div>
            <div style="font-family:ui-monospace, SFMono-Regular, Menlo, monospace;">"{req['query']}"</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_result_card(req: Dict[str, Any]):
    if req["execution_status"] == "completed" and req.get("result"):
        st.markdown(
            f"""
            <div class="tool-result-card">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:.75rem;padding-bottom:.5rem;border-bottom:1px solid rgba(0,0,0,.06);">
                <div style="font-size:1.2rem;background:linear-gradient(135deg,#34C759 0%,#30D158 100%);color:white;width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;">✅</div>
                <div>
                  <div style="font-size:1.05rem;font-weight:700;color:#1D1D1F;">Tool Execution Successful</div>
                  <div style="color:#8E8E93;font-size:.8rem;">Search completed</div>
                </div>
              </div>
              <div style="color:#34C759;font-size:.9rem;margin:.25rem 0;font-weight:600;">🔍 {req['tool_name']} executed</div>
              <div style="background:rgba(248,249,250,.9);border:1px solid rgba(0,0,0,.06);border-radius:8px;padding:.8rem;margin-top:.5rem;font-size:.9rem;line-height:1.5;color:#1D1D1F;white-space:pre-wrap;max-height:320px;overflow:auto;">{req['result']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif req["execution_status"] == "failed":
        st.error(f"❌ Tool execution failed: {req.get('error', 'Unknown error')}")


# =============================
# App
# =============================

def _init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = RealAgent()
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # list of {role, content, timestamp}
    if "current_request" not in st.session_state:
        st.session_state.current_request = None  # dict mirror of ToolRequest
    if "processing" not in st.session_state:
        st.session_state.processing = False  # True when model should produce final answer
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "tool_context" not in st.session_state:
        st.session_state.tool_context = None  # {result, query}
    if "last_llm_response" not in st.session_state:
        st.session_state.last_llm_response = None


def _reset_everything():
    st.session_state.conversation = []
    st.session_state.current_request = None
    st.session_state.processing = False
    st.session_state.tool_context = None
    st.session_state.last_llm_response = None


def _handle_permission_flow(permission_area, results_area):
    """Returns True if UI handled a pending/executing/complete flow and stopped downstream rendering."""
    req = st.session_state.get("current_request")
    if not req:
        return False

    # PENDING: Ask for permission (outside chat)
    if req["permission_status"] == "pending":
        with permission_area:
            st.markdown("### 🛡️ Tool Permission Request")
            render_tool_permission_card(req)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("✅ Allow", key="allow_tool"):
                    req["permission_status"] = "approved"
                    req["execution_status"] = "executing"
                    st.rerun()
            with c2:
                if st.button("❌ Deny", key="deny_tool"):
                    req["permission_status"] = "denied"
                    st.session_state.current_request = None
                    st.session_state.processing = True  # proceed to answer without tools
                    st.rerun()
            with c3:
                if st.button("✏️ Modify Query", key="modify_tool"):
                    st.session_state.modify_mode = True
                    st.rerun()
            with c4:
                if st.button("ℹ️ More Info", key="info_tool"):
                    st.info(
                        f"""
                        **Tool Details:**\n
                        - **Tool**: {req['tool_name']}\n
                        - **Purpose**: {req['tool_description']}\n
                        - **Your Query**: "{req['original_user_query']}"\n
                        - **Agent's Search Query**: "{req['query']}"\n
                        - **Reasoning**: {req['reasoning']}
                        """
                    )

            if st.session_state.get("modify_mode", False):
                st.markdown("#### ✏️ Modify Search Query")
                new_q = st.text_input("Enter modified search query:", value=req["query"], key="modified_query")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("💾 Save & Execute", key="save_modified"):
                        req["query"] = new_q
                        req["permission_status"] = "modified"
                        req["execution_status"] = "executing"
                        st.session_state.modify_mode = False
                        st.rerun()
                with cc2:
                    if st.button("❌ Cancel", key="cancel_modify"):
                        st.session_state.modify_mode = False
                        st.rerun()
        st.stop()
        return True

    # EXECUTING: Run the tool
    if req["execution_status"] == "executing":
        with results_area:
            st.markdown("### 🔄 Executing Tool…")
            with st.spinner(f"Executing {req['tool_name']}..."):
                tool_fn = st.session_state.agent.available_tools[req["tool_name"]]["function"]
                result = tool_fn(req["query"])
                if result.get("success"):
                    req["result"] = result["results"]
                    req["execution_status"] = "completed"
                else:
                    req["error"] = result.get("error", "Unknown error")
                    req["execution_status"] = "failed"
                st.rerun()
        st.stop()
        return True

    # COMPLETED/FAILED: Show results and branch
    if req["execution_status"] in ("completed", "failed"):
        with results_area:
            st.markdown("### 📋 Tool Results")
            render_tool_result_card(req)
            if req["execution_status"] == "completed":
                st.markdown("#### 📋 Evaluate Results")
                r1, r2, r3 = st.columns(3)
                with r1:
                    if st.button("✅ Accept & Continue", key="accept_results"):
                        st.session_state.tool_context = {
                            "result": req["result"],
                            "query": req["original_user_query"],
                        }
                        st.session_state.current_request = None
                        st.session_state.processing = True
                        st.rerun()
                with r2:
                    if st.button("🔄 Retry with Different Query", key="retry_results"):
                        req["permission_status"] = "pending"
                        req["execution_status"] = "not_started"
                        req["result"] = None
                        st.rerun()
                with r3:
                    if st.button("❌ Reject & Answer Without Tool", key="reject_results"):
                        st.session_state.tool_context = None
                        st.session_state.current_request = None
                        st.session_state.processing = True
                        st.rerun()
            else:  # failed
                f1, f2 = st.columns(2)
                with f1:
                    if st.button("🔄 Retry", key="retry_failed"):
                        req["permission_status"] = "pending"
                        req["execution_status"] = "not_started"
                        req["error"] = None
                        st.rerun()
                with f2:
                    if st.button("❌ Cancel", key="cancel_failed"):
                        st.session_state.current_request = None
                        # Do not flip processing on; user may type again
                        st.rerun()
        st.stop()
        return True

    return False


def main():
    st.set_page_config(page_title="Real Agent with Tool Permissions", page_icon="🤖", layout="wide")

    _init_state()

    # Styles and static containers
    st.markdown(get_permission_styles(), unsafe_allow_html=True)
    st.title("🤖 Real Agent with Tool Permissions")
    st.markdown("*A real AI agent powered by Ollama that requests permission before using tools*")

    # Dedicated, non-chat containers for tool UI
    permission_area = st.container()
    results_area = st.container()

    # Sidebar
    with st.sidebar:
        st.header("🎮 Demo Info")
        st.markdown(
            """
            **How it works:**
            1. 🧠 Agent analyzes your query using Ollama
            2. 🔧 If tools are needed, it asks your permission
            3. ✅ You approve/deny/modify the usage
            4. 🚀 Tool executes and shows results (outside chat)
            5. 💬 Assistant posts one final answer in chat
            """
        )
        st.markdown("---")
        st.markdown("**Agent Model:**")
        st.info(st.session_state.agent.model)

        debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode, help="Show raw LLM analysis output")
        st.session_state.debug_mode = debug_mode

        if st.button("🔄 Reset Conversation"):
            _reset_everything()
            st.rerun()

        if st.session_state.debug_mode and st.session_state.get("last_llm_response"):
            st.markdown("---")
            st.markdown("**🐛 Last LLM Analysis (raw):**")
            st.code(st.session_state.last_llm_response or "", language="text")

    # Handle any active tool permission/execution/result flow BEFORE rendering chat
    if _handle_permission_flow(permission_area, results_area):
        return  # flow UI has been rendered and a rerun/stop was issued

    # Chat history (clean: only user & final assistant messages)
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])  # content is plain text/markdown

    # New user input
    user_input = st.chat_input("Ask me anything… I'll request permission if I need to use tools!")
    if user_input:
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time(),
        })
        # Begin analysis step (outside chat)
        with permission_area:
            with st.status("🧠 Analyzing your query…", expanded=True) as status:
                with request_context(f"user-query-{uuid.uuid4()}"):
                    analysis = st.session_state.agent.analyze_query(user_input)
                    st.session_state.last_llm_response = getattr(st.session_state.agent, 'last_raw_response', None)
                status.update(label="Analysis complete", state="complete")

        # Show analysis card (outside chat)
        with permission_area:
            render_analysis_card(analysis)

        # If tool is needed, open permission flow; else, produce a final answer now
        if analysis.get("needs_tool"):
            tool_info = st.session_state.agent.available_tools[analysis["tool_name"]]
            tool_request = ToolRequest(
                id=str(uuid.uuid4())[:8],
                tool_name=analysis["tool_name"],
                tool_description=tool_info["description"],
                query=analysis.get("search_query", user_input),
                reasoning=analysis.get("reasoning", "Tool usage recommended."),
                original_user_query=user_input,
                permission_status=PermissionStatus.PENDING,
                execution_status=ToolExecutionStatus.NOT_STARTED,
            )
            req_dict = asdict(tool_request)
            req_dict["permission_status"] = tool_request.permission_status.value
            req_dict["execution_status"] = tool_request.execution_status.value
            st.session_state.current_request = req_dict
            st.rerun()
        else:
            # Direct answer path: ask model for the final assistant response (no tools)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ]
            final = ""
            for chunk in st.session_state.agent.ollama_service.chat_stream(
                model=st.session_state.agent.model,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            ):
                if chunk.get("content"):
                    final += chunk["content"]
            st.session_state.conversation.append({
                "role": "assistant",
                "content": final,
                "timestamp": time.time(),
            })
            st.rerun()

    # If we're flagged to produce a final message (post-tool), do it now
    if st.session_state.processing and st.session_state.current_request is None:
        # Gather context: last user message + optional tool results
        user_query = next((m["content"] for m in reversed(st.session_state.conversation) if m["role"] == "user"), "")
        tool_ctx = st.session_state.tool_context

        if tool_ctx:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": "I retrieved relevant web results."},
                {"role": "system", "content": f"Tool results (for assistant reference):\n{tool_ctx['result']}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query},
            ]

        final = ""
        for chunk in st.session_state.agent.ollama_service.chat_stream(
            model=st.session_state.agent.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        ):
            if chunk.get("content"):
                final += chunk["content"]

        st.session_state.conversation.append({
            "role": "assistant",
            "content": final,
            "timestamp": time.time(),
        })
        st.session_state.processing = False
        st.session_state.tool_context = None
        st.rerun()


if __name__ == "__main__":
    main()
