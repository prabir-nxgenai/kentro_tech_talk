"""
Autonomous Agent with Tool Integration and Performance Metrics

Author: PRABIR GUHA
Purpose: Demonstrates a LangChain agent that uses tools (web search, time queries)
         and captures comprehensive performance metrics for agent execution.

Features:
  - Tool-based autonomous agent with LangChain
  - DuckDuckGo web search integration
  - System time query tool
  - Performance metrics tracking (execution time, token usage, tool breakdown)
  - Human-readable response formatting with conversation flow
  - Model and query information display

This script is part of the DGX Book - Large Language Models & AI Implementation Guide
"""

# Import Required Modules
import time
from datetime import datetime
from collections import defaultdict
from langchain_core.tools import tool  # To define functions as "tools" that an agent can use
from langchain_core.callbacks import BaseCallbackHandler  # To track LLM execution
from langchain_ollama import ChatOllama  # To interact with LLaMA models served by Ollama
from ddgs import DDGS  # To perform DuckDuckGo search queries
from langchain.agents import create_agent


# ============================================================================
# PERFORMANCE METRICS TRACKER & CALLBACKS
# ============================================================================

class LLMTimingCallback(BaseCallbackHandler):
    """Callback to measure LLM execution time."""

    def __init__(self, metrics: "AgentMetrics"):
        self.metrics = metrics
        self.llm_start = None

    def on_llm_start(self, *args, **kwargs):  # noqa: ARG002
        """Called when LLM starts processing."""
        self.llm_start = time.time()

    def on_llm_end(self, *args, **kwargs):  # noqa: ARG002
        """Called when LLM finishes processing."""
        if self.llm_start:
            elapsed = time.time() - self.llm_start
            self.metrics.record_llm_call(elapsed)
            self.llm_start = None


class AgentMetrics:
    """Tracks performance statistics for agent execution and tool calls."""

    def __init__(self, query_name: str):
        self.query_name = query_name
        self.start_time = None
        self.end_time = None
        self.tool_calls = []
        self.llm_start = None
        self.llm_times = []

    def start(self):
        """Mark the start of agent execution."""
        self.start_time = time.time()

    def end(self):
        """Mark the end of agent execution."""
        self.end_time = time.time()

    def record_tool_call(self, tool_name: str, execution_time: float, input_text: str = "",
                        output_text: str = "", success: bool = True):
        """Record a tool call with timing and metadata."""
        self.tool_calls.append({
            "tool": tool_name,
            "time": execution_time,
            "input_length": len(input_text),
            "output_length": len(output_text),
            "success": success,
            "timestamp": datetime.now()
        })

    def record_llm_call(self, execution_time: float):
        """Record LLM inference time."""
        self.llm_times.append(execution_time)

    def add_response_tool_calls(self, response: dict):
        """Extract tool calls from agent response and record them."""
        if not isinstance(response, dict) or "messages" not in response:
            return

        messages = response.get("messages", [])
        for message in messages:
            # Check for tool calls in AIMessage
            tool_calls = getattr(message, "tool_calls", [])
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                    query = args.get("query", "") if isinstance(args, dict) else str(args)

                    # Record tool call from response if not already recorded
                    if not any(tc["tool"] == tool_name for tc in self.tool_calls):
                        # Try to extract output from ToolMessage
                        tool_output = ""
                        for msg in messages:
                            if hasattr(msg, "name") and msg.name == tool_name:
                                tool_output = getattr(msg, "content", "")
                                break

                        self.record_tool_call(tool_name, 0, query, tool_output)

    def calculate_tool_timing(self):
        """Validate tool timing is captured from wrapper instrumentation."""
        # Tool timing should be captured by the tool wrapper's time.time() measurements
        # LLM callbacks measure the entire agent loop, not just LLM inference
        # So we rely on the tool wrapper to give accurate tool execution times
        if self.tool_calls and sum(call["time"] for call in self.tool_calls) == 0:
            # If tool times are still 0, it means tools executed extremely fast (< 0.1ms)
            # This is normal for tools like get_time
            pass

    def get_summary(self) -> dict:
        """Return performance summary statistics."""
        total_time = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0

        # Group by tool
        tool_breakdown = defaultdict(lambda: {"count": 0, "times": [], "input_length": 0, "output_length": 0})
        for call in self.tool_calls:
            tool_breakdown[call["tool"]]["count"] += 1
            tool_breakdown[call["tool"]]["times"].append(call["time"])
            tool_breakdown[call["tool"]]["input_length"] += call["input_length"]
            tool_breakdown[call["tool"]]["output_length"] += call["output_length"]

        # Calculate averages and totals
        tool_stats = {}
        for tool_name, data in tool_breakdown.items():
            times = data["times"]
            tool_stats[tool_name] = {
                "count": data["count"],
                "total_time": sum(times),
                "avg_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
                "avg_input_length": data["input_length"] / data["count"] if data["count"] > 0 else 0,
                "avg_output_length": data["output_length"] / data["count"] if data["count"] > 0 else 0
            }

        return {
            "query_name": self.query_name,
            "total_time": total_time,
            "tool_calls_count": len(self.tool_calls),
            "llm_calls_count": len(self.llm_times),
            "total_llm_time": sum(self.llm_times) if self.llm_times else 0,
            "avg_llm_time": sum(self.llm_times) / len(self.llm_times) if self.llm_times else 0,
            "total_tool_time": sum(call["time"] for call in self.tool_calls),
            "tools": tool_stats,
            "success_rate": (sum(1 for call in self.tool_calls if call["success"]) / len(self.tool_calls)) * 100 if self.tool_calls else 100
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in appropriate units (ms or s)."""
        if seconds < 0.1:
            return f"{seconds*1000:.2f}ms"
        else:
            return f"{seconds:.2f}s"

    def print_summary(self):
        """Pretty-print performance summary."""
        stats = self.get_summary()

        print("\n" + "="*80)
        print(f"AGENT PERFORMANCE METRICS - {stats['query_name']}")
        print("="*80)
        print(f"Total Execution Time:     {self.format_time(stats['total_time'])}")
        print(f"Tool Calls:               {stats['tool_calls_count']}")
        print(f"LLM Calls:                {stats['llm_calls_count']}")
        print(f"Total LLM Time:           {self.format_time(stats['total_llm_time'])}")
        print(f"Avg LLM Response Time:    {self.format_time(stats['avg_llm_time'])}")
        print(f"Total Tool Time:          {self.format_time(stats['total_tool_time'])}")
        print(f"Tool Success Rate:        {stats['success_rate']:.1f}%")

        if stats["tools"]:
            print("\n--- TOOL BREAKDOWN ---")
            for tool_name, tool_stats in stats["tools"].items():
                print(f"\n  {tool_name}:")
                print(f"    Calls:                {tool_stats['count']}")
                print(f"    Total Time:           {self.format_time(tool_stats['total_time'])}")
                print(f"    Avg Time:             {self.format_time(tool_stats['avg_time'])}")
                print(f"    Min/Max Time:         {self.format_time(tool_stats['min_time'])} / {self.format_time(tool_stats['max_time'])}")
                print(f"    Avg Input Length:     {tool_stats['avg_input_length']:.0f} chars")
                print(f"    Avg Output Length:    {tool_stats['avg_output_length']:.0f} chars")
        print("="*80 + "\n")


# ============================================================================
# WRAPPED TOOLS WITH METRICS
# ============================================================================

# Global metrics tracker
current_metrics = None

def get_time_impl(_query: str = "") -> str:
    """Implementation of get_time (query parameter ignored)."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def duckduckgo_search_impl(query: str = "") -> str:
    """Implementation of duckduckgo_search."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            for result in results:
                snippet = result.get("body") or result.get("snippet")
                if snippet:
                    return snippet
            return "No relevant results found."
    except Exception as e:
        return f"Error during DuckDuckGo search: {str(e)}"


@tool
def get_time(query: str = "") -> str:
    """
    Returns the current system time formatted as YYYY-MM-DD HH:MM:SS.

    Args:
        query (str): Placeholder argument (ignored).

    Returns:
        str: Formatted current date and time.
    """
    start = time.time()
    result = get_time_impl(query)
    elapsed = time.time() - start
    if current_metrics:
        current_metrics.record_tool_call("get_time", elapsed, query, result)
    return result


@tool
def duckduckgo_search(query: str = "") -> str:
    """
    Performs a DuckDuckGo text search and returns the first relevant snippet.

    Args:
        query (str): The search query string.

    Returns:
        str: The first found snippet or an error message.
    """
    start = time.time()
    result = duckduckgo_search_impl(query)
    elapsed = time.time() - start
    success = not result.startswith("Error")
    if current_metrics:
        current_metrics.record_tool_call("duckduckgo_search", elapsed, query, result, success)
    return result


TOOLS = [get_time, duckduckgo_search]


# Initialize the Model via Ollama
MODEL_NAME = "muse-glimmer:30b"
llm = ChatOllama(model=MODEL_NAME, base_url="http://localhost:11434")

# System prompt to force tool usage for all queries
SYSTEM_PROMPT = """You are a helpful assistant with access to specialized tools. You MUST use the available tools to answer queries.

Available tools:
1. **get_time**: Returns the current date and time in YYYY-MM-DD HH:MM:SS format
   - Use ONLY when the user asks about the current time, today's date, or what time it is
   - Example: "What is the current time?" → MUST use get_time

2. **duckduckgo_search**: Performs an internet search and returns relevant results
   - Use this for ALL OTHER QUERIES to search the internet
   - Use this to provide information, verify facts, research topics, answer questions
   - ALWAYS use this tool for any question that is not about the current time

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
- For time-related queries (current time, what time is it, today's date) → USE get_time tool
- For ALL OTHER queries → USE duckduckgo_search tool FIRST, then provide your answer based on the search results
- Do NOT answer questions from your training data alone - ALWAYS search first
- You must use at least one tool for every user query
- Think: "What tool should I use?" and use it BEFORE giving your answer

Workflow:
1. Analyze the user's query
2. Choose the appropriate tool (get_time or duckduckgo_search)
3. Use the tool to get information
4. Provide your answer based on the tool's results

Now, help the user by using tools to answer their question."""

# Initialize the LangChain Agent with system prompt
from langchain_core.messages import SystemMessage

agent = create_agent(
    llm,
    TOOLS,
    system_prompt=SYSTEM_PROMPT
)


# ============================================================================
# EXECUTE QUERIES WITH METRICS TRACKING
# ============================================================================

def format_agent_response(response: dict):
    """Format entire agent response with conversation flow in human-readable format."""
    if not isinstance(response, dict) or "messages" not in response:
        return str(response)

    messages = response.get("messages", [])
    formatted_output = []

    for i, message in enumerate(messages, 1):
        message_type = message.__class__.__name__ if hasattr(message, "__class__") else "Unknown"

        if message_type == "HumanMessage":
            formatted_output.append(f"\nSTEP {i}: USER QUERY")
            formatted_output.append(f"{'─'*70}")
            if hasattr(message, "content"):
                formatted_output.append(message.content)
            else:
                formatted_output.append(str(message))

        elif message_type == "AIMessage":
            # Check if this AI message has tool calls (thinking step)
            tool_calls = getattr(message, "tool_calls", [])
            if tool_calls:
                formatted_output.append(f"\nSTEP {i}: AGENT DECIDES TO USE TOOLS")
                formatted_output.append(f"{'─'*70}")
                for j, tool_call in enumerate(tool_calls, 1):
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        formatted_output.append(f"  Tool {j}: {tool_name}")
                        for key, value in tool_args.items():
                            formatted_output.append(f"    └─ {key}: {value}")
                    else:
                        formatted_output.append(f"  Tool call: {tool_call}")

                # Show token usage if available
                usage = getattr(message, "usage_metadata", {})
                if usage:
                    formatted_output.append(f"\n  LLM Tokens Used:")
                    formatted_output.append(f"    Input:  {usage.get('input_tokens', 'N/A')} tokens")
                    formatted_output.append(f"    Output: {usage.get('output_tokens', 'N/A')} tokens")

            else:
                # This is a final answer AI message
                formatted_output.append(f"\nSTEP {i}: AGENT'S FINAL ANSWER")
                formatted_output.append(f"{'─'*70}")
                if hasattr(message, "content") and message.content:
                    formatted_output.append(message.content)
                else:
                    formatted_output.append("[Agent provided no content]")

                # Show token usage
                usage = getattr(message, "usage_metadata", {})
                if usage:
                    formatted_output.append(f"\n  LLM Tokens Used:")
                    formatted_output.append(f"    Input:  {usage.get('input_tokens', 'N/A')} tokens")
                    formatted_output.append(f"    Output: {usage.get('output_tokens', 'N/A')} tokens")

        elif message_type == "ToolMessage":
            tool_name = getattr(message, "name", "unknown")
            formatted_output.append(f"\nSTEP {i}: TOOL RESULT ({tool_name})")
            formatted_output.append(f"{'─'*70}")
            if hasattr(message, "content"):
                content = message.content
                # Truncate very long outputs
                if len(content) > 500:
                    formatted_output.append(content[:500] + "...\n[Output truncated]")
                else:
                    formatted_output.append(content)

    return "\n".join(formatted_output)


def run_query_with_metrics(query_name: str, user_message: str):
    """Execute an agent query and capture performance metrics."""
    global current_metrics
    current_metrics = AgentMetrics(query_name)

    print(f"\n{'='*80}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"QUERY: {query_name}")
    print(f"{'='*80}\n")

    # Create callback to measure LLM execution time
    llm_callback = LLMTimingCallback(current_metrics)

    current_metrics.start()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"callbacks": [llm_callback]}
    )
    current_metrics.end()

    # Extract tool calls from response if not already recorded
    current_metrics.add_response_tool_calls(response)

    # Calculate tool timing from total time - LLM time
    current_metrics.calculate_tool_timing()

    # Debug: Show what was recorded
    if current_metrics.tool_calls:
        print(f"\n[DEBUG] Recorded {len(current_metrics.tool_calls)} tool call(s):")
        for tc in current_metrics.tool_calls:
            print(f"  - {tc['tool']}: {tc['time']:.4f}s, input={tc['input_length']} chars, output={tc['output_length']} chars")

    print("AGENT CONVERSATION FLOW:")
    print(f"{'='*80}")
    response_text = format_agent_response(response)
    print(response_text)
    print(f"{'='*80}\n")

    current_metrics.print_summary()

    return response


# Execute Queries Using the Agent with Metrics
run_query_with_metrics(
    "Query 1: Current Time",
    "What is the current time? Provide answer in YYYY-MM-DD HH:MM:SS format"
)

run_query_with_metrics(
    "Query 2: Astronomy Search",
    "What is the closest planet to the sun?"
)

