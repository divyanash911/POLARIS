
# **POLARIS Proof of Concept (POC) Implementation Plan**

## **1. POC Objectives & Scope**

### **1.1. Primary Goals**
This Proof of Concept (POC) aims to validate the core architectural principles of a fully AI-native POLARIS framework by building a simplified, functional version that can autonomously manage the SWIM testbed.

The key validation points are:
1.  **End-to-End Viability:** Prove that a signal can flow from the SWIM system, through the entire POLARIS architecture, and result in a meaningful control action.
2.  **Architectural Separation:** Demonstrate the logical separation of concerns between a lightweight Kernel, an LLM-powered Digital Twin, and an LLM-powered Reasoner.
3.  **LLM as a World Model:** Validate the feasibility of using an LLM to maintain a qualitative state representation of SWIM and predict the effects of actions upon it.
4.  **LLM for Controlled Reasoning:** Show that another LLM can be safely used as a bounded reasoning component, using the Digital Twin as a tool to make informed decisions.

### **1.2. Simplifying Assumptions for the POC**
To ensure rapid development, we make the following explicit simplifications:
*   **No Docker/Containerization:** The NATS message broker will be run directly as a Go executable on the local machine. All other components are Python scripts.
*   **No Kernel Fast Path:** The Kernel's role is simplified to that of a message router and a final checkpoint for actions. It will not have its own deterministic control rules.
*   **Simplified Networking:** All components will run on `localhost` and communicate via the local NATS server. All NATS topics are unauthenticated.
*   **No Service Discovery:** The Agentic Plane components (`DigitalTwin` and `Reasoner`) will use hardcoded NATS topic names instead of a dynamic Service Registry and Router.
*   **Single-Threaded Processing:** Each component will handle one message at a time. We are not building a high-throughput, concurrent system.
*   **Mocked Tool Calls:** The network communication inside the Reasoner's tool (for simulating actions) will be mocked to simplify the logic, though the interface itself will be designed for a real network call.

## **2. System Architecture & Components**

### **2.1. Architectural Diagram**
```mermiad
flowchart TB
    subgraph LocalMachine["Local Machine (localhost)"]
        direction TB

        %% Top row
        NATS["NATS Server<br/>(Executable)"]
        SWIM["SWIM Process<br/>(UDP Ports 4001, 4002)"]

        NATS -- "NATS" --> SWIM

        %% Python processes
        kernel["kernel.py"]
        twin["digital_twin_agent.py"]
        reasoner["reasoner_langchain.py"]

        monitor["monitor_adapter.py"]
        execution["execution_adapter.py"]

        %% SWIM UDP connections to processes
        SWIM -- "UDP" --> kernel
        SWIM -- "UDP" --> twin
        SWIM <--> reasoner

        %% Internal process connections
        kernel --> twin
        twin --> reasoner
        reasoner --> monitor
        reasoner --> execution
    end
```

### **2.2. Component Responsibilities**

| Component | Responsibility | Technology |
| :--- | :--- | :--- |
| **NATS Server** | The central message bus for all POLARIS components. | Go Executable |
| **SWIM** | The target system to be adapted. | Java Executable |
| **`monitor_adapter.py`** | Listens to SWIM's UDP telemetry and translates it into POLARIS `TelemetryEvent`s. | Python, `asyncio` |
| **`execution_adapter.py`** | Receives POLARIS `ControlAction`s and translates them into SWIM's UDP commands. | Python, `nats-py` |
| **`kernel.py`** | The central orchestrator. Forwards telemetry and dispatches final actions. | Python, `nats-py` |
| **`digital_twin_agent.py`** | The "World Model." Maintains a conversational state of SWIM and simulates action effects. | Python, `langchain` |
| **`reasoner_langchain.py`** | The "Decision Maker." Uses the Digital Twin as a tool to reason about the best adaptation plan. | Python, `langchain` |

---

## **3. Detailed Implementation & Interfaces**

### **3.1. Infrastructure: NATS Server**
*   **Setup:** Download the `nats-server` executable from the [official NATS releases page](https://github.com/nats-io/nats-server/releases).
*   **Execution:** Run `./nats-server` from your terminal. It will automatically listen on the default port `4222`. No other configuration is needed.

### **3.2. Adapters: `monitor_adapter.py`**
*   **Interface:**
    *   **Input:** UDP datagrams on `localhost:4001` from SWIM (e.g., `avg_latency:150.7;...`).
    *   **Output:** JSON messages on the NATS topic `polaris.telemetry.events`.
*   **Implementation (`asyncio` for UDP):**
    ```python
    import asyncio
    import json
    from nats.aio.client import Client as NATS

    class UDPServerProtocol:
        def __init__(self, nc):
            self.nc = nc

        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            message = data.decode()
            # Parse the string: "key1:val1;key2:val2"
            parts = [p.split(':') for p in message.strip().split(';') if ':' in p]
            for key, value in parts:
                telemetry_event = {
                    "event_type": "TelemetryEvent",
                    "payload": {"name": f"swim.{key}", "value": float(value)}
                }
                # Publish to NATS
                asyncio.create_task(self.nc.publish(
                    "polaris.telemetry.events",
                    json.dumps(telemetry_event).encode()
                ))

    async def main():
        nc = NATS()
        await nc.connect("nats://localhost:4222")
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: UDPServerProtocol(nc),
            local_addr=('127.0.0.1', 4001)
        )
        while True: await asyncio.sleep(1) # Keep running

    if __name__ == '__main__':
        asyncio.run(main())
    ```

### **3.3. Adapters: `execution_adapter.py`**
*   **Interface:**
    *   **Input:** JSON messages on the NATS topic `polaris.actions.swim_adapter`.
    *   **Output:** UDP datagrams to `localhost:4002` for SWIM (e.g., `addServer`).
*   **Implementation:**
    ```python
    import asyncio
    import json
    from nats.aio.client import Client as NATS
    import socket

    async def main():
        nc = NATS()
        await nc.connect("nats://localhost:4222")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        swim_addr = ('127.0.0.1', 4002)

        async def message_handler(msg):
            data = json.loads(msg.data.decode())
            action_type = data.get("action_type")
            command = ""
            if action_type == "ADD_SERVER":
                command = "addServer"
            elif action_type == "REMOVE_SERVER":
                command = "removeServer"
            elif action_type == "ADJUST_QOS":
                value = data.get("params", {}).get("value")
                if value is not None:
                    command = f"setDimmer {value}"
            
            if command:
                sock.sendto(command.encode(), swim_addr)
                print(f"EXECUTOR: Sent command '{command}' to SWIM")

        await nc.subscribe("polaris.actions.swim_adapter", cb=message_handler)
        while True: await asyncio.sleep(1)

    if __name__ == '__main__':
        asyncio.run(main())
    ```

### **3.4. Core Component: `kernel.py`**
*   **Interface:**
    *   **Input 1:** `polaris.telemetry.events` (from `monitor_adapter`).
    *   **Input 2:** `polaris.responses.coordination` (from `reasoner`).
    *   **Output 1:** `polaris.telemetry.events` (forwarded to `digital_twin_agent`).
    *   **Output 2:** `polaris.requests.reasoning` (when triggered).
    *   **Output 3:** `polaris.actions.swim_adapter` (final action).
*   **Logic:**
    1.  Subscribe to `polaris.telemetry.events`. For each message:
        *   Forward it to `polaris.telemetry.for_twin`.
        *   Check a simple rule: `if avg_latency > 180: publish_reasoning_request()`.
    2.  Subscribe to `polaris.responses.coordination`. For each message (a final plan):
        *   Extract the action part of the plan.
        *   Publish it as a `ControlAction` to `polaris.actions.swim_adapter`.

### **3.5. Agentic Component: `digital_twin_agent.py`**
*   **Interface:**
    *   **Input 1:** `polaris.telemetry.for_twin` (state updates).
    *   **Input 2:** `polaris.requests.simulation` (simulation queries).
    *   **Output:** `polaris.responses.simulation` (simulation results).
*   **Implementation Detail (The Core Logic):**
    ```python
    # ... (imports for langchain, nats, etc.) ...
    # Initialize the LLM chain with message history
    chain = prompt | llm 
    chat_history = {} # A dict to hold conversation history per session_id

    # NATS handler for telemetry updates
    async def telemetry_handler(msg):
        data = json.loads(msg.data)
        update_text = f"UPDATE: The {data['payload']['name']} is now {data['payload']['value']}."
        # Invoke the chain, updating its history
        # For a single twin, the session_id can be static
        chain.invoke({"input": update_text}, config={"configurable": {"session_id": "swim_twin"}})
        print(f"TWIN: State updated with: {update_text}")

    # NATS handler for simulation requests
    async def simulation_handler(msg):
        data = json.loads(msg.data)
        action = data.get("action")
        sim_text = f"SIMULATE: What is the most likely effect on all key metrics if the action '{action}' is performed now? Respond ONLY with a JSON object..."
        # Invoke the chain to get a simulation result
        response = chain.invoke({"input": sim_text}, config={"configurable": {"session_id": "swim_twin"}})
        # Publish the LLM's content as the response
        await nc.publish("polaris.responses.simulation", response.content.encode())
    ```
    *(The `RunnableWithMessageHistory` object from LangChain can simplify history management.)*

### **3.6. Agentic Component: `reasoner_langchain.py`**
*   **Interface:**
    *   **Input:** `polaris.requests.reasoning`.
    *   **Output:** `polaris.responses.coordination`.
    *   **Internal Interaction:** Sends requests to `polaris.requests.simulation` and receives responses from `polaris.responses.simulation`.
*   **Implementation Detail (The Tool):**
    The `simulate_action` tool will not be mocked but will perform a real NATS request/response cycle.
    ```python
    @tool
    def simulate_action(action_type: str) -> dict:
        """Queries the Digital Twin Agent to simulate the effect of an action."""
        # This requires a NATS client instance to be available
        request = {"action": action_type}
        
        # 1. Need a way to wait for a specific response.
        # This can be done using NATS request-reply pattern.
        # nc.request("polaris.requests.simulation", json.dumps(request).encode(), timeout=10)
        # The response message's data is the result.
        
        # For simplicity in a first pass, a simple pub/sub with a short sleep might work,
        # but the request-reply pattern is correct.
        
        print(f"REASONER: Simulating action '{action_type}'...")
        # ... NATS request logic ...
        response_json = # ... parsed from NATS response ...
        return response_json
    ```

---

## **4. Setup & Execution Guide**

### **4.1. Prerequisites**
1.  **Install Python 3.9+**.
2.  **Download `nats-server` executable** for your OS and place it in your project directory.
3.  **Clone SWIM repository**.
4.  **Create Python virtual environment** and `pip install nats-py "pydantic>=2.0" langchain langchain_openai python-dotenv asyncio`.
5.  **Create `.env` file** with your `OPENAI_API_KEY`.

### **4.2. Step-by-Step Execution**
1.  **Terminal 1: Start NATS:** `./nats-server`
2.  **Terminal 2: Start SWIM:** `java -jar swim.jar ...` (as per its docs)
3.  **Terminal 3: Start Monitor Adapter:** `python monitor_adapter.py`
4.  **Terminal 4: Start Execution Adapter:** `python execution_adapter.py`
5.  **Terminal 5: Start Kernel:** `python kernel.py`
6.  **Terminal 6: Start Digital Twin Agent:** `python digital_twin_agent.py`
7.  **Terminal 7: Start Reasoner Agent:** `python reasoner_langchain.py`
8.  **Observe:** Watch the logs from all terminals. You should see telemetry flowing, the twin updating, the reasoner getting triggered, simulations running, and finally, commands being sent back to SWIM.