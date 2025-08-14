### **Migration Plan: A Modular, Plugin-Driven Adapter Architecture**

**Document Purpose:** This document serves as a detailed instruction set and a reference guide for refactoring the POLARIS adaptation system's adapters. It guides the migration from a tightly-coupled, system-specific implementation to a generic, modular, and extensible plugin-based architecture, complete with robust core services and supporting observability tools.

#### **1. Context and Current State**

The current implementation consists of `SwimMonitorAdapter` and `SwimExecutionAdapter`, both **tightly coupled** to the "SWIM" managed system. This coupling is manifested through embedded TCP client logic, hardcoded commands, and duplicated NATS handling. The existing implementation, however, contains valuable, sophisticated logic for data modeling, error handling, and detailed logging that must be preserved and standardized within the new, more modular architecture.

#### **2. High-Level Objectives & Architectural Vision**

The end state is a robust, plugin-driven architecture that achieves the following:

1.  **Decoupling & Extensibility:** The core **Adaptation Framework** will be system-agnostic, with system-specific logic encapsulated in self-contained **Managed System Plugins**.
2.  **Clarity & Maintainability:** A managed system's capabilities will be defined declaratively in a `config.yaml` file, validated against a formal schema.
3.  **Code Reusability:** Common logic (NATS, configuration, data models) will be abstracted into shared packages.
4.  **Robustness:** Valuable error handling, retry mechanisms, and data models will be preserved and standardized.
5.  **Observability:** The system will include a dedicated tool for real-time visibility into the internal message bus, aiding in debugging and interpretation.

#### **3. Detailed Architectural Blueprint**

**3.1. Final Project Structure with Roles**

```
/
├── extern/                       # A self-contained "Managed System Plugin" for SWIM.
│   ├── __init__.py               # Makes the 'extern' directory a Python package.
│   ├── config.yaml               # The "What": Defines SWIM's metrics, actions, and connector.
│   └── connector.py              # The "How": Implements the communication logic for SWIM.
│
└── src/
    ├── config/                   # Contains framework-wide configurations and schemas.
    │   ├── managed_system.schema.json # Formal schema for validating any plugin's config.yaml.
    │   └── polaris_config.yaml   # High-level configuration for the POLARIS framework itself.
    │
    ├── polaris/                  # The main source code for the Adaptation Framework.
    │   ├── adapters/             # Contains the generic Monitor and Execution adapters.
    │   │   ├── base.py           # Defines the core contracts (ManagedSystemConnector) and base classes.
    │   │   ├── execution.py      # Generic ExecutionAdapter orchestrator.
    │   │   └── monitor.py        # Generic MonitorAdapter orchestrator.
    │   │
    │   ├── common/               # Shared, reusable utilities for the entire framework.
    │   │   ├── config.py         # Manages loading and validation of configuration files.
    │   │   ├── logging_setup.py  # Provides a consistent logger instance to all components.
    │   │   └── nats_client.py    # Abstraction for all NATS communication logic.
    │   │
    │   └── models/               # Contains the shared data models for internal communication.
    │       ├── __init__.py
    │       ├── actions.py        # Defines ControlAction, ExecutionResult, etc.
    │       └── telemetry.py      # Defines TelemetryEvent.
    │
    └── scripts/                  # Standalone scripts for running and observing the system.
        ├── start_component.py    # Main entry point to launch adapters with a specific plugin.
        └── nats_spy.py           # New visibility tool to monitor NATS messages.
```

---

### **Phase 0: Preparatory Work**

**Objective:** Establish the foundational contracts, schemas, and project structure before major code changes.

*   **Task 0.1: Finalize File Structure:** Create the new `src/polaris/models/` directory.
*   **Task 0.2: Define Configuration Schema:** Create and populate `src/config/managed_system.schema.json`. This schema must define the structure for `system_name`, `implementation`, `connection`, `monitoring` (including metrics and strategies), and `execution` (including actions and their parameters with validation rules).

---

### **Phase 1: Core Framework Implementation**

**Objective:** Build the reusable, generic components of the adaptation framework.

*   **Task 1.1: Migrate and Enhance Data Models:**
    *   **Location:** `src/polaris/models/`
    *   **Action:** Move `TelemetryEvent` to `telemetry.py` and `ControlAction`, `ExecutionResult`, etc., to `actions.py`. Refactor them using `pydantic.BaseModel` for automatic validation and type safety.

*   **Task 1.2: Abstract NATS Communication:**
    *   **Location:** `src/polaris/common/nats_client.py`
    *   **Action:** Create a `NATSClient` class that encapsulates all NATS connection, reconnection (with backoff), publishing, and subscription logic.

*   **Task 1.3: Standardize Configuration Management:**
    *   **Location:** `src/polaris/common/config.py`
    *   **Action:** Create a `ConfigurationManager` to load and validate both the main `polaris_config.yaml` (with environment variable overrides) and plugin configurations against the JSON schema.

*   **Task 1.4: Define Base Classes and Contracts:**
    *   **Location:** `src/polaris/adapters/base.py`
    *   **Action:** Define the `ManagedSystemConnector` Abstract Base Class and a `BaseAdapter` class that handles common initialization logic.

---

### **Phase 2: SWIM Plugin Implementation**

**Objective:** Encapsulate all SWIM-specific logic into a self-contained plugin.

*   **Task 2.1: Create Plugin Configuration:**
    *   **Location:** `extern/config.yaml`
    *   **Action:** Create the file according to the defined schema.

*   **Task 2.2: Implement the Plugin Connector:**
    *   **Location:** `extern/connector.py`
    *   **Action:** Create the `SwimTCPConnector` class, migrating the retry and error handling logic from the original adapters into its `execute_command` method.

---

### **Phase 3: Adapter Refactoring**

**Objective:** Transform the adapters into generic orchestrators.

*   **Task 3.1: Refactor Monitor Adapter:**
    *   **Location:** `src/polaris/adapters/monitor.py`
    *   **Action:** Rewrite the adapter to be generic, using the core framework components (`NATSClient`, `ConfigurationManager`, loaded `Connector`) to perform its duties.

*   **Task 3.2: Refactor Execution Adapter:**
    *   **Location:** `src/polaris/adapters/execution.py`
    *   **Action:** Rewrite the adapter to be generic, using the core components to validate and execute actions defined in the plugin configuration.

---

### **Phase 4: Supporting Tools & Integration**

**Objective:** Build tools for observability and integrate all components.

*   **Task 4.1: Implement NATS Visibility Tool:**
    *   **Location:** `src/scripts/nats_spy.py`
    *   **Action:** Create a command-line script that connects to the NATS server and subscribes to key subjects (e.g., `polaris.>` wildcard). It will pretty-print incoming messages to the console, using colors to differentiate subjects.

*   **Task 4.2: Integrate Entry Point:**
    *   **Location:** `src/scripts/start_component.py`
    *   **Action:** Implement `argparse` logic to launch a specific component (`monitor`, `executor`) and inject the path to the desired managed system plugin (`--plugin-dir`).

---

### **Phase 5: Finalization**

**Objective:** Finalize the codebase by improving documentation and removing obsolete code.

*   **Task 5.1: Create Documentation:**
    *   **Action:** Update the main `README.md` to reflect the new architecture. Create a `README.md` inside `extern/` that explains the plugin architecture and provides a guide on how to create a new plugin.
*   **Task 5.2: Code Cleanup:**
    *   **Action:** Ensure all new classes and methods have clear docstrings. Delete the old, system-specific adapter files and any now-unused code.

---

### **Verifiable Migration Checklist**

**Phase 0: Preparatory Work**
- [ ] New directory `src/polaris/models/` is created.
- [ ] `src/config/managed_system.schema.json` is created and populated.

**Phase 1: Core Framework Implementation**
- [ ] Data models are migrated to `src/polaris/models/` and converted to `pydantic`.
- [ ] `NATSClient` is implemented in `src/polaris/common/nats_client.py`.
- [ ] `ConfigurationManager` is implemented in `src/polaris/common/config.py`.
- [ ] `ManagedSystemConnector` ABC is defined in `src/polaris/adapters/base.py`.

**Phase 2: SWIM Plugin Implementation**
- [ ] `extern/config.yaml` is created and is valid against the schema.
- [ ] `SwimTCPConnector` is implemented in `extern/connector.py`.
- [ ] The connector correctly implements the retry and error handling logic.

**Phase 3: Adapter Refactoring**
- [ ] `MonitorAdapter` is refactored to be generic.
- [ ] `ExecutionAdapter` is refactored to be generic.
- [ ] All direct SWIM, TCP, and NATS logic is removed from adapter files.

**Phase 4: Supporting Tools & Integration**
- [ ] `nats_spy.py` utility is implemented and can visualize messages on the bus.
- [ ] `start_component.py` is implemented and can launch the refactored adapters with a specified plugin.

**Phase 5: Finalization**
- [ ] Project documentation is updated to reflect the new architecture.
- [ ] A guide for creating new plugins is written.
- [ ] All obsolete code and files have been removed from the repository.