# POLARIS Proof of Concept (POC)

This project contains a simplified implementation of the POLARIS framework to autonomously manage the SWIM system.

## Directory Structure
- **/bin**: Contains the NATS server executable.
- **/extern**: Contains external dependencies like the SWIM project.
- **/src**: Contains all the Python source code for the POLARIS components.
- **.env**: Contains the API keys for services like OpenAI.
- **requirements.txt**: Lists the Python dependencies.

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place your NATS server executable in the `/bin` directory.
5. Clone the SWIM project into the `/extern` directory.
6. Fill in your OpenAI API key in the `.env` file.

