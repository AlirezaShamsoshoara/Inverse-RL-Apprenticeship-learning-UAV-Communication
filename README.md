# UAV Communication Using Inverse Reinforcement Learning (IRL) 🚁

This project implements a suite of Reinforcement Learning (RL) and Inverse Reinforcement Learning (IRL) methods for UAV communication, including Behavioral Cloning, Q-learning, Deep RL, and other learning policies. The repository explores various RL algorithms for autonomous navigation and decision-making in UAVs.

## 📑 Table of Contents
- Requirements
- Installation
- Project Structure
- Configuration
- Usage
- License

## 📦 Requirements

- Python 3.6 or later
- Libraries specified in `requirements.txt`

To install the required libraries, use the following command:

    pip install -r requirements.txt

## 🚀 Installation

To get started, clone this repository:

    git clone https://github.com/YourUsername/Inverse-RL-Apprenticeship-learning-UAV-Communication.git
    cd Inverse-RL-Apprenticeship-learning-UAV-Communication

## 📂 Project Structure

- `main.py`: The primary script for running the IRL-based UAV communication project.
- `config.py`: Configuration file that includes global settings and parameters for different modes (e.g., `IRL_SGD`, `IRL_DQN`, `DRL`, `QRL`, etc.).
- `behavioral.py`: Implements imitation learning through behavioral cloning.
- `deeprl.py`: Defines functions for Deep Reinforcement Learning.
- `expert.py`: Contains expert operation functions.
- `inverserlDQN.py`: Implements Inverse Reinforcement Learning using a Deep Q-Network.
- `location.py`: Handles location and allocation functions.
- `qlearning.py`: Basic Q-learning reinforcement learning script.
- `randompolicy.py`: Implements a random policy.
- `shortestpath.py`: Uses the shortest path approach, such as Dijkstra's algorithm.

## ⚙️ Configuration

The `config.py` file contains several parameters and settings that you can modify to control different aspects of the project. Key configuration options include:

- **Mode**: Specifies the operation mode. Possible values include:
  - `Expert`
  - `IRL_SGD`
  - `IRL_DQN`
  - `DRL` (Deep Reinforcement Learning)
  - `QRL` (Q-learning Reinforcement Learning)
  - `BC` (Behavioral Cloning)
  - `Shortest` (Shortest Path)
  - `Random` (Random Policy)
  - `ResultsIRL` (Show IRL results)
  - `EvaluationTraining` (Evaluate model training)
  - `EvaluationScenario` (Evaluate scenario results)
  - `EvaluationError` (Evaluate model errors)

  Example setting: `Mode = 'IRL_SGD'`

- **Config_Flags**: A dictionary containing flags for various settings:
  - `SAVE_path`: Set to `True` to save paths generated during runs
  - `Display`: Set to `True` to display graphs and logs for real-time monitoring

- **Config_General**: General settings, such as default learning rate and discount factor. These may vary depending on the mode you choose.

- **Config_Path**: Controls parameters specific to pathfinding methods, which may include settings like maximum path length or heuristic functions if applicable.

- **Config_Power**: Parameters related to power management for UAV communication. This can include transmit power levels, power thresholds, and other UAV-specific settings.

## 📝 Usage

### Running the Main Script

The main script, `main.py`, controls the execution of the entire project. To run it, execute the following command in your terminal:

    python main.py

### Running Individual Modules

Each Python file represents a distinct part of the project. Here are some examples of running individual modules:

1. **Behavioral Cloning**: To execute the behavioral cloning functions, run
       python behavioral.py

2. **Deep Q-Network (Inverse RL)**: Run the DQN-based IRL approach by executing
       python inverserlDQN.py

3. **Q-Learning**: To run Q-learning standalone, use
       python qlearning.py

4. **Shortest Path**: Execute shortest path functions with
       python shortestpath.py

5. **Random Policy**: Run the random policy module using
       python randompolicy.py

Make sure to configure the desired mode and parameters in `config.py` before running these scripts.

## 📜 License

This project is licensed under the MIT License.