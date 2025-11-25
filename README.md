# RL_Project

A project implementing 3 reinforcement learning algorithms for the Hockey environment (https://github.com/martius-lab/hockey-env)

## Supported Algorithms

- **TD3**: Twin Delayed Deep Deterministic Policy Gradient (implemented by Luca Reichmann)
- **SAC**: Soft Actor-Critic (implemented by Albert Borchardt)
- **RND**: Random Network Distillation (implemented by Marvin Fischer)

---

## Supported Environments

- **Hockey-v0**: Hockey environment (modes and opponents can be configured in the code)
- **Pendulum-v1**: Classic control environment
- **LunarLander-v2**: Lunar landing environment
- **LunarLanderContinuous-v2**: Continuous Lunar landing environment  
- **HalfCheetah-v4**: MuJoCo locomotion environment

---

---

### Steps to Set Up a Virtual Environment and Install Dependencies

1. **Navigate to project directory**:

   ```bash
   cd /path/to/project
   ```

2. **Create virtual environment**:

   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment**:

   - On **Linux/macOS**:
     ```bash
     source venv/bin/activate

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install `hockey-env` package**:

   ```bash
   python3 -m pip install git+https://github.com/martius-lab/hockey-env.git
   ```
