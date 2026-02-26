# MASS Quickstart Guide

> Muscle-Actuated Skeletal System, Scalable Muscle-actuated Human Simulation and Control  
> Lee et al., ACM SIGGRAPH 2019 ([Paper](http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf) | [Video](https://youtu.be/a3jfyJ9JVeM))

---

## What is MASS?

MASS is a physics-based simulation of a **full-body musculoskeletal human**. Unlike most character animation systems that apply torques directly to joints, MASS uses **284 muscles** to drive a skeleton with **56 degrees of freedom**, just like a real human body.

You can train either a standard human or a human assisted by a hip exoskeleton. Switching between these modes happens at runtime using the `--mode` flag.

*   **Original Mode**: Standard muscle-actuated walking using two networks (SimulationNN and MuscleNN).
*   **Exo Mode**: Adds a hip exoskeleton with 6 degrees of freedom (3 axes for each hip). This mode adds a third network (ExoPolicyNN) to control the exoskeleton torques.

The system uses **Deep Reinforcement Learning (PPO)** to train a controller that learns to walk by coordinating muscle activations. Training produces up to three neural networks depending on the mode:

| Network | Role | Architecture |
|---------|------|--------------|
| **SimulationNN** (Actor-Critic) | Decides what the character should do (actions + value estimate) | 256 → 256 hidden, ReLU |
| **MuscleNN** | Converts desired joint torques into individual muscle activations | 1024 → 512 → 512 hidden, LeakyReLU |
| **ExoPolicyNN** | Controls the 6 hip exoskeleton torques (Exo mode only) | 256 → 256 hidden, ReLU |

**In simple terms**: the actor says "move the knee this way", the muscle network figures out which muscles to use, and the exo network provides extra power at the hips to help.

---

## Platform Support Overview

| Platform | Docker Backend | x86_64 Emulation | Training Speed |
|----------|---------------|-------------------|----------------|
| **Windows 10/11** | WSL 2 | No (WSL2 is native x86_64) | Fast |
| **macOS (Intel)** | Docker VM | No (native x86_64) | Fast |
| **macOS (Apple Silicon)** | Docker VM + Rosetta/QEMU | Yes (emulated) | Very Slow (~10-50x) |
| **Linux x86_64** | Native | No | Fastest |

> **Recommendation**: If you have a choice, use **Windows** or **Linux x86_64** for training. macOS with Apple Silicon (M1/M2/M3/M4) runs correctly but very slowly due to emulation.

---

## Prerequisites

### All Platforms

- [Git](https://git-scm.com/downloads), to clone the project

### Windows

1. **Windows 10 version 2004+** or **Windows 11** (required for WSL 2)
2. Install **WSL 2**:
   - Open PowerShell **as Administrator** and run:
     ```powershell
     wsl --install
     ```
   - Restart your computer when prompted
   - After restart, a Ubuntu terminal will open. Set your username and password.
3. Install **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)**
   - During installation, make sure **"Use WSL 2 based engine"** is checked.
   - After installation, open Docker Desktop → Settings → General. Confirm **"Use the WSL 2 based engine"** is enabled.
   - Go to Settings → Resources → WSL Integration. Enable integration with your Ubuntu distribution.

> **Important**: On Windows, all commands in this guide should be run inside the **WSL 2 Ubuntu terminal**. You can open it by searching "Ubuntu" in the Start menu, or by typing `wsl` in PowerShell.

### macOS

- Install **[Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)** (Apple Silicon or Intel both work)
- Open Docker Desktop and wait for it to fully start. The whale icon in the menu bar stops animating when it's ready.

### Linux

- Install Docker Engine:
  ```bash
  # Ubuntu / Debian
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo usermod -aG docker $USER
  ```
  **Log out and log back in** for the group change to take effect.

- Or install [Docker Desktop for Linux](https://docs.docker.com/desktop/install/linux-install/)

> **Why Docker?** MASS depends on the DART 6.8 physics engine and a Linux-specific OpenGL stack. Docker lets us run a complete Ubuntu 22.04 environment with everything pre-configured. You don't have to install dozens of libraries manually.

---

## Project Structure

```
DART_MASS/
├── Dockerfile              # Defines the Ubuntu 22.04 + DART 6.8 + MASS environment
├── docker-compose.yml      # Container configuration (volumes, resources, etc.)
├── docker-entrypoint.sh    # Starts virtual display for headless rendering
├── build.sh                # Helper script: builds the Docker image (Mac/Linux)
├── run.sh                  # Helper script: starts the container (Mac/Linux)
├── nn_output/              # Training outputs appear here (mounted from container)
├── MASS/                   # The MASS source code
│   ├── core/               # C++ simulation engine (skeleton, muscles, environment)
│   ├── python/             # Python training code + pybind11 bindings
│   │   ├── main.py         # Training entry point (PPO algorithm), supports --mode original|exo
│   │   ├── Model.py        # Neural network definitions (SimulationNN, MuscleNN, ExoPolicyNN)
│   │   └── EnvManager.cpp  # C++/Python bridge (pybind11)
│   ├── render/             # OpenGL visualization (GLUT)
│   ├── data/               # Skeleton, muscle definitions, motion references
│   │   ├── metadata.txt    # Simulation config (control Hz, sim Hz, rewards, etc.)
│   │   ├── human.xml       # Skeleton definition (56 DOFs)
│   │   ├── muscle284.xml   # Muscle attachment definitions (284 muscles)
│   │   └── motion/         # Reference BVH motion files (walk, run, etc.)
│   └── CMakeLists.txt      # Build configuration
└── dart-mass-main/         # Reference source for exo training extensions (read-only, not used at runtime)
```

---

## Step 1: Clone the Project

### Windows (in WSL 2 Ubuntu terminal)

```bash
mkdir -p ~/Coder && cd ~/Coder
git clone <your-project-repo-url> DART_MASS
cd DART_MASS
```

> **Important**: Clone into the WSL 2 filesystem (`~/Coder/`), **NOT** into `/mnt/c/` (Windows drive). Using `/mnt/c/` will cause severe performance issues with Docker volume mounts.

### macOS / Linux

```bash
mkdir -p ~/Coder && cd ~/Coder
git clone <your-project-repo-url> DART_MASS
cd DART_MASS
```

---

## Step 2: Build the Docker Image

### Windows (in WSL 2 Ubuntu terminal)

```bash
cd ~/Coder/DART_MASS

# Fix line endings (Windows may corrupt them)
sed -i 's/\r$//' docker-entrypoint.sh build.sh run.sh

# Build the image
docker compose build --progress=plain
```

### macOS / Linux

```bash
cd ~/Coder/DART_MASS
chmod +x build.sh run.sh
./build.sh
```

Or directly:

```bash
cd ~/Coder/DART_MASS
docker compose build --progress=plain
```

**This takes 15 to 30 minutes on first build.** It downloads Ubuntu 22.04, compiles the DART 6.8.5 physics engine from source, installs PyTorch, and compiles the MASS project. Subsequent builds use cache and are much faster.

Wait until you see output ending with:

```
Image mass-sim:latest Built
```

---

## Step 3: Start the Container

### All Platforms

```bash
cd ~/Coder/DART_MASS
mkdir -p nn_output
docker compose up -d
```

Verify the container is running:

```bash
docker compose ps
```

You should see:

```
NAME       IMAGE             ...   STATUS
mass-sim   mass-sim:latest   ...   Up XX seconds
```

The container is now running in the background with:
- A virtual display (Xvfb) for headless OpenGL rendering.
- The `nn_output/` folder on your host mapped to `/workspace/MASS/nn` inside the container.
- The `MASS/data/` folder on your host mapped to `/workspace/MASS/data` inside the container. You can edit `metadata.txt` without rebuilding.

---

## Step 4: Enter the Container

### All Platforms

```bash
docker compose exec mass bash
```

You're now inside the Ubuntu 22.04 container at `/workspace/MASS/`. Everything is pre-compiled and ready.

> **Tip**: This is a Linux shell regardless of your host OS. All commands from this point forward are the same on Windows, macOS, and Linux.

---

## Step 5: Start Training

Inside the container, you can choose between two training modes.

### Original Training
This mode trains the standard muscle-actuated model from the original MASS paper.
```bash
cd /workspace/MASS/python
python3 main.py -d ../data/metadata.txt --mode original
```

### Exoskeleton Training
This mode adds a hip exoskeleton controller to the simulation.
```bash
cd /workspace/MASS/python
python3 main.py -d ../data/metadata.txt --mode exo
```

### Mode Comparison

| Feature | Original Mode | Exo Mode |
|---------|---------------|----------|
| **Networks** | 2 (Sim, Muscle) | 3 (Sim, Muscle, Exo) |
| **Exo Actions** | 0 | 6 (Hip torques) |
| **PD Actions** | 50 | 50 |
| **States** | 136 | 136 |
| **Optimizer** | Separate | Joint (Sim + Exo) |

### What You'll See

The training loop will print progress. In **Exo Mode**, you will see an additional line for exo actions:

```
num states: 136, num actions: 50, num exo actions: 6
SIM : 2048
(2048, 334)
Optimizing sim nn : 10/10
Optimizing muscle nn : 3/3
# 1 === 0h:5m:32s ===
|||Loss Actor               : 0.0012
|||Loss Critic              : 0.0234
|||Loss Muscle              : 0.0008
|||Noise                    : 1.000
|||Num Transition So far    : 2048
|||Num Transition           : 2048
|||Num Episode              : 16
|||Avg Return per episode   : 12.345
|||Avg Reward per transition: 0.096
|||Avg Step per episode     : 128.0
|||Max Avg Return So far    : 12.345 at #1
=============================================
```

**Key metrics to watch:**
- **Avg Return per episode**: higher is better. This is the main indicator of learning progress.
- **Loss Actor / Critic**: should generally decrease over time.
- **Noise**: exploration noise. It starts at 1.0 and naturally decreases as the policy improves.

### Training Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Parallel environments | 16 | 16 copies of the simulation run simultaneously |
| Buffer size | 2048 | Transitions collected per iteration before optimization |
| Batch size | 128 | Mini-batch size for neural network updates |
| PPO epochs | 10 | Passes over collected data per iteration |
| Muscle NN epochs | 3 | Passes for muscle network optimization |
| Max iterations | 50,000 | Total training iterations |
| Control frequency | 30 Hz | Policy makes decisions 30 times per second |
| Simulation frequency | 600 Hz | Physics runs at 600 Hz (20 sub-steps per control step) |
| Discount factor (gamma) | 0.99 | How much future rewards are valued |
| Exo Reward Penalty | Note | Exo mode adds penalties: 0.02 * activation + 0.001 * exo_power |

### How Long Does Training Take?

| Platform | Time per iteration (approx.) | Notes |
|----------|------------------------------|-------|
| **Linux x86_64** | Seconds | Native speed. Best for serious training. |
| **Windows (WSL 2)** | Seconds | WSL 2 runs native x86_64, nearly as fast as Linux. |
| **macOS (Intel)** | Seconds to minutes | Docker VM overhead, but no emulation. |
| **macOS (Apple Silicon)** | Several minutes | QEMU x86_64 emulation is very slow. |

For reference, the original paper's training on a Linux workstation:
- Noticeable walking behavior: ~1,000 to 2,000 iterations
- Good walking policy: ~10,000+ iterations

> **Apple Silicon users**: Start with a short run (5 to 10 iterations) to verify everything works. For serious training, use a Windows/Linux machine or a cloud instance.

---

## Step 6: Find Your Trained Models

Training saves neural network checkpoints to the `nn/` folder inside the container. This maps to `nn_output/` on your host machine.

| File | Description |
|------|-------------|
| `current.pt` / `current_muscle.pt` | Latest Simulation and Muscle networks |
| `current_exo.pt` | Latest Exo network (Exo mode only) |
| `max.pt` / `max_muscle.pt` | Best Simulation and Muscle networks |
| `max_exo.pt` | Best Exo network (Exo mode only) |
| `0.pt` / `0_muscle.pt` / `0_exo.pt` | Checkpoint at iteration 0 (exo file in Exo mode only) |
| `1.pt` / `1_muscle.pt` / `1_exo.pt` | Checkpoint at iteration 100 (exo file in Exo mode only) |
| `episode_metrics_baseline.csv` | Training metrics for the character |
| `episode_metrics_exo.csv` | Power and activation metrics for the exoskeleton (Exo mode only) |

Check the outputs on your host at any time:

**Windows (WSL 2 terminal)**:
```bash
ls ~/Coder/DART_MASS/nn_output/
```

You can also access WSL files from Windows File Explorer by navigating to `\\wsl$\Ubuntu\home\<your-username>\Coder\DART_MASS\nn_output\`.

**macOS / Linux**:
```bash
ls ~/Coder/DART_MASS/nn_output/
```

---

## Step 7: Resume Training from a Checkpoint

If training was interrupted, you can resume from a saved model. Inside the container:

```bash
cd /workspace/MASS/python

# Resume original training
python3 main.py -d ../data/metadata.txt --mode original -m current

# Resume exo training
python3 main.py -d ../data/metadata.txt --mode exo -m current
```

The `-m` flag takes the model name without the `.pt` extension and without the path prefix. It automatically loads `current.pt` and `current_muscle.pt` in original mode. In exo mode, it also loads `current_exo.pt`.

---

## Step 8: Visualize with the Render Tool (Optional)

The render module provides an OpenGL visualization. Inside the container:

```bash
# View untrained simulation
cd /workspace/MASS
./build/render/render ./data/metadata.txt

# View with trained model
./build/render/render ./data/metadata.txt ./nn/max.pt ./nn/max_muscle.pt
```

> **Note**: The render tool does not use the exo network. It only visualizes the skeleton motion.

> **Note**: Rendering runs inside a virtual display (Xvfb) in the container. To see the actual window on your host, you need X11 forwarding:
> - **Windows**: Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [X410](https://x410.dev/). Set `export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0` in WSL before entering the container.
> - **macOS**: Install [XQuartz](https://www.xquartz.org/). Run `xhost +localhost` and pass `-e DISPLAY=host.docker.internal:0` to Docker.
> - **Linux**: Run `xhost +local:docker` and pass `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` to Docker.
>
> For most training workflows, you don't need the render tool. Just watch the training metrics and the saved models.

---

## Common Commands Reference

All commands below run on the **host** terminal (WSL 2 on Windows, Terminal on macOS/Linux), unless marked with "inside container".

```bash
# ===== Host commands =====

# Build the Docker image (first time or after Dockerfile changes)
docker compose build --progress=plain

# Start the container in background
docker compose up -d

# Enter the container
docker compose exec mass bash

# Stop the container
docker compose down

# Check container status
docker compose ps

# View container logs
docker compose logs mass

# ===== Inside container =====

# Start original training
cd /workspace/MASS/python && python3 main.py -d ../data/metadata.txt --mode original

# Start exo training
cd /workspace/MASS/python && python3 main.py -d ../data/metadata.txt --mode exo

# Resume training from checkpoint
cd /workspace/MASS/python && python3 main.py -d ../data/metadata.txt --mode exo -m current

# Run render
cd /workspace/MASS && ./build/render/render ./data/metadata.txt

# Exit container (back to host)
exit
```

---

## Understanding the Training Pipeline

Here's what happens in each training iteration:

```
┌─────────────────────────────────────────────────────────┐
│  1. SIMULATE (GenerateTransitions)                      │
│     16 parallel envs collect 2048 transitions           │
│                                                         │
│     For each control step (30 Hz):                      │
│       a) SimulationNN observes state → outputs PD action│
│       b) ExoPolicyNN outputs 6 hip torques (Exo mode)   │
│       c) MuscleNN converts PD torques to activations    │
│       d) Both PD and Exo torques applied to physics     │
│       e) Physics steps 20x at 600 Hz                    │
│       f) Collect (state, action, reward, value, logprob)│
│                                                         │
│  2. COMPUTE TD & GAE                                    │
│     Calculate advantages for PPO                        │
│                                                         │
│  3. OPTIMIZE SimulationNN (+ ExoPolicyNN in Exo mode)   │
│     10 epochs of PPO on collected transitions           │
│     (combined actor loss + critic loss + entropy)       │
│                                                         │
│  4. OPTIMIZE MuscleNN                                   │
│     3 epochs of supervised learning                     │
│     (match desired torques via muscle activations)      │
│                                                         │
│  5. EVALUATE & SAVE                                     │
│     Print metrics, save checkpoints                     │
└─────────────────────────────────────────────────────────┘
```

---

## Understanding metadata.txt

The file `MASS/data/metadata.txt` controls the simulation. Here's what each line means:

```
use_muscle true       # Enable muscle-actuated model (vs. direct torque control)
con_hz 30             # Control frequency: policy acts 30 times/second
sim_hz 600            # Simulation frequency: physics runs 600 times/second
skel_file /data/human.xml          # Skeleton definition file
muscle_file /data/muscle284.xml    # Muscle attachment file (284 muscles)
bvh_file /data/motion/walk.bvh true   # Reference motion (walk) + mirror flag
reward_param 0.75 0.1 0.0 0.15    # Reward weights: [pose, velocity, ?, end-effector]
```

> **Warning**: Do not modify `metadata.txt` between training and testing. The networks are trained for a specific configuration. Changing parameters will make saved models incompatible.

---

## Troubleshooting

### General

#### "Cannot connect to the Docker daemon"
Docker Desktop is not running. Open Docker Desktop and wait for it to fully start.

#### Build fails with "no space left on device"
Docker has run out of disk space.
- **Windows / macOS**: Docker Desktop → Settings → Resources. Increase disk image size.
- **Linux**: Run `docker system prune` to clean up unused images/containers.
- All platforms: `docker system prune -a` removes all unused images. This will require a full rebuild.

#### Training crashes with "Segmentation fault"
This should not happen with our patched version. If it does, rebuild the image: `docker compose build --no-cache --progress=plain`.

#### "No module named 'pymss'"
You're running Python outside the correct directory. Make sure you're in `/workspace/MASS/python/` when running `main.py`. The `pymss.so` file is located there.

### Windows-Specific

#### WSL 2 not installed / "WslRegisterDistribution failed"
- Make sure you run `wsl --install` in **Administrator** PowerShell.
- Restart your computer after installation.
- If still failing, enable "Virtual Machine Platform" in Windows Features (Control Panel → Programs → Turn Windows features on or off).

#### "Permission denied" when running build.sh or run.sh
On Windows, the bash scripts may not have execute permission. Use `docker compose` commands directly instead:
```bash
# Instead of ./build.sh:
docker compose build --progress=plain

# Instead of ./run.sh:
mkdir -p nn_output && docker compose up -d
```

#### Docker Desktop won't start / "Hardware assisted virtualization is not enabled"
- Enter BIOS/UEFI (restart and press F2/Del/F12 during boot).
- Enable **Intel VT-x** or **AMD-V** (usually under CPU or Advanced settings).
- Save and restart.

#### Line ending errors ("$'\r': command not found")
Windows may have converted file line endings from LF to CRLF. Fix them:
```bash
sed -i 's/\r$//' docker-entrypoint.sh build.sh run.sh
docker compose build --no-cache --progress=plain
```

#### Slow file I/O / extremely slow build
Make sure your project is cloned **inside WSL** (`~/Coder/DART_MASS`), **not** on the Windows filesystem (`/mnt/c/Users/...`). Cross-filesystem access in WSL is extremely slow.

### macOS-Specific

#### Training is extremely slow (Apple Silicon / M1 / M2 / M3 / M4)
This is expected. The Docker container runs x86_64 code through QEMU emulation. This is ~10 to 50x slower than native. For production training, use a Windows/Linux x86_64 machine or cloud instance.

#### "Permission denied" when running scripts
```bash
chmod +x build.sh run.sh
```

### Linux-Specific

#### "Got permission denied while trying to connect to the Docker daemon socket"
Your user is not in the `docker` group:
```bash
sudo usermod -aG docker $USER
```
**Log out and log back in** for this to take effect.

---

## Further Reading

- [Original Paper (PDF)](http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf)
- [Project Page](http://mrl.snu.ac.kr/research/ProjectScalable/Page.htm)
- [DART Physics Engine](https://dartsim.github.io/)
- [PPO Algorithm (OpenAI)](https://openai.com/research/openai-baselines-ppo)
- [MASS GitHub Repository](https://github.com/lsw9021/MASS)
