# Mario RL - Quick Start Guide

## ✅ Everything is Ready!

**Mario environment**: ✅ Working (25 features)
**Neural network**: ✅ 256x256 neurons
**Visualization**: ✅ Configured
**Python**: ✅ 3.10.9 with all dependencies

---

## 🚀 Start Training

**Double-click this file:**
```
start_training.bat
```

Or run from terminal:
```powershell
cd C:\Users\Abraham\Downloads\mario-rl
.\.venv\Scripts\python.exe train_mario.py
```

---

## 📊 Monitor Progress

**TensorBoard** (see training metrics):
```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```
Then open: http://localhost:6006

**Checkpoints** saved every 50 episodes to: `checkpoints/`

---

## 🎮 Watch Mario Play

After training (or during breaks), watch Mario with the beautiful visualization:

```powershell
.\.venv\Scripts\python.exe -m snake_rl.viewer.agent_viewer --config configs/dqn_mario.yaml --ckpt checkpoints/latest.pt
```

You'll see:
- **Left**: Mario gameplay  
- **Right**: Glowing neural network showing AI decisions

---

## 📁 Project Files

- `mario_env.py` - Mario game wrapper
- `networks.py` - 256x256 Q-network
- `unified_viewer.py` - Visualization
- `train_mario.py` - Training script
- `dqn_mario.yaml` - Configuration

---

## ⏱️ Training Time

- **Quick test**: 100 episodes (~30 min)
- **Good results**: 1000 episodes (~5 hours)
- **Best results**: 5000 episodes (~24 hours)

Mario learns slower than Snake - be patient! 🍄
