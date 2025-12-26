import asyncio
import websockets
import json
import threading
import numpy as np

# Global shared state
latest_data = {
    "layers": [],
    "outputs": [],
    "decision": 0,
    "inputs": []
}
lock = threading.Lock()

def downsample(arr, target_n):
    """Downsample a numpy array to target_n elements."""
    if len(arr) <= target_n:
        return arr.tolist()
    indices = np.linspace(0, len(arr) - 1, target_n, dtype=int)
    return arr[indices].tolist()

async def ws_handler(websocket):
    print("New client connected!")
    try:
        while True:
            with lock:
                data = latest_data
            if data['inputs']:
                await websocket.send(json.dumps(data))
            await asyncio.sleep(0.04) # ~25 FPS updates
    except websockets.exceptions.ConnectionClosed:
        pass

async def start_server():
    print("🚀 WebSocket server starting on ws://localhost:8765")
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future() # Run forever

def run_server_thread():
    asyncio.run(start_server())

def start_background_server():
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    server_thread.start()

def select_meaningful_inputs(arr):
    """Select specific meaningful indices for visualization."""
    # Strict check: 204 is expected. If significantly less, downsample.
    if len(arr) < 200: 
        print(f"⚠️ Input size mismatch: {len(arr)} (Expected 204). Downsampling.")
        return downsample(arr, 17)
    
    # Indices based on mario_env.py structure
    # 0-7: Mario State
    # 8-19: Enemies (sets of 4)
    # 20-201: Grid (13 rows x 14 cols)
    # 202: On Ground
    # 203: Momentum
    
    # Grid Calculation:
    # Index = 20 + row * 14 + col
    # Mario is at Col ~2 (relative to window start 0).
    # Ground is usually Rows 10-11 (0 is top).
    
    indices = [
        0, 1, 2, 3,       # Pos X, Y, Vel X, Y
        4, 5,             # Big, Fire
        8, 9,             # Enemy 1 DX, DY
        12, 13,           # Enemy 2 DX, DY
        202, 203,         # Ground, Momentum
        
        # Grid Samples
        20 + 10*14 + 3,    # Pared Frente (Row 10, Col 3) - 1 tile ahead of feet
        20 + 11*14 + 3,    # Suelo Frente (Row 11, Col 3) - Ground 1 tile ahead
        20 + 11*14 + 5,    # Suelo Medio (Row 11, Col 5)  - Ground 3 tiles ahead
        20 + 12*14 + 4,    # Hueco (Row 12, Col 4)       - Pit check 2 tiles ahead
        20 + 8*14 + 2,     # Bloque Cabeza (Row 8, Col 2) - Directly above Mario
    ]
    
    # Safety: Clamp indices just in case
    safe_indices = [min(i, len(arr)-1) for i in indices]
    
    # Cast to standard float to ensure JSON serialization works
    return [float(arr[i]) for i in safe_indices]

def update_visualization(activations, action):
    global latest_data
    
    vis_input = select_meaningful_inputs(activations['input'])
    vis_h1 = downsample(activations['hidden1'], 30) # Increased density
    vis_h2 = downsample(activations['hidden2'], 30)
    vis_out = activations['output'].tolist()
    
    with lock:
        latest_data = {
            "inputs": vis_input,
            "layers": [vis_h1, vis_h2], 
            "outputs": vis_out,
            "decision": action
        }
