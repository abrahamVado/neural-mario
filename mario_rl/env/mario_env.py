"""Super Mario Bros environment wrapper for RL training."""
from __future__ import annotations
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np


class MarioEnv:
    """Simplified Mario environment for DQN training.
    
    Extracts key features from the game state instead of using raw pixels.
    This makes training faster and easier to debug.
    
    State features (31 total):
        - Mario position (2): x, y normalized
        - Mario velocity (2): vx, vy normalized  
        - Mario state (3): is_big, is_fire, can_shoot
        - Nearby enemies (12): closest 3 enemies (pos_x, pos_y, vx, vy)
        - Obstacles ahead (4): blocks at 4 different distances
        - Gaps/pits (4): pit detection at 4 distances
        - Collectibles (2): coin nearby, powerup nearby
        - Progress (2): x_pos_change, distance_to_flag
        
    Actions (7):
        NOOP, RIGHT, RIGHT+A (jump), RIGHT+B (run),
        RIGHT+A+B (run+jump), A (jump), LEFT
    """
    
    def __init__(self, world: int = 1, stage: int = 1, max_steps: int = 5000, apply_cheats: bool = False):

        # Create Super Mario Bros environment
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        import gym
        
        # Create base environment
        base_env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
        
        # Remove TimeLimit wrapper if present (causes API compatibility issues)
        if isinstance(base_env, gym.wrappers.TimeLimit):
            base_env = base_env.env
        
        # Simplify action space
        self.env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        
        self.max_steps = max_steps
        self.apply_cheats = apply_cheats
        self.current_step = 0
        self.prev_x_pos = 0
        self.max_x_pos = 0

        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.current_step = 0
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.stuck_frames = 0
        self.enemy_history = {} # Track enemy positions for velocity: {slot_id: (x, y)}
        
        obs = self.env.reset()
        
        # 🍄 SUPER POWER INJECTION 🍄
        if self.apply_cheats:
            self.env.unwrapped.ram[0x0756] = 2 # Force Fire Mario
            
        info = self.env.unwrapped._get_info()

        
        return self._extract_features(obs, info)
    
    def step(self, action: int):
        """Take action and return next state, reward, done, info.
        
        Implements Frame Skipping (k=4) to allow for higher jumps and faster training.
        """
        total_reward = 0.0
        done = False
        info = {}
        
        # Repeat action 4 times (frame skipping)
        for _ in range(4):
            obs, reward, done, info = self.env.step(action)
            
            # Custom reward shaping
            r = self._shape_reward(info, done)
            total_reward += r
            
            if done:
                break
                
        self.current_step += 1
        
        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True
        
        state = self._extract_features(obs, info)
        
        return state, total_reward, done, info
    
    def _extract_features(self, obs, info) -> np.ndarray:
        """Extract simplified state features from game observation using RAM."""
        # Access RAM
        ram = self.env.unwrapped.ram
        
        # --- Mario State ---
        mario_x = info.get('x_pos', 0)
        mario_y = info.get('y_pos', 0)
        
        # Normalize positions (typical screen width ~256, height ~240)
        norm_x = (mario_x % 256) / 256.0
        norm_y = mario_y / 240.0

        # Mario Velocity
        m_vx = ram[0x0057]
        m_vy = ram[0x009F]
        if m_vx > 127: m_vx -= 256
        if m_vy > 127: m_vy -= 256
        
        norm_vx = m_vx / 40.0 
        norm_vy = m_vy / 40.0
        
        # Mario status
        status = info.get('status', 'small')
        is_big = 1.0 if status != 'small' else 0.0
        is_fire = 1.0 if status == 'fireball' else 0.0
        can_shoot = is_fire
        
        # Progress
        x_change = (mario_x - self.prev_x_pos) / 10.0
        self.prev_x_pos = mario_x
        self.max_x_pos = max(self.max_x_pos, mario_x)
        
        # --- Enemy Detection (Keep 3 closest) ---
        enemies = []
        current_enemies = {}
        
        for i in range(5):
            if ram[0xF0 + i] != 0:
                ex = ram[0x6E + i]
                ey = ram[0x87 + i]
                
                dx = (ex - (mario_x % 256)) / 256.0
                dy = (ey - mario_y) / 240.0
                
                if dx < -0.5: dx += 1.0
                if dx > 0.5: dx -= 1.0
                
                e_vx = 0.0
                e_vy = 0.0
                if i in self.enemy_history:
                    prev_ex, prev_ey = self.enemy_history[i]
                    d_ex = ex - prev_ex
                    if d_ex < -100: d_ex += 256 
                    if d_ex > 100: d_ex -= 256
                    d_ey = ey - prev_ey
                    e_vx = d_ex / 256.0
                    e_vy = d_ey / 240.0
                    
                current_enemies[i] = (ex, ey)
                
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    enemies.append((dx, dy, e_vx, e_vy))
        
        self.enemy_history = current_enemies
        enemies.sort(key=lambda p: p[0]**2 + p[1]**2)
        while len(enemies) < 3:
            enemies.append((0.0, 0.0, 0.0, 0.0))
            
        # --- FULL SCREEN GRID VISION ---
        # Grid Size: 13 rows (height) x 14 cols (width)
        # Mario is usually at col ~6 on screen.
        # We capture from col -2 (behind) to col +11 (ahead)
        
        m_col = (mario_x // 16)
        
        # Define window relative to Mario's world position
        start_col = m_col - 2
        end_col = m_col + 11  # Inclusive range length = 14
        
        grid_features = []
        
        # Iterate rows 0 (top) to 12 (bottom)
        for row in range(13):
             for col in range(start_col, end_col + 1):
                 tile_val = self._get_tile_at(ram, col, row)
                 
                 # Normalize: 0 = Empty, 1 = Solid, 0.5 = Breakable/Item
                 # Bricks: 0x51-0x58 (Standard breakable)
                 # Question Blocks: 0xC0-0xC2 (Active)
                 is_solid = 0.0
                 if tile_val != 0:
                     if (0x50 <= tile_val <= 0x58) or (0xC0 <= tile_val <= 0xD0):
                         is_solid = 0.5 # Breakable / Interactable target
                     else:
                         is_solid = 1.0 # Hard / Ground
                 
                 grid_features.append(is_solid)
                 
        # Total Grid Inputs: 13 * 14 = 182
        
        # --- Long Jump Helpers ---
        # 0x001D = Float State (00=Stand/Walk, 01=Jump/Air, 02=Climb, 03=Water)
        is_on_ground = 1.0 if ram[0x001D] == 0 else 0.0
        
        # Momentum check (Run speed typically > 40 dec)
        # m_vx is signed
        has_momentum = 1.0 if abs(m_vx) > 40 else 0.0
        
        # Combine all features
        features = [
            norm_x, norm_y, norm_vx, norm_vy,
            is_big, is_fire, can_shoot,
            x_change,
            # Enemies (12)
            *enemies[0], *enemies[1], *enemies[2],
            # Grid (182)
            *grid_features,
            # Long Jump Helpers (Replacing Coin/Powerup placeholders)
            is_on_ground,     # 27: Is on ground?
            has_momentum,     # 28: Has running momentum?
        ]
        
        # Total Dimension: 8 + 12 + 182 = 202
        return np.array(features, dtype=np.float32)

    def _get_tile_at(self, ram, col, row):
        """Read tile from RAM at grid coordinates (col, row).
        Handles page wrapping for the scrolling 0x0500 buffer.
        """
        # Constrain row to valid screen area [0, 12]
        if row < 0: row = 0
        if row > 12: row = 12
        
        # Determine page (0 or 1) based on column
        # Screen width is 16 tiles.
        # The memory layout alternates pages every 16 columns of world space?
        # Actually, SMB maps world x to page index.
        page = (col // 16) % 2
        sub_col = col % 16
        
        # Offset calculation
        # Pages are at 0x500 and 0x6A0 (length 416 = 13*32 bytes? No, 13*16 = 208 bytes)
        # 0x0500 + 208 = 0x05D0.
        # Let's stick effectively to:
        # Page 0: 0x0500
        # Page 1: 0x06A0 ? (Check: 0x500 + 0xD0 = 0x5D0. Where is 0x6A0 come from?)
        # Let's assume standard SMB RAM map:
        # P1: 0x500. P2: 0x5D0. (If contiguous)
        # But some docs say 0x6A0. Let's trust the contagious nature or just check 0x500 base.
        # 13 rows per page. 16 byte per row? (Since 16 cols).
        
        if page == 0:
            base_addr = 0x0500
        else:
            base_addr = 0x05D0 # 0x500 + 13*16 = 0x5D0
            
        offset = base_addr + (row * 16) + sub_col
        
        # Safety check
        if offset >= len(ram): return 0
        
        return ram[offset]

    def _shape_reward(self, info, done) -> float:
        """Custom reward shaping using RAM info."""
        x_pos = info.get('x_pos', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        
        # Reward for moving right
        reward = (x_pos - self.prev_x_pos) * 0.1
        
        # Reward for score increases
        score_diff = score - self.prev_score
        coin_diff = coins - self.prev_coins
        
        if score_diff > 0:
            # Check source of score
            if coin_diff > 0:
                # It's a coin
                reward += score_diff / 20.0 # moderate reward for coins
            elif score_diff == 50:
                # Breaking a block! (Usually 50 pts)
                reward += score_diff / 2.0 # Good Work! +25 reward
                print(f"🧱 SMASH! Bonus +{reward:.1f}")
            else:
                # It's a KILL (or powerup, or block break)
                # Kills usually 100, 200, 400, 500, 800, 1000
                # Give MASSIVE punishment to Goombas by rewarding Mario heavily
                reward += score_diff / 2.0  # 50 reward for a 100pt goomba (huge!)
                print(f"🗡️ KILL/ACTION! +{score_diff} pts. Reward +{score_diff/2.0:.1f}")

        self.prev_score = score
        self.prev_coins = coins
        
        # Stuck penalty
        # If x_pos hasn't changed significantly
        if abs(x_pos - self.prev_x_pos) < 2:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
            
        if self.stuck_frames > 60:
            reward -= 1.0 # Penalty for staying still too long
            
        # FORCE RESTART IF STUCK
        if self.stuck_frames > 250: # ~4 seconds
            done = True
            reward -= 10.0 # Heavy penalty for needing force reset
        
        # Big reward for reaching flag
        if info.get('flag_get', False):
            reward += 500.0
            
        # Penalty for dying
        if done and not info.get('flag_get', False):
            reward -= 25.0
            
        # Encouragement for speed (small negative per step)
        reward -= 0.05
        
        # MOMENTUM BONUS: Encourage holding B (Running)
        # Check if moving fast
        ram = self.env.unwrapped.ram
        m_vx = ram[0x0057]
        if m_vx > 40: # Running speed
            reward += 0.1
            
        return reward

    def render(self):
        """Render the game (for debugging)."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
